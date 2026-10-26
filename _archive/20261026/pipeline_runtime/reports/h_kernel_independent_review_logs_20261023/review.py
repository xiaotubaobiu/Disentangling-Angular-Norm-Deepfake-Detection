"""Isolated CPU review; writes only the caller-specified review log."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import mpmath as mp
import numpy as np
import pandas as pd

from reliability_v2 import coverage_h as coverage
from reliability_v2 import stats_h as stats

ROOT = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
METHODS = ('a', 'b')
METRICS = ('pooled_auc', 'macro_auc', 'fpr', 'tpr', 'pooled_ba', 'macro_tpr', 'macro_ba')
COUNTS = ('n_real', 'n_fake', 'false_positives', 'true_positives')
TERMS = {
    'H10-H00': (('H10', 1), ('H00', -1)),
    'H01-H00': (('H01', 1), ('H00', -1)),
    'H11-H10': (('H11', 1), ('H10', -1)),
    '(H11-H10)-(H01-H00)': (('H11', 1), ('H10', -1), ('H01', -1), ('H00', 1)),
    'H11-H01': (('H11', 1), ('H01', -1)),
}


def close(actual, expected):
    if math.isnan(float(expected)):
        assert math.isnan(float(actual)), (actual, expected)
    else:
        assert abs(float(actual) - float(expected)) <= 1e-12, (actual, expected)


def cal():
    base = pd.DataFrame({'video_uid': [f'cal/{i}' for i in range(40)], 'label': 0,
        'method_id': 'real', 'group_id': [f'calgroup/{i}' for i in range(40)],
        'score': np.asarray([-2.] * 20 + [0.] * 17 + [1.] * 3, dtype=np.float64)})
    other = base.copy(deep=True)
    other['score'] = np.asarray([-1.] * 30 + [2.] * 10, dtype=np.float64)
    return {'B': base, 'AB': other}


def cells():
    base = pd.DataFrame({'video_uid': [f'u/{i:02d}' for i in range(12)],
        'label': [0] * 4 + [1] * 8,
        'method_id': ['real'] * 4 + ['a'] * 4 + ['b'] * 4,
        'group_id': [f'g/{i // 2}' for i in range(12)],
        'score': np.asarray([-2., -1., 0., 1., -1., 0., 1., 2., 0., 1., 2., 3.])})
    result = {name: base.copy(deep=True) for name in ('H00', 'H01', 'H10', 'H11')}
    result['H10']['score'] = -base.score
    result['H11']['score'] = np.zeros(12, dtype=np.float64)
    return result


def brute(q, threshold, methods=METHODS):
    y = q.label.to_numpy()
    score = q.score.to_numpy()
    ranks = list(q.rank_score if 'rank_score' in q else q.score)
    r = [v for v, label in zip(ranks, y) if label == 0]
    f = [v for v, label in zip(ranks, y) if label == 1]
    assert r and f
    def auc(reals, fakes):
        return sum(2 if fv > rv else 1 if fv == rv else 0 for fv in fakes for rv in reals) / (2 * len(reals) * len(fakes))
    usable = bool(q.ranking_usable.all()) if 'ranking_usable' in q else True
    predicted = score >= threshold
    fp, tp = int(((y == 0) & predicted).sum()), int(((y == 1) & predicted).sum())
    nr, nf = len(r), len(f)
    present = all((q.method_id == method).any() for method in methods)
    mt = float(np.mean([predicted[q.method_id.to_numpy() == method].mean() for method in methods])) if present else math.nan
    ma = float(np.mean([auc(r, [v for v, method in zip(ranks, q.method_id) if method == m]) for m in methods])) if present and usable else math.nan
    return dict(n_real=nr, n_fake=nf, false_positives=fp, true_positives=tp,
        pooled_auc=auc(r, f) if usable else math.nan, macro_auc=ma,
        fpr=fp/nr, tpr=tp/nf, pooled_ba=(1-fp/nr+tp/nf)/2,
        macro_tpr=mt, macro_ba=(1-fp/nr+mt)/2)


def threshold(x):
    x = np.asarray(x, dtype=np.float64)
    allowed = math.floor(.05 * len(x) + 1e-12)
    candidates = sorted(set(float(np.nextafter(v, np.inf)) for v in x))
    feasible = [t for t in candidates if int((x >= t).sum()) <= allowed]
    return min(feasible)


def stats_explicit_2000_draws():
    q, calibration = cells(), cal()
    out = stats.analyze_cohort(q, calibration, fixed_methods=METHODS, repeats=2000,
        min_groups=1, max_group_fraction=1.)
    tau = {key: threshold(value.score) for key, value in calibration.items()}
    assert out['thresholds'] == tau
    compared = 0
    point = out['point_metrics'].set_index(['cell', 'threshold_kind'])
    for name, cell in q.items():
        for kind in ('default', 'source_real'):
            t = 0. if kind == 'default' else tau['B' if name in ('H00', 'H01') else 'AB']
            expected = brute(cell, t)
            for metric, value in expected.items():
                close(point.loc[(name, kind), metric], value)
                compared += 1
    counts = {'invalid_pooled': 0, 'invalid_macro': 0, 'duplicated_draws': 0,
              'negative_effects': 0, 'zero_effects': 0}
    max_error = 0.
    for record in out['bootstrap']['draws'].to_dict('records'):
        multiplicity = dict(record['group_draw_counts'])
        assert sum(multiplicity.values()) == 6
        counts['duplicated_draws'] += int(max(multiplicity.values()) > 1)
        expanded = {name: pd.concat([cell.loc[cell.group_id.eq(group)]
            for group, times in multiplicity.items() for _ in range(times)], ignore_index=True)
            for name, cell in q.items()}
        if set(expanded['H00'].label) != {0, 1}:
            assert not record['pooled_valid']
            counts['invalid_pooled'] += 1
            counts['invalid_macro'] += 1
            continue
        expected = {}
        missing = any(not expanded['H00'].method_id.eq(m).any() for m in METHODS)
        assert bool(record['macro_operating_valid']) == (not missing)
        assert bool(record['macro_valid']) == (not missing)
        counts['invalid_macro'] += int(missing)
        for name, cell in expanded.items():
            for kind in ('default', 'source_real'):
                t = 0. if kind == 'default' else tau['B' if name in ('H00', 'H01') else 'AB']
                expected[(name, kind)] = brute(cell, t)
                for metric, value in expected[(name, kind)].items():
                    actual = record[f'{kind}:cell:{name}:{metric}']
                    close(actual, value)
                    if math.isfinite(value): max_error = max(max_error, abs(actual-value))
                    compared += 1
        for comparison, expression in TERMS.items():
            for kind in ('default', 'source_real'):
                for metric in METRICS:
                    value = sum(sign * expected[(name, kind)][metric] for name, sign in expression)
                    close(record[f'{kind}:comparison:{comparison}:{metric}'], value)
                    if math.isfinite(value):
                        counts['negative_effects'] += int(value < -1e-10)
                        counts['zero_effects'] += int(abs(value) <= 1e-10)
                    compared += 1
    assert counts['invalid_pooled'] == out['bootstrap']['pooled_invalid_repeats'] > 0
    assert counts['invalid_macro'] == out['bootstrap']['macro_invalid_repeats'] > counts['invalid_pooled']
    assert counts['duplicated_draws'] and counts['negative_effects'] and counts['zero_effects']
    return dict(repeats=2000, explicit_metric_comparisons=compared, max_absolute_error=max_error, **counts)


def stats_precision_and_na():
    q = cells()
    with mp.workdps(120):
        ranks = [mp.mpf(1) + mp.mpf(i) * mp.mpf('1e-90') for i in range(12)]
    for cell in q.values():
        cell['score'] = 1.
        cell['rank_score'] = pd.Series(ranks, dtype=object)
    out = stats.analyze_cohort(q, cal(), fixed_methods=METHODS, repeats=80,
        min_groups=1, max_group_fraction=1.)
    assert out['point_metrics'].pooled_auc.eq(1.).all()
    for record in out['bootstrap']['draws'].to_dict('records'):
        if record['pooled_valid']:
            for name in q:
                assert record[f'default:cell:{name}:pooled_auc'] == 1.
    q['H10']['ranking_usable'] = False
    out = stats.analyze_cohort(q, cal(), fixed_methods=('a', 'b', 'absent'), repeats=80,
        min_groups=1, max_group_fraction=1.)
    point = out['point_metrics']
    assert point.macro_auc.isna().all() and point.macro_tpr.isna().all()
    assert point.loc[point.cell.eq('H10'), 'pooled_auc'].isna().all()
    assert point.loc[point.cell.eq('H10'), 'tpr'].notna().all()
    assert out['bootstrap']['ranking_invalid_repeats'] > 0
    return {'original_precision_auc': 1., 'rank_unresolved_operating_metrics_retained': True,
            'missing_fixed_method_invalidated_macro_without_dropping_rows': True}


def stats_fp64_validation():
    accepted = []
    for dtype in (np.float16, np.float32, np.longdouble, np.complex128):
        part = cells()['H00']
        part['score'] = part.score.astype(dtype)
        try: stats.normalize_video_table(part)
        except ValueError: pass
        else: accepted.append(np.dtype(dtype).name)
    assert not accepted, f'Non-FP64 operating score types accepted: {accepted}'
    return {'float16_float32_extended_float_complex_rejected': True}


def stats_seed_and_gates():
    rows = pd.DataFrame([dict(seed=s, backbone='one', cohort='all_B', aggregation='mean_logit',
        comparison='H10-H00', threshold_kind='default', metric='pooled_auc', estimate=x)
        for s, x in zip((7, 42, 123, 2026, 3407), (-.4, 0., .1, -.1, .2))])
    summary = stats.summarize_seed_comparisons(rows).iloc[0]
    close(summary.estimate_mean, rows.estimate.mean())
    close(summary.estimate_sample_sd, rows.estimate.std(ddof=1))
    for bad in (rows.iloc[:-1], pd.concat([rows, rows.iloc[:1]])):
        try: stats.summarize_seed_comparisons(bad)
        except ValueError: pass
        else: raise AssertionError('missing/duplicate training seeds accepted')
    rows.loc[0, 'estimate'] = np.nan
    assert math.isnan(stats.summarize_seed_comparisons(rows).iloc[0].estimate_mean)
    out = stats.analyze_cohort(cells(), cal(), fixed_methods=METHODS, repeats=3)
    assert out['bootstrap']['intervals'] is None
    concentrated = cells()
    for cell in concentrated.values(): cell['group_id'] = ['big'] * 10 + ['small1', 'small2']
    out = stats.analyze_cohort(concentrated, cal(), fixed_methods=METHODS, repeats=3, min_groups=1)
    assert out['bootstrap']['intervals'] is None
    return {'five_seed_ddof_1': True, 'small_and_concentrated_group_intervals_unavailable': True}


def candidate(i, role, method='real', dataset='FFPP'):
    return {'video_uid': f'{dataset}/{role}/{method}/{i}', 'video_id': str(i), 'dataset': dataset,
        'domain': 'target' if dataset == 'DFD' else 'source', 'method': method,
        'label': int(method != 'real'), 'role': role, 'group_id': f'source/{i}'}


def manifest(video, failures=(), early=False, duplicate=None):
    frames = []
    if not early:
        for i in range(8):
            frame = {'mother_index': i * 3, 'mother_timestamp': i / 7,
                'mother_rgb_sha256': f'{video["video_uid"]}/mother/{i}',
                'original_detection': {'qualified_face_count': 3 if i == 0 else 1, 'multiface': i == 0},
                'a_redetection': {'qualified_face_count': 2 if i == 2 else 1, 'multiface': i == 2}}
            for version, padding in (('A', .2), ('B', .3), ('AB', .4)):
                frame[version] = {'ok': True, 'rgb_sha256': f'{video["video_uid"]}/{version}/{i}',
                    'path': f'/synthetic/{video["video_uid"]}/{version}/{i}.png', 'padding_ratio': padding,
                    'mask_path': f'/synthetic/mask/{video["video_uid"]}/{version}/{i}', 'mask_byte_sha256': 'mask'}
            frames.append(frame)
        for version, i, reason in failures:
            frames[i][version] = {'ok': False, 'reason': reason}
            if version == 'AB': frames[i].pop('a_redetection', None)
        if duplicate:
            frames[1][duplicate]['rgb_sha256'] = frames[0][duplicate]['rgb_sha256']
    eligible = {v: len(frames) == 8 and all(f[v]['ok'] for f in frames)
        and len({f[v]['rgb_sha256'] for f in frames if f[v]['ok']}) == 8 for v in coverage.VERSIONS}
    out = {'video': copy.deepcopy(video), 'frames': frames, 'eligible': eligible,
        'common_B_AB': eligible['B'] and eligible['AB']}
    if early: out['video_failure'] = 'insufficient_unique_mothers'
    else:
        out['version_failures'] = {v: [f'{i}:{f[v]["reason"]}' for i, f in enumerate(frames) if not f[v]['ok']]
            + (['insufficient_unique_final_rgb'] if duplicate == v else []) for v in coverage.VERSIONS}
    return out


def independent_tables(manifests, quarantine):
    tables = {key: [] for key in coverage.TABLE_KEYS}
    for key in tables:
        parts = key.split('/')
        role, version = parts[-1].upper(), parts[-2]
        common = len(parts) == 3 or role in ('TRAIN', 'DEV')
        for item in manifests:
            video = item['video']
            if video['role'] != role or video['video_uid'] in quarantine: continue
            if not (item['common_B_AB'] if common else item['eligible']['B']): continue
            for i, frame in enumerate(item['frames']):
                mid = f'{video["video_uid"]}/mother/{i:03d}/{frame["mother_rgb_sha256"]}'
                tables[key].append(dict(video_uid=video['video_uid'], role=role, method_id=video['method'],
                    label=video['label'], domain=video['domain'], processing_version=version,
                    mother_id=mid, mother_rgb_sha256=frame['mother_rgb_sha256'], rgb_sha256=frame[version]['rgb_sha256'],
                    mother_frame_index=frame['mother_index'], mother_timestamp_seconds=frame['mother_timestamp'],
                    image_path=frame[version]['path'], sample_id=mid, group_id=video['group_id']))
    return tables


def coverage_toy():
    specs = [(0,'SOURCE_TEST','real','FFPP'), (1,'SOURCE_TEST','a','FFPP'), (2,'SOURCE_TEST','b','FFPP'),
        (3,'TARGET_TEST','real','DFD'), (4,'TARGET_TEST','dfd','DFD'), (5,'TRAIN','real','FFPP'),
        (6,'TRAIN','a','FFPP'), (7,'DEV','real','FFPP'), (8,'SOURCE_CAL','real','FFPP'),
        (9,'SOURCE_UNUSED','b','FFPP'), (10,'SOURCE_TEST','a','FFPP'), (11,'TARGET_TEST','dfd','DFD')]
    c = [candidate(*args) for args in specs]
    m = [manifest(v) for v in c]
    for i in (1, 6): m[i] = manifest(c[i], [('AB', 3, 'no_qualified_face')])
    m[2] = manifest(c[2], early=True)
    m[4] = manifest(c[4], [('B', 0, 'crop_failure')])
    m[10] = manifest(c[10], [('A', 1, 'landmarks_degenerate'), ('AB', 1, 'A_failed')])
    m[11] = manifest(c[11], duplicate='AB')
    m[0]['frames'][1].pop('original_detection')
    m[5]['frames'][2]['original_detection'] = {'qualified_face_count': 0, 'multiface': False}
    m[1]['frames'][0]['AB']['padding_ratio'] = .9
    m[0]['frames'][0]['B'].pop('padding_ratio')
    m[0]['frames'][1]['B']['padding_ratio'] = math.nan
    m[0]['frames'][2]['B'].pop('mask_byte_sha256')
    quarantine = {c[3]['video_uid'], c[9]['video_uid']}
    tables = independent_tables(m, quarantine)
    before = copy.deepcopy((c, m, tables))
    out = coverage.summarize_coverage(c, m, active_tables=tables, quarantined_video_uids=quarantine)
    assert repr((c, m, tables)) == repr(before)
    s = out['summary']
    assert s['candidate_videos'] == 12
    assert s['mother_slots'] == {'per_video':8,'prescribed':96,'selected':88,'unselected':8}
    assert {k:v['numerator'] for k,v in s['processing_eligibility'].items()} == {'A':10,'B':10,'AB':7,'common_B_AB':6}
    assert s['administrative_exclusions'] == {'source_unused_videos':1,'quarantined_videos':2,'excluded_union_videos':2,'remaining_candidate_videos':10}
    assert s['available_cohorts']['B']['numerator'] == 8
    assert s['available_cohorts']['common_B_AB']['numerator'] == 4
    for version in coverage.VERSIONS:
        assert s['rejection_reasons'][version]['video_reasons']['insufficient_unique_mothers'] == 1
    assert s['rejection_reasons']['AB']['frame_reasons'] == {'A_failed':1,'no_qualified_face':2}
    assert s['rejection_reasons']['AB']['video_reasons']['insufficient_unique_final_rgb'] == 1
    g = out['geometry_summary']
    d = g['detections']['original_detection']
    assert (d['expected_detection_slots'],d['recorded_detections'],d['missing_without_failure_evidence'],d['invalid_detection_records']) == (88,86,1,1)
    assert d['multiface_frames']['numerator'] == 11 and d['multiface_frames']['denominator'] == 86
    assert g['detections']['a_redetection']['expected_detection_slots'] == 87
    assert g['detections']['a_redetection']['recorded_detections'] == 85
    assert g['detections']['a_redetection']['missing_after_recorded_failure'] == 2
    p = g['padding']['AB']
    assert p['all_successful_frames']['produced_frames'] == 85
    assert p['eligible_video_frames']['produced_frames'] == 56
    assert p['ineligible_video_frames']['produced_frames'] == 29
    assert p['ineligible_video_frames']['max'] == .9
    assert g['padding']['B']['all_successful_frames']['missing_padding_frames'] == 1
    assert g['padding']['B']['all_successful_frames']['invalid_padding_frames'] == 1
    assert g['padding']['B']['all_successful_frames']['mask_metadata']['missing_masks'] == 1
    assert len(out['active_tables']) == 13 and all(v['status']=='VERIFIED' for v in out['active_tables'].values())
    assert out['active_tables']['B/SOURCE_TEST']['natural_videos'] == 3
    assert out['active_tables']['common/B/SOURCE_TEST']['natural_videos'] == 1
    assert out['active_tables']['B/TARGET_TEST']['natural_videos'] == 1
    assert out['active_tables']['common/B/TARGET_TEST']['natural_videos'] == 0
    assert out['active_tables']['B/train']['mother_rows'] == 8
    assert len(out['coverage_by_role_method_label']) == 10
    return {'synthetic_candidates':12,'all_13_active_tables_verified':True,
        'all_B_available':8,'common_available':4,'early_failure_counted_in_all_three_versions':True,
        'administrative_union_separate':True,'missing_invalid_geometry_not_imputed':True,
        'toy_counts_are_not_production_counts':True}


def production_refuses_unfrozen():
    assert list(inspect.signature(coverage.report_coverage).parameters) == ['root']
    assert not (ROOT/'data/H/freeze.json').exists(), 'Production freeze arrived; revise this expectation and rerun'
    try: coverage.report_coverage(ROOT)
    except (OSError, ValueError) as error:
        return {'status':'EXPECTED_REFUSAL','error_type':type(error).__name__,'reason':str(error),
                'no_production_coverage_acceptance_claim':True}
    raise AssertionError('Production coverage accepted unfrozen H data')


def profile_2000():
    rng = np.random.default_rng(20261023)
    base = pd.DataFrame({'video_uid':[f'profile/{i:03d}' for i in range(120)],
        'group_id':[f'profile_group/{i//6:02d}' for i in range(120)],
        'label':[0 if i%6<2 else 1 for i in range(120)],
        'method_id':['real' if i%6<2 else 'a' if i%6<4 else 'b' for i in range(120)]})
    q = {}
    for name in ('H00','H01','H10','H11'):
        q[name] = base.copy()
        q[name]['score'] = rng.normal(size=120)
    start = time.perf_counter()
    out = stats.analyze_cohort(q, cal(), fixed_methods=METHODS, repeats=2000)
    elapsed = time.perf_counter()-start
    assert out['bootstrap']['status']=='ok' and len(out['bootstrap']['draws'])==2000
    return {'groups':20,'natural_videos_per_cell':120,'cells':4,'repeats':2000,'wall_seconds':elapsed,
        'kernel_bootstrap_seconds':out['bootstrap']['bootstrap_seconds'],
        'gpu_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'thread_limit':os.environ.get('OMP_NUM_THREADS')}


def main():
    output = Path(sys.argv[1])
    result = {'schema':'H_KERNEL_ISOLATED_REVIEW_V1','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scope':'synthetic_CPU_kernels_only; production_H_data_unfrozen',
        'checks':[], 'gpu_work_performed':False}
    for name, fn in [('stats_explicit_2000_draws',stats_explicit_2000_draws),
        ('stats_precision_and_na',stats_precision_and_na),('stats_fp64_validation',stats_fp64_validation),
        ('stats_seed_and_gates',stats_seed_and_gates),('coverage_independent_toy',coverage_toy),
        ('production_refuses_unfrozen',production_refuses_unfrozen),('profile_2000',profile_2000)]:
        start=time.perf_counter()
        try: detail=fn(); status='PASS'
        except Exception as error: detail={'error':str(error),'traceback':traceback.format_exc()}; status='FAIL'
        result['checks'].append({'name':name,'status':status,'seconds':time.perf_counter()-start,'detail':detail})
        print(name,status,flush=True)
    paths = [Path(stats.__file__),Path(coverage.__file__),Path(__file__),ROOT/'configs/plan_frozen_input.md']
    for name in ('bootstrap_fast','scoring_d','analysis_d','geometry_h','data_h'):
        paths.append(Path(stats.__file__).parent/f'{name}.py')
    result['code_and_plan_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result['status']='PASS' if all(c['status']=='PASS' for c in result['checks']) else 'FAIL'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),'checks':len(result['checks'])}),flush=True)


if __name__=='__main__': main()
