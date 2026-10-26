"""Finite independent audit of the first D analysis; no pipeline imports."""
from pathlib import Path
from decimal import Decimal
from bisect import bisect_left, bisect_right
import csv
import gc
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
import numpy as np

R = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V = Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
RUN = 'D_formal_efficientnet_b0_seed123'
OUT = R / 'analysis/D' / RUN
REPORT = R / 'reports/D_formal_efficientnet_b0_seed123_terminal_acceptance.json'
checks = 0
max_error = 0.0
started = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, saved):
    global checks, max_error
    checks += 1
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        assert saved is None or saved == '' or str(saved).lower() == 'nan', (actual, saved)
        return
    delta = abs(float(actual) - float(saved))
    assert math.isfinite(delta) and delta <= 1e-10, (actual, saved, delta)
    max_error = max(max_error, delta)


def auc(real, fake):
    real = sorted(real)
    return sum(bisect_left(real, x) + bisect_right(real, x) for x in fake) / (2 * len(real) * len(fake))


def threshold(values):
    values = sorted(values)
    allowed = math.floor(.05 * len(values) + 1e-12)
    return float(np.nextafter(values[len(values) - allowed - 1], np.inf))


assert not REPORT.exists()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
receipt = read(OUT / 'complete.json')
completion_digest = sha(OUT / 'complete.json')
identity = receipt['identity']
assert receipt['status'] in ('ANALYSIS_COMPLETE', 'ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
assert receipt['independent_recompute'] == 'PASSED_FOR_ALL_REPORTABLE_CELLS'
assert identity['calibration_repeats'] == 200 and identity['test_bootstrap_repeats'] == 2000
assert identity['resampling_seed'] == 20261023
inventory = receipt['artifact_sha256']
expected_inventory = {'identity.json', 'training_radii.npy', 'training_reference.json', 'independent_rank_recomputation.json', 'limitations.json', 'rq4/scope.json'}
expected_inventory.update(f'rq4/{role}_frame_margins.parquet' for role in ['source_cal', 'target_cal', 'target_test'])
for role in ['source_test', 'target_full']:
    expected_inventory.update(f'{role}/{name}' for name in ['frame_margins.parquet', 'video_scores.csv', 'metrics.csv', 'effects.csv', 'control_video_scores.csv', 'video_diagnostics.csv', 'method_diagnostics.csv', 'numerical_audits.csv', 'pair_transition_ledgers.json', 'closure_bootstrap.json'])
    expected_inventory.update(f'{role}/normal_bootstrap_{representation}_{aggregation}.json' for representation in ['raw', 'l2'] for aggregation in ['mean_logit', 'mean_probability'])
for flow in ['raw_mean_logit', 'raw_mean_probability', 'l2_mean_logit', 'l2_mean_probability']:
    expected_inventory.update(f'rq4/{flow}/{name}' for name in ['source_cal_scores.csv', 'target_cal_scores.csv', 'target_test_scores.csv', 'calibration.csv', 'independent_calibration_recompute.json', 'oracle.json', 'primary_n20_vs_source_bootstrap.json'])
assert set(inventory) == expected_inventory
actual = {str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file() and p.name != 'complete.json'}
assert actual == set(inventory)
assert not any(p.is_symlink() for p in OUT.rglob('*'))
for name, expected in inventory.items():
    assert sha(OUT / name) == expected, name
print('ARTIFACT_HASHES_PASS', len(inventory), flush=True)
for field, name in [('analysis_code_sha256', 'analysis_d.py'), ('scoring_code_sha256', 'scoring_d.py'), ('bootstrap_code_sha256', 'bootstrap_fast.py')]:
    assert sha(V / 'reliability_v2' / name) == identity[field]
base = identity['cache_context']
assert sha(R / 'configs/execution.json') == base['config_sha256']
assert sha(Path(base['checkpoint_path'])) == base['checkpoint_sha256']
assert sha(R / 'data/D/freeze.json') == identity['D_freeze_sha256']
cache = R / 'features/D' / RUN
assert sha(cache / 'complete.json') == identity['cache_completion_sha256']
marker = read(cache / 'complete.json')
admission = read(R / 'reports/D_formal_efficientnet_b0_seed123_analysis_admission.json')
assert marker['identity'] == base and len(marker['roles']) == len(admission['roles']) == 7
frozen = read(R / 'data/D/freeze.json')
chunk_count = 0
for role, entry in marker['roles'].items():
    folder = cache / role
    assert sha(folder / 'complete.json') == entry['complete_sha256'] == admission['roles'][role]['cache_completion_sha256']
    meta = read(folder / 'complete.json')
    assert sha(R / f'data/D/{role}.parquet') == frozen['manifest_sha256'][f'{role}.parquet'] == meta['cached_manifest_sha256']
    assert sha(folder / 'manifest.parquet') == meta['cached_manifest_sha256']
    assert meta['precision'] == 'FP32' and meta['autocast'] is False
    for item in meta['chunks']:
        assert sha(folder / item['file']) == item['sha256']
        chunk_count += 1
    for key in ['weight', 'bias']:
        assert sha(folder / meta['head'][key + '_file']) == meta['head'][key + '_sha256']
print('SEVEN_CACHE_BOUND_BYTES_PASS', chunk_count, flush=True)
rank = read(R / 'reports/D_formal_efficientnet_b0_seed123_saved_rank_recompute.json')
assert rank['cells'] == 16 and rank['numeric_scalar_checks'] == 128
for path, expected in rank['source_input_sha256'].items():
    assert sha(Path(path)) == expected
methods = sorted(frozen['method_registry']['FS9'] + frozen['method_registry']['FR12'])
calibration_checks = []
common_selection = {}
for flow in ['raw_mean_logit', 'raw_mean_probability', 'l2_mean_logit', 'l2_mean_probability']:
    folder = OUT / 'rq4' / flow
    source = rows(folder / 'source_cal_scores.csv')
    target = rows(folder / 'target_cal_scores.csv')
    test = rows(folder / 'target_test_scores.csv')
    assert all(x['label'] == '0' for x in source + target)
    ids = sorted(x['video_uid'] for x in target)
    assert len(ids) == len(set(ids)) and not set(ids) & {x['video_uid'] for x in test}
    assert not {x['video_uid'] for x in source} & {x['video_uid'] for x in test}
    x = np.array([float(r['score']) for r in test])
    y = np.array([int(r['label']) for r in test])
    method = np.array([r['method_id'] for r in test])
    ranks = [Decimal(r['rank_score_decimal']) for r in test]
    usable = all(r['ranking_usable'] == 'True' for r in test)
    pooled = auc([z for z, label in zip(ranks, y) if label == 0], [z for z, label in zip(ranks, y) if label == 1]) if usable else None
    missing = [m for m in methods if not ((y == 1) & (method == m)).any()]
    macro = sum(auc([z for z, label in zip(ranks, y) if label == 0], [z for z, label, m in zip(ranks, y, method) if label == 1 and m == name]) for name in methods) / len(methods) if usable and not missing else None
    selections = {}
    rng = np.random.default_rng(20261023)
    for repeat in range(200):
        permutation = rng.permutation(len(ids))
        for budget in [10, 20, 40, 80]:
            selections[(repeat, budget)] = [ids[i] for i in permutation[:budget]] if len(ids) >= budget else []
    if common_selection:
        assert selections == common_selection
    common_selection = selections
    table = rows(folder / 'calibration.csv')
    assert len(table) == 802
    seen = set()
    target_scores = {r['video_uid']: float(r['score']) for r in target}
    for row in table:
        kind = row['calibration_kind']
        if kind == 'default':
            expected = 0. if flow.endswith('mean_logit') else .5
        elif kind == 'source_real':
            expected = threshold([float(r['score']) for r in source])
        else:
            assert kind == 'target_real'
            key = (int(row['repeat']), int(row['budget']))
            assert key not in seen
            seen.add(key)
            selected = selections[key]
            assert json.loads(row['calibration_ids']) == selected
            if not selected:
                assert row['status'] != 'ok'
                continue
            expected = threshold([target_scores[i] for i in selected])
            unique = len({tuple(sorted(v)) for (rep, budget), v in selections.items() if budget == key[1] and v})
            close(unique, row['unique_set_count'])
        close(expected, row['threshold'])
        predicted = x >= expected
        fp, tp = int(predicted[y == 0].sum()), int(predicted[y == 1].sum())
        fpr, tpr = fp / int((y == 0).sum()), tp / int((y == 1).sum())
        for field, value in [('false_positives', fp), ('true_positives', tp), ('fpr', fpr), ('tpr', tpr), ('pooled_ba', (1 - fpr + tpr) / 2), ('pooled_auc', pooled), ('macro_auc', macro)]:
            close(value, row[field])
        mt = sum(float(predicted[(y == 1) & (method == m)].mean()) for m in methods) / len(methods) if not missing and usable else None
        close(mt, row['macro_tpr'])
        close((1 - fpr + mt) / 2 if mt is not None else None, row['macro_ba'])
        assert row['test_id_hash'] == hashlib.sha256('\n'.join(sorted(r['video_uid'] for r in test)).encode()).hexdigest()
    assert len(seen) == 800
    native = read(folder / 'independent_calibration_recompute.json')
    assert len(native) == sum(r['status'] == 'ok' for r in table)
    assert all(r['status'] == 'PASSED' for r in native)
    oracle = read(folder / 'oracle.json')
    assert isinstance(oracle, list) and len(oracle) == 3
    for row in oracle:
        raw = row['threshold']
        tau = Decimal(raw['decimal']) if isinstance(raw, dict) else Decimal(str(raw))
        predicted = np.array([v >= tau for v in ranks])
        fp, tp = int(predicted[y == 0].sum()), int(predicted[y == 1].sum())
        for field, value in [('false_positives', fp), ('true_positives', tp), ('fpr', fp / int((y == 0).sum())), ('tpr', tp / int((y == 1).sum()))]:
            close(value, row[field])
        assert row['diagnostic_only'] is True
    calibration_checks.append(dict(flow=flow,rows=len(table),target_subsets=len(seen),source_real_videos=len(source),target_cal_videos=len(target),test_videos=len(test),status='PASS_INDEPENDENT_THRESHOLDS_SELECTIONS_OPERATING_AUC_ORACLE_COUNTS'))
    print('RQ4_PASS', flow, len(table), flush=True)

bootstrap_checks = []
for name in sorted(n for n in inventory if n.endswith('bootstrap.json') or '/normal_bootstrap_' in n):
    obj = read(OUT / name)
    assert obj['repeats'] == 2000 and obj['seed'] == 20261023
    group_key = 'closure_groups' if 'closure_groups' in obj else 'source_groups'
    groups = obj[group_key]
    assert groups == sorted(set(groups))
    G = len(groups)
    lookup = {g: i for i, g in enumerate(groups)}
    draw_rows = obj['draws']
    if not draw_rows:
        assert G < obj['min_groups'] or obj['largest_group_fraction'] > obj['max_group_fraction']
        assert obj['intervals'] is None
        bootstrap_checks.append(dict(file=name,status='PASS_PRESERVED_INSUFFICIENT_GROUPS',groups=G,draws=0))
        continue
    assert len(draw_rows) == 2000
    rng = np.random.default_rng(20261023)
    map_key = 'closure_draw_counts' if group_key == 'closure_groups' else 'group_draw_counts'
    for i, row in enumerate(draw_rows):
        assert row['repeat'] == i
        expected = np.bincount(rng.choice(G, size=G, replace=True), minlength=G)
        observed = np.zeros(G, dtype=np.int64)
        for group, count in row[map_key].items():
            assert group in lookup and isinstance(count, int) and count > 0
            observed[lookup[group]] = count
        assert np.array_equal(observed, expected)
    intervals = obj.get('intervals') or []
    for interval in intervals:
        key = interval.get('metric', interval.get('comparison'))
        values = [row[key] for row in draw_rows if row.get('pooled_valid', True) and row.get(key) is not None]
        assert interval['valid_repeats'] == len(values)
        assert interval['interval_type'] == 'pointwise_conditional_percentile'
        if values:
            close(float(np.quantile(values, .025)), interval['lower'])
            close(float(np.quantile(values, .975)), interval['upper'])
        else:
            close(None, interval['lower']);close(None, interval['upper'])
    bootstrap_checks.append(dict(file=name,status='PASS_ALL_RECORDED_SEEDED_MULTIPLICITIES_AND_INTERVALS',groups=G,draws=len(draw_rows),intervals=len(intervals)))
    print('BOOTSTRAP_PASS', name, G, len(draw_rows), flush=True)
    del obj, draw_rows
    gc.collect()
assert len(bootstrap_checks) == 14
assert sha(OUT / 'complete.json') == completion_digest
native_rank = read(OUT / 'independent_rank_recomputation.json')
assert len(native_rank) == 2 and sum(len(x['cells']) for x in native_rank) == 16
limitations = read(OUT / 'limitations.json')
assert not limitations['numerical_unresolved']
resources = (R / 'reports/D_formal_efficientnet_b0_seed123_analysis_time_v.log').read_text()
assert 'Exit status: 0' in resources and 'Swaps: 0' in resources
assert not Path('/proc/2084956').exists()
peak_kib = int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resources).group(1))
result = dict(utc=datetime.now(timezone.utc).isoformat(),status='PASS_REAL_D_MODEL_COMPLETE_ARTIFACT_CAL_BOOTSTRAP_RESOURCE_ACCEPTANCE',run_id=RUN,completion_sha256=sha(OUT/'complete.json'),artifact_count=len(inventory),artifact_bytes=sum((OUT/n).stat().st_size for n in inventory),seven_cache_roles_verified_by_bound_current_bytes=True,cache_chunks=chunk_count,fresh_source_image_reaudit_repeated=False,saved_rank_cells_independently_verified=16,calibration_checks=calibration_checks,bootstrap_checks=bootstrap_checks,bootstrap_draws_verified=sum(x['draws'] for x in bootstrap_checks),bootstrap_scope='All seeded multiplicities and saved percentile endpoints independently verified. Rank table and CAL/oracle operating values independently recomputed; per-draw bootstrap AUC/effect kernels rely on separately regression-tested implementation, not recomputed here.',numeric_scalar_checks=checks,max_absolute_error=max_error,resource_exit_status=0,resource_peak_RSS_KiB=peak_kib,resource_elapsed='35:56.98',audit_seconds=time.monotonic()-started,all_five_seeds_complete=False,all_experiments_and_papers_complete=False)
REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','artifact_count','artifact_bytes','bootstrap_draws_verified','numeric_scalar_checks','max_absolute_error','resource_peak_RSS_KiB','audit_seconds']}),flush=True)
