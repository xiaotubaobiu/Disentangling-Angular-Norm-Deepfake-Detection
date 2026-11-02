"""CPU-only D analysis of newly qualified, frozen formal FP32 caches.

This entry point never trains, chooses a recipe, changes a role or reads a
development checkpoint. Saved score exports are independently recomputed.
The 200 calibration subsets and 2000 source-group bootstrap draws are separate
conditional operations, never additional independent training seeds.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

from . import scoring_d as score

CELLS = ['grouping', 'representation', 'aggregation']
TUPLE_COLUMNS = ('member_video_uids', 'member_group_ids', 'precision_history')


def independent_auc(labels, values):
    """Unweighted empirical AUC, using integer twice-concordance counts.

    Separate implementation from the scoring module. Sorting leaves arbitrary
    precision values intact, and only the final integer ratio becomes float.
    """
    y, x = list(labels), list(values)
    if len(y) != len(x) or not y or not set(y) <= {0, 1}:
        raise ValueError('aligned binary labels required')
    if any(not mp.isfinite(v) for v in x):
        raise ValueError('finite ranking values required')
    nr, nf = y.count(0), y.count(1)
    if not nr or not nf:
        raise ValueError('independent AUC requires two classes')
    order = sorted(range(len(x)), key=x.__getitem__)
    numerator, real_below, start = 0, 0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and x[order[end]] == x[order[start]]:
            end += 1
        nreal = sum(y[i] == 0 for i in order[start:end])
        nfake = end - start - nreal
        numerator += nfake * (2 * real_below + nreal)
        real_below += nreal
        start = end
    return numerator / (2 * nr * nf)


def independent_operating(labels, values, threshold):
    y, x = np.asarray(labels), list(values)
    if y.ndim != 1 or len(y) != len(x) or not set(y) <= {0, 1}:
        raise ValueError('aligned binary labels required')
    nr, nf = int((y == 0).sum()), int((y == 1).sum())
    if not nr or not nf or any(not mp.isfinite(v) for v in x):
        raise ValueError('finite two-class operating table required')
    if mp.isnan(threshold):
        raise ValueError('threshold cannot be NaN')
    predicted = np.asarray([v >= threshold for v in x])
    fp, tp = int(((y == 0) & predicted).sum()), int(((y == 1) & predicted).sum())
    return dict(false_positives=fp, true_positives=tp, fpr=fp / nr,
                tpr=tp / nf, pooled_ba=((nr - fp) / nr + tp / nf) / 2)


def training_reference(train, features):
    """Canonical RGB rows carry summed association mass, not eight-row videos."""
    h = np.asarray(features)
    if h.dtype != np.float32 or h.ndim != 2 or len(h) != len(train) or not np.isfinite(h).all():
        raise ValueError('aligned raw FP32 canonical TRAIN features required')
    w = train.sampling_mass.to_numpy(dtype=np.float64)
    if not np.isfinite(w).all() or (w <= 0).any() or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('canonical sampling masses must be positive and sum to one')
    if not set(train.label) <= {0, 1} or not np.isclose(w[train.label.eq(0)].sum(), .5, atol=1e-12, rtol=0):
        raise ValueError('TRAIN real and fake sampling mass must each equal one half')
    radius = np.linalg.norm(h.astype(np.float64), axis=1)
    order = np.argsort(radius, kind='stable')
    index = np.searchsorted(np.cumsum(w[order]), .5, side='left')
    return dict(radii=radius, weights=w, canonical_rows=len(train),
                weighted_radius_median=float(radius[order[index]]), zero_norm_count=int((radius < 1e-12).sum()))


def write_scores(path, videos):
    q = score.precision_safe_score_table(videos)
    for column in TUPLE_COLUMNS:
        if column in q:
            q[column] = q[column].map(lambda v: json.dumps(list(v), ensure_ascii=False))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(path, index=False, float_format='%.17g')


def read_scores(path):
    q = pd.read_csv(path, dtype={'rank_score_decimal': str}, float_precision='round_trip', keep_default_na=False)
    for column in TUPLE_COLUMNS:
        if column in q:
            q[column] = q[column].map(lambda v: tuple(json.loads(v)))
    return score.restore_rank_scores(q)


def _equal(actual, expected, description, tolerance=1e-10):
    if pd.isna(actual) and pd.isna(expected):
        return
    if pd.isna(actual) or pd.isna(expected) or abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f'{description} mismatch: independently {actual!r}, stored {expected!r}')


def verify_saved_metrics(videos, metrics, fixed_methods):
    """Recompute every saved cell's pooled and fixed-method macro AUC."""
    if len(metrics) != 8 or metrics.duplicated(CELLS).any():
        raise ValueError('exactly eight distinct metric cells required')
    out = []
    for row in metrics.to_dict('records'):
        part = videos
        for column in CELLS:
            part = part[part[column].eq(row[column])]
        if part.empty or part.video_uid.duplicated().any():
            raise ValueError('missing or duplicate score rows in saved cell')
        usable = bool(part.ranking_usable.all())
        auc = independent_auc(part.label, part.rank_score) if usable else float('nan')
        _equal(auc, row['pooled_auc'], 'independent AUC')
        macro = float('nan')
        missing = [m for m in fixed_methods if not (part.method_id.eq(m) & part.label.eq(1)).any()]
        valid = usable and row['grouping'] == 'correct' and bool(fixed_methods) and not missing
        if bool(row['macro_valid']) != valid:
            raise ValueError('independent fixed-method macro validity mismatch')
        if valid:
            macro = float(np.mean([independent_auc(part.loc[part.label.eq(0) | part.method_id.eq(m), 'label'],
                part.loc[part.label.eq(0) | part.method_id.eq(m), 'rank_score']) for m in fixed_methods]))
        _equal(macro, row['macro_auc'], 'independent macro AUC')
        out.append(dict(**{c: row[c] for c in CELLS}, ranking_usable=usable,
            pooled_auc_abs_error=abs(auc - row['pooled_auc']) if usable else None,
            macro_auc_abs_error=abs(macro - row['macro_auc']) if valid else None,
            status='PASSED' if usable else 'UNRESOLVED_NUMERIC_ORDERING'))
    return out


def analyze_rank_role(frames, features, W, b, *, training_radii, training_weights, fixed_methods, seed):
    margins = score.frame_margins(frames, features, W, b)
    videos, metrics = score.score_cached_frames(margins, fixed_methods=fixed_methods)
    return dict(frames=margins, videos=videos, metrics=metrics,
        effects=score.d_effects(metrics, seed=seed),
        controls=score.controls_and_diagnostics(margins, training_radii=training_radii, training_weights=training_weights,
                                               fixed_methods=fixed_methods))


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict('records'))
    if isinstance(value, (np.ndarray, pd.Series)):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, mp.mpf):
        return dict(numeric_kind='mpmath', decimal=mp.nstr(value, n=350, strip_zeros=False))
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard_bootstrap_group_support(result, videos):
    """Attach class/method support to both fast normal and joint bootstraps."""
    if 'closure' in result:
        units=videos[['video_uid','label','method_id']].drop_duplicates()
        membership=result['closure'][['video_uid','closure_id']].drop_duplicates()
        units=units.merge(membership,on='video_uid',how='left',validate='one_to_one')
        return score.enforce_bootstrap_group_support(result,units,group_column='closure_id')
    return score.enforce_bootstrap_group_support(result,videos)


def formal_cache_status(config_path):
    config = json.loads(Path(config_path).read_text())
    root = Path(config['artifact_root'])
    tasks = {key: task for key, task in config.get('runs', {}).items()
        if task.get('phase') == 'formal' and task.get('recipe_frozen') is True and task.get('route', 'D') == 'D'}
    ready = []
    for key in tasks:
        marker = root / 'features/D' / key / 'complete.json'
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'CACHE_COMPLETE':
            ready.append(key)
    return dict(status='FORMAL_CACHE_MARKERS_READY' if len(ready) == 15 else 'WAITING_FORMAL_CACHES',
        expected_formal_runs=15, formal_registered=len(tasks), formal_caches_complete=len(ready),
        cache_complete_run_ids=sorted(ready), marker_only=True,
        limitation='Markers are inventory only; analyze_run fully verifies live formal/cache provenance before use.')


def _csv(path, table):
    """Generic tables: preserve tuples/dicts and arbitrary precision thresholds."""
    q = table.copy()
    for column in q:
        if q[column].dtype == object:
            q[column] = q[column].map(lambda v: json.dumps(_jsonable(v), ensure_ascii=False)
                if isinstance(v, (tuple, list, dict, np.ndarray, mp.mpf)) else v)
    q.to_csv(path, index=False, float_format='%.17g')


def _independent_calibration(fixed_test, rows, fixed_methods):
    auc = independent_auc(fixed_test.label, fixed_test.rank_score) if fixed_test.ranking_usable.all() else float('nan')
    have_methods = bool(fixed_methods) and all((fixed_test.method_id.eq(m) & fixed_test.label.eq(1)).any() for m in fixed_methods)
    macro_auc = float('nan')
    if have_methods and bool(fixed_test.ranking_usable.all()):
        macro_auc = float(np.mean([independent_auc(fixed_test.loc[fixed_test.label.eq(0) | fixed_test.method_id.eq(m), 'label'],
            fixed_test.loc[fixed_test.label.eq(0) | fixed_test.method_id.eq(m), 'rank_score']) for m in fixed_methods]))
    audits = []
    for row in rows.to_dict('records'):
        if row.get('status') != 'ok':
            continue
        op = independent_operating(fixed_test.label, fixed_test.score, row['threshold'])
        for key, value in op.items():
            _equal(value, row[key], f'independent calibration {key}')
        _equal(auc, row['pooled_auc'], 'independent calibration AUC')
        mt = []
        for method in fixed_methods:
            fake = fixed_test[fixed_test.label.eq(1) & fixed_test.method_id.eq(method)]
            if fake.empty:
                mt = []; break
            mt.append(float(fake.score.ge(row['threshold']).mean()))
        if mt and bool(fixed_test.ranking_usable.all()):
            _equal(float(np.mean(mt)), row['macro_tpr'], 'independent calibration macro TPR')
            _equal((1 - op['fpr'] + float(np.mean(mt))) / 2, row['macro_ba'], 'independent calibration macro BA')
            _equal(macro_auc, row['macro_auc'], 'independent calibration macro AUC')
        audits.append(dict(calibration_kind=row['calibration_kind'], repeat=row['repeat'], budget=row['budget'],
                           status='PASSED', false_positives=op['false_positives']))
    return audits


def oracle_points(videos, *, fixed_methods=None):
    """Exact actual ROC vertices with one sort; audit only the three winners.

    Repeated full AUC computation at each threshold is unnecessary: thresholds
    alter decisions, while same-pool AUC is constant. No interpolated vertex.
    """
    if not bool(videos.ranking_usable.all()):
        raise ArithmeticError('oracle unavailable until high-precision ordering resolves')
    values, labels = videos.rank_score.tolist(), videos.label.tolist()
    nr, nf = labels.count(0), labels.count(1)
    if not nr or not nf:
        raise ValueError('two-class oracle requires both labels')
    order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
    top = values[order[0]]
    all_real = mp.inf if isinstance(top, mp.mpf) else float(np.nextafter(top, np.inf))
    vertices = [dict(threshold=all_real, fpr=0., tpr=0., pooled_ba=.5)]
    fp, tp, start = 0, 0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        fp += sum(labels[i] == 0 for i in order[start:end])
        tp += sum(labels[i] == 1 for i in order[start:end])
        vertices.append(dict(threshold=values[order[start]], fpr=fp/nr, tpr=tp/nf,
                             pooled_ba=(1-fp/nr+tp/nf)/2))
        start = end
    rows = []
    for name, limit in [('max_ba', None), ('fpr_le_0.05', .05), ('fpr_le_0.10', .10)]:
        candidates = vertices if limit is None else [r for r in vertices if r['fpr'] <= limit + 1e-15]
        keys = ('pooled_ba', 'tpr', 'threshold') if limit is None else ('tpr', 'pooled_ba', 'threshold')
        winner = max(candidates, key=lambda r: tuple(r[k] for k in keys))
        result = score.metric_summary(videos, winner['threshold'], fixed_methods, threshold_score_field='rank_score')
        result.update(criterion=name, n_actual_vertices=len(vertices), diagnostic_only=True)
        rows.append(result)
    return pd.DataFrame(rows)


def analyze_run(config_path, run_id):
    """Production analysis, fixed at 200 CAL subsets and 2000 TEST draws.

    Guard and loaders reject development, stale, partial, or historical caches.
    Formal checkpoint selection uses source DEV. The v2 epoch budget can be
    selected on isolated CDF DEV; its frozen provenance remains explicit.
    """
    from . import cache_d
    from . import bootstrap_fast as boot
    base, config, task = cache_d._context(config_path, run_id)
    root = Path(config['artifact_root'])
    cache_root = root / 'features/D' / run_id
    marker = cache_root / 'complete.json'
    if not marker.is_file() or json.loads(marker.read_text()).get('status') != 'CACHE_COMPLETE':
        raise ValueError('all seven frozen formal cache roles must be complete before analysis')
    freeze_path = root / 'data/D/freeze.json'
    freeze = json.loads(freeze_path.read_text())
    if freeze.get('status') != 'FROZEN' or freeze.get('method_count') != 21:
        raise ValueError('frozen D FS9+FR12 registry is required')
    for role in cache_d.ROLES:
        path = root / 'data/D' / f'{role}.parquet'
        if _digest(path) != freeze['manifest_sha256'].get(path.name):
            raise ValueError('role manifest differs from frozen D registry')
    methods = sorted(freeze['method_registry']['FS9'] + freeze['method_registry']['FR12'])
    output = root / 'analysis/D' / run_id
    identity = dict(cache_context=base, cache_completion_sha256=_digest(marker),
        D_freeze_sha256=_digest(freeze_path), analysis_code_sha256=_digest(__file__),
        scoring_code_sha256=_digest(score.__file__), bootstrap_code_sha256=_digest(boot.__file__),
        calibration_repeats=200, test_bootstrap_repeats=2000, resampling_seed=20261023)
    def expected_role(role):
        return cache_d._role_identity(base, role, root / 'data/D' / f'{role}.parquet')
    complete = output / 'complete.json'
    if complete.is_file():
        prior = json.loads(complete.read_text())
        if prior.get('identity') != identity:
            raise ValueError('existing analysis identity changed; use a separately registered analysis revision')
        for name, digest in prior['artifact_sha256'].items():
            if _digest(output / name) != digest:
                raise ValueError('committed analysis artifact hash mismatch')
        for role in cache_d.ROLES:
            cache_d.verify_cache(cache_root / role, identity=expected_role(role))
        return prior
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('partial analysis identity differs; register a separate analysis revision')
    _save_json(output / 'identity.json', identity)
    cache_d.verify_cache(cache_root / 'dev', identity=expected_role('dev'))
    train, h, W, b = cache_d.load_features(cache_root / 'train', identity=expected_role('train'))
    reference = training_reference(train, h)
    np.save(output / 'training_radii.npy', reference['radii'], allow_pickle=False)
    _save_json(output / 'training_reference.json', {k: v for k, v in reference.items() if k not in ('radii', 'weights')})
    del h, train
    independent_reports = []
    numerical_unresolved = []
    statistical_support = []
    for role in ('source_test', 'target_full'):
        frames, h, w, bias = cache_d.load_features(cache_root / role, identity=expected_role(role))
        if not np.array_equal(w, W) or not np.array_equal(bias, b):
            raise ValueError('cache roles do not share the same original FP32 head')
        result = analyze_rank_role(frames, h, W, b, training_radii=reference['radii'],
            training_weights=reference['weights'], fixed_methods=methods, seed=task['seed'])
        folder = output / role; folder.mkdir(exist_ok=True)
        result['frames'].to_parquet(folder / 'frame_margins.parquet', index=False)
        write_scores(folder / 'video_scores.csv', result['videos'])
        _csv(folder / 'metrics.csv', result['metrics']); _csv(folder / 'effects.csv', result['effects'])
        checks = verify_saved_metrics(read_scores(folder / 'video_scores.csv'),
            pd.read_csv(folder / 'metrics.csv', float_precision='round_trip'), methods)
        independent_reports.append(dict(role=role, cells=checks))
        numerical_unresolved.extend([dict(role=role, **r) for r in checks if r['status'] != 'PASSED'])
        controls = result['controls']
        for row in controls['numerical_audits'].to_dict('records'):
            if not row.get('ranking_usable', True):
                numerical_unresolved.append(dict(role=role, analysis='numerical_control', **row))
        for keys, part in controls['controls'].groupby(['control', 'grouping', 'representation', 'aggregation', 'scale'], sort=False):
            if not bool(part.ranking_usable.all()):
                numerical_unresolved.append(dict(role=role, analysis='control_scores',
                    **dict(zip(('control', 'grouping', 'representation', 'aggregation', 'scale'), keys))))
        write_scores(folder / 'control_video_scores.csv', controls['controls'])
        for name in ('video_diagnostics', 'method_diagnostics', 'numerical_audits'):
            _csv(folder / f'{name}.csv', controls[name])
        _save_json(folder / 'pair_transition_ledgers.json', controls['transition_ledgers'])
        closure=guard_bootstrap_group_support(boot.bootstrap_d_fast(result['frames'], fixed_methods=methods),result['frames'])
        _save_json(folder / 'closure_bootstrap.json', closure)
        statistical_support.append(dict(role=role,analysis='joint_source_collision_closure',
            **{key:closure[key] for key in ('n_real_groups','n_fake_groups','method_fake_group_counts','conditional_interval_stability')}))
        for representation in ('raw', 'l2'):
            for aggregation in ('mean_logit', 'mean_probability'):
                v = result['videos'].query('grouping == "correct" and representation == @representation and aggregation == @aggregation')
                normal=guard_bootstrap_group_support(boot.bootstrap_video_metrics_fast(v, fixed_methods=methods),v)
                _save_json(folder / f'normal_bootstrap_{representation}_{aggregation}.json',
                           normal)
                statistical_support.append(dict(role=role,analysis='correct_video_groups',representation=representation,aggregation=aggregation,
                    **{key:normal[key] for key in ('n_real_groups','n_fake_groups','method_fake_group_counts','conditional_interval_stability')}))
        del result, h, frames
    rq4 = output / 'rq4'; rq4.mkdir(exist_ok=True)
    margins = {}
    for role in ('source_cal', 'target_cal', 'target_test'):
        frames, h, w, bias = cache_d.load_features(cache_root / role, identity=expected_role(role))
        if not np.array_equal(w, W) or not np.array_equal(bias, b):
            raise ValueError('RQ4 head differs between frozen roles')
        margins[role] = score.frame_margins(frames, h, W, b)
        margins[role].to_parquet(rq4 / f'{role}_frame_margins.parquet', index=False)
        del frames, h
    _save_json(rq4 / 'scope.json', freeze['cdf_calibration_scope'])
    for representation in ('raw', 'l2'):
        for aggregation in ('mean_logit', 'mean_probability'):
            folder = rq4 / f'{representation}_{aggregation}'; folder.mkdir(exist_ok=True)
            pools = {role: score.aggregate_videos(q, representation=representation, aggregation=aggregation)
                     for role, q in margins.items()}
            for role, q in pools.items():
                write_scores(folder / f'{role}_scores.csv', q)
            pools = {role: read_scores(folder / f'{role}_scores.csv') for role in pools}
            rows = score.calibration_experiment(pools['source_cal'], pools['target_cal'], pools['target_test'],
                aggregation=aggregation, fixed_methods=methods)
            _csv(folder / 'calibration.csv', rows)
            _save_json(folder / 'independent_calibration_recompute.json',
                       _independent_calibration(pools['target_test'],
                           pd.read_csv(folder / 'calibration.csv', float_precision='round_trip'), methods))
            try:
                oracle = oracle_points(pools['target_test'], fixed_methods=methods)
                _save_json(folder / 'oracle.json', oracle)
                for row in json.loads((folder / 'oracle.json').read_text()):
                    threshold = row['threshold']
                    if isinstance(threshold, dict):
                        with mp.workdps(370):
                            threshold = mp.mpf(threshold['decimal'])
                    op = independent_operating(pools['target_test'].label, pools['target_test'].rank_score, threshold)
                    for key, value in op.items():
                        _equal(value, row[key], f'independent oracle {key}')
            except ArithmeticError as error:
                _save_json(folder / 'oracle.json', dict(status='UNRESOLVED_NUMERIC_ORDERING', reason=str(error)))
                numerical_unresolved.append(dict(role='target_test', representation=representation, aggregation=aggregation))
            source_threshold = rows.loc[rows.calibration_kind.eq('source_real'), 'threshold'].item()
            target_thresholds = rows.loc[rows.calibration_kind.eq('target_real') & rows.budget.eq(20) & rows.status.eq('ok'), 'threshold'].tolist()
            primary = boot.bootstrap_operating_comparison(pools['target_test'], source_threshold=source_threshold,
                target_thresholds=target_thresholds) if target_thresholds else dict(
                    status='UNAVAILABLE_CALIBRATION_BUDGET', calibration_budget=20,
                    available_target_real_videos=len(pools['target_cal']), intervals=None)
            primary=guard_bootstrap_group_support(primary,pools['target_test'])
            _save_json(folder / 'primary_n20_vs_source_bootstrap.json', primary)
            statistical_support.append(dict(role='target_test',analysis='target_calibration_vs_source',representation=representation,aggregation=aggregation,
                **{key:primary[key] for key in ('n_real_groups','n_fake_groups','method_fake_group_counts','conditional_interval_stability')}))
    _save_json(output / 'independent_rank_recomputation.json', independent_reports)
    _save_json(output / 'limitations.json', dict(
        target_scope=freeze['cdf_calibration_scope'], numerical_unresolved=numerical_unresolved,
        target_development_used=bool(base.get('target_development_used',False)),
        formal_trajectory_target_used=False,epoch_recipe_sha256=base.get('epoch_recipe_sha256'),
        development_scope='Isolated CDF DEV may select the frozen epoch budget; the formal best EMA is selected on source DEV only.',
        independent_group_support=statistical_support,
        sparse_group_intervals='Insufficient real/fake or method group support makes affected conditional intervals NA; the minimum is a reporting gate, not a significance guarantee.',
        numerical_score_spaces='Rank AUC/oracle/bootstrap retain FP64 or arbitrary-precision ordering; calibrated deployment decisions use the explicitly frozen FP64 empirical score space.',
        intervals='Pointwise conditional on the frozen model, known source/proxy groups, CAL pool and recorded subsets.',
        calibration_subset_spread='Reported separately in the 200 rows per budget; these are not independent training models.',
        interpretation='Fixed original-head inference interventions; no claim about all classifiers or strict population NP guarantees.'))
    artifacts = {str(p.relative_to(output)): _digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p != complete}
    receipt = dict(status='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' if numerical_unresolved else 'ANALYSIS_COMPLETE',
        utc=datetime.now(timezone.utc).isoformat(), run_id=run_id, seed=task['seed'], backbone=task['backbone'],
        target_development_used=bool(base.get('target_development_used',False)),formal_trajectory_target_used=False,
        epoch_recipe_sha256=base.get('epoch_recipe_sha256'),
        identity=identity, artifact_sha256=artifacts, independent_recompute='PASSED_FOR_ALL_REPORTABLE_CELLS')
    _save_json(complete, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'run'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.command == 'run' and not args.run_id:
        parser.error('run requires --run-id')
    result = formal_cache_status(args.config) if args.command == 'status' else analyze_run(args.config, args.run_id)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
