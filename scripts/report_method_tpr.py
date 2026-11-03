"""Derive method TPR from accepted saved FP64 scores and calibration thresholds.

This CPU-only supplement leaves the frozen analysis inventory unchanged.
CAL-subset spread and training-seed variation have separate output columns.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

SEEDS = {7, 42, 123, 2026, 3407}
BACKBONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')
FLOW_KEYS = ['backbone', 'domain', 'representation', 'aggregation']
METHOD_KEYS = ['calibration_kind', 'budget', 'method_id']
REPOSITORY = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_checked_csv(path, expected, **kwargs):
    if digest(path) != expected:
        raise ValueError('saved input hash differs')
    result = pd.read_csv(path, float_precision='round_trip', **kwargs)
    if digest(path) != expected:
        raise ValueError('saved input hash changed during read')
    return result


def method_operating_rows(videos, calibration, methods):
    """Reconstruct every stored operating point before splitting fake methods."""
    required = {'video_uid', 'label', 'method_id', 'score'}
    if not required <= set(videos) or videos.empty or videos.video_uid.duplicated().any():
        raise ValueError('unique natural video scores required')
    y = videos.label.to_numpy()
    values = videos.score.to_numpy(dtype=np.float64)
    if set(y) != {0, 1} or not np.isfinite(values).all():
        raise ValueError('finite two-class scores required')
    if 'weight' in videos and not videos.weight.eq(1).all():
        raise ValueError('method report requires unweighted natural videos')
    methods = list(methods)
    if len(set(methods)) != len(methods) or set(videos.loc[y == 1, 'method_id']) != set(methods):
        raise ValueError('complete fixed method registry required')
    keys = ['calibration_kind', 'budget', 'repeat']
    if not set(keys + ['status']) <= set(calibration) or calibration.empty or calibration.duplicated(keys).any():
        raise ValueError('unique saved calibration points required')
    if not set(calibration.status) <= {'ok', 'unavailable'}:
        raise ValueError('unknown calibration availability')
    valid = calibration.status.eq('ok').to_numpy()
    source = calibration.loc[valid]
    nr, nf = int((y == 0).sum()), int((y == 1).sum())
    tp = np.full((len(calibration), len(methods)), np.nan)
    fpr = np.full(len(calibration), np.nan)
    if len(source):
        if not source.operating_score_field.eq('score').all():
            raise ValueError('saved FP64 operating score field must be score')
        thresholds = source.threshold.to_numpy(dtype=np.float64)
        if not np.isfinite(thresholds).all():
            raise ValueError('finite saved calibration thresholds required')
        real = np.sort(values[y == 0])
        fp = nr - np.searchsorted(real, thresholds, side='left')
        sizes = np.array([int(((y == 1) & videos.method_id.eq(m)).sum()) for m in methods])
        for j, method in enumerate(methods):
            fake = np.sort(values[(y == 1) & videos.method_id.eq(method)])
            tp[valid, j] = len(fake) - np.searchsorted(fake, thresholds, side='left')
        fpr[valid] = fp / nr
        pooled_tpr = tp[valid].sum(axis=1) / nf
        macro_tpr = (tp[valid] / sizes).mean(axis=1)
        expected = dict(n_real=np.full(len(source), nr), n_fake=np.full(len(source), nf),
            false_positives=fp, true_positives=tp[valid].sum(axis=1), fpr=fpr[valid],
            tpr=pooled_tpr, pooled_ba=(1-fpr[valid]+pooled_tpr)/2,
            macro_tpr=macro_tpr, macro_ba=(1-fpr[valid]+macro_tpr)/2)
        for name, want in expected.items():
            saved = pd.to_numeric(source[name], errors='raise').to_numpy(float)
            # A ranking-specific macro gate can be NA even when fixed FP64
            # operating decisions are defined. Never manufacture a macro AUC.
            compare = np.isfinite(saved) if name in ('macro_tpr', 'macro_ba') else np.ones(len(saved), bool)
            if not np.isfinite(saved[compare]).all() or not np.allclose(saved[compare], want[compare], atol=1e-12, rtol=0):
                raise ValueError('saved operating point does not reconstruct: ' + name)
    result = []
    for j, method in enumerate(methods):
        n = int(((y == 1) & videos.method_id.eq(method)).sum())
        q = calibration[keys + ['status']].copy().reset_index(drop=True)
        q['method_id'] = method
        q['n_real'] = nr
        q['n_fake'] = n
        q['true_positives'] = tp[:, j]
        q['tpr'] = tp[:, j] / n
        q['fpr'] = fpr
        q['balanced_accuracy'] = (1 - fpr + q.tpr) / 2
        result.append(q)
    return pd.concat(result, ignore_index=True)


def subset_summary(rows):
    """One summary per model/flow/method; the 200 subsets are not training seeds."""
    groups = [k for k in ['seed'] + FLOW_KEYS if k in rows] + METHOD_KEYS
    result = []
    for values, q in rows.groupby(groups, sort=False, dropna=False):
        tag = dict(zip(groups, values))
        target = tag['calibration_kind'] == 'target_real'
        expected = 200 if target else 1
        if len(q) != expected or target and set(q['repeat']) != set(range(200)):
            raise ValueError('all registered CAL subsets required within each model')
        if q.n_fake.nunique() != 1 or q.n_real.nunique() != 1:
            raise ValueError('method TEST pool changed across subsets')
        valid = q.status.eq('ok') & q.tpr.notna() & q.balanced_accuracy.notna()
        complete = bool(valid.all())
        row = dict(**tag, n_subsets=len(q), n_valid_subsets=int(valid.sum()),
                   complete_subsets=complete, n_real=int(q.n_real.iloc[0]), n_fake=int(q.n_fake.iloc[0]))
        for field, short in [('tpr', 'tpr'), ('balanced_accuracy', 'ba'), ('fpr', 'fpr')]:
            x = q.loc[valid, field].to_numpy(float)
            row[short + '_mean'] = float(x.mean()) if complete else np.nan
            row[short + '_subset_sd'] = float(x.std(ddof=1)) if complete and len(x) > 1 else np.nan
        result.append(row)
    return pd.DataFrame(result)


def five_seed_summary(rows):
    """Use five model-level CAL means; retain NA when any estimate is missing."""
    groups = [k for k in FLOW_KEYS if k in rows] + METHOD_KEYS
    result = []
    for values, q in rows.groupby(groups, sort=False, dropna=False):
        if len(q) != 5 or set(q.seed) != SEEDS or q.seed.duplicated().any():
            raise ValueError('exact five registered training seeds required')
        tag = dict(zip(groups, values))
        for name in ['n_fake', 'n_real']:
            if name in q and q[name].nunique() != 1:
                raise ValueError('different TEST pools between training seeds')
        for field, metric in [('tpr_mean', 'tpr'), ('ba_mean', 'balanced_accuracy')]:
            x = q[field].to_numpy(float)
            if np.isinf(x).any():
                raise ValueError('infinite method operating estimate')
            valid = np.isfinite(x)
            if 'complete_subsets' in q:
                valid &= q.complete_subsets.to_numpy(bool)
            result.append(dict(**tag, metric=metric, n_training_seeds=5,
                n_valid_seeds=int(valid.sum()), complete_five_seeds=bool(valid.all()),
                estimate_mean=float(x.mean()) if valid.all() else np.nan,
                estimate_sample_sd=float(x.std(ddof=1)) if valid.all() else np.nan))
    return pd.DataFrame(result)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _existing(folder, identity):
    path = folder / 'complete.json'
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value['identity'] != identity or any(digest(folder/k) != h for k, h in value['artifact_sha256'].items()):
        raise ValueError('supplementary method report identity/hash differs')
    return value


def report_run(root, run):
    root = Path(root).resolve()
    config_path = root / 'configs/execution.json'
    config = json.loads(config_path.read_text())
    task = config['runs'][run]
    source = root / 'analysis/D' / run
    parent = json.loads((source/'complete.json').read_text())
    freeze = json.loads((root/'data/D/freeze.json').read_text())
    if (task.get('phase') != 'formal' or task.get('route') != 'D' or task['seed'] not in SEEDS
        or parent.get('status') not in ('ANALYSIS_COMPLETE', 'ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
        or parent.get('run_id') != run or parent.get('seed') != task['seed']
        or parent.get('backbone') != task['backbone'] or parent.get('formal_trajectory_target_used') is not False
        or parent.get('independent_recompute') != 'PASSED_FOR_ALL_REPORTABLE_CELLS'
        or parent.get('epoch_recipe_sha256') != task.get('epoch_recipe_sha256')
        or parent['identity']['D_freeze_sha256'] != digest(root/'data/D/freeze.json')):
        raise ValueError('completed registered formal D analysis required')
    methods = sorted(freeze['method_registry']['FS9'] + freeze['method_registry']['FR12'])
    if len(methods) != 21 or len(set(methods)) != 21:
        raise ValueError('fixed21 method registry required')
    inputs = {str(p.relative_to(root)): digest(p) for p in
              [source/'complete.json', config_path, root/'data/D/freeze.json']}
    tables = []
    for representation in ['raw', 'l2']:
        for aggregation in ['mean_logit', 'mean_probability']:
            base = source/'rq4'/f'{representation}_{aggregation}'
            names = [base/'target_test_scores.csv', base/'calibration.csv']
            for path in names:
                relative = str(path.relative_to(source))
                inputs[str(path.relative_to(root))] = parent['artifact_sha256'][relative]
            videos = read_checked_csv(names[0], inputs[str(names[0].relative_to(root))],
                usecols=lambda name: name in {'video_uid', 'label', 'method_id', 'score', 'weight', 'grouping', 'representation', 'aggregation'},
                dtype={'video_uid': str, 'method_id': str})
            cal = read_checked_csv(names[1], inputs[str(names[1].relative_to(root))])
            if (not videos.grouping.eq('correct').all() or not videos.representation.eq(representation).all()
                or not videos.aggregation.eq(aggregation).all() or len(cal) != 802
                or cal.test_id_hash.nunique() != 1
                or cal.test_id_hash.iloc[0] != hashlib.sha256('\n'.join(sorted(videos.video_uid)).encode()).hexdigest()):
                raise ValueError('frozen flow/TEST identity differs')
            table = method_operating_rows(videos, cal, methods)
            tables.append(table.assign(seed=task['seed'], backbone=task['backbone'],
                domain='target_test', representation=representation, aggregation=aggregation))
    rows = pd.concat(tables, ignore_index=True)
    summary = subset_summary(rows)
    identity = dict(schema='D_METHOD_OPERATING_IDENTITY_V1', run_id=run,
        source_analysis_complete_sha256=inputs[str((source/'complete.json').relative_to(root))],
        script_sha256=digest(__file__), input_sha256=inputs, methods=methods,
        decision='saved_FP64_score_greater_or_equal_saved_threshold',
        scope='Method operating-point derivation; original strict summary retains full raw-input/cache verification.')
    folder = root/'reports/D_method_operating'/run
    previous = _existing(folder, identity)
    if previous:
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    _json(folder/'identity.json', identity)
    rows.to_parquet(folder/'per_subset_method_operating.parquet', index=False)
    summary.to_csv(folder/'per_seed_method_operating.csv', index=False, float_format='%.17g')
    for name, expected in inputs.items():
        if digest(root/name) != expected:
            raise ValueError('upstream input changed during method derivation')
    result = dict(schema='D_METHOD_OPERATING_COMPLETE_V1', status='METHOD_OPERATING_COMPLETE',
        identity=identity, method_count=21, per_subset_rows=len(rows), per_seed_rows=len(summary),
        valid_operating_points_reconstructed=int(rows.status.eq('ok').sum()//21),
        artifact_sha256={name:digest(folder/name) for name in
            ['identity.json', 'per_subset_method_operating.parquet', 'per_seed_method_operating.csv']})
    _json(folder/'complete.json', result)
    return result


def report_backbone(root, backbone):
    root = Path(root).resolve()
    original = root/f'analysis/D/{backbone}/five_seed_summary/complete.json'
    complete = json.loads(original.read_text())
    if (complete.get('schema') != 'D_FIVE_SEED_SUMMARY_COMPLETE_V1'
        or complete.get('status') not in ('SUMMARY_COMPLETE', 'SUMMARY_COMPLETE_WITH_NUMERIC_NA')
        or complete.get('backbone') != backbone or complete.get('training_seed_count') != 5
        or set(complete['identity']['seeds']) != SEEDS):
        raise ValueError('original strict five-seed summary required before method aggregation')
    snapshots = {str(original.relative_to(root)): digest(original)}
    for name, expected in complete['artifact_sha256'].items():
        if digest(original.parent/name) != expected:
            raise ValueError('original strict summary artifact hash differs')
    tables = []
    for seed in sorted(SEEDS):
        run = f'D_formal_{backbone}_seed{seed}'
        report = report_run(root, run)
        if report['identity']['source_analysis_complete_sha256'] != complete['identity']['analysis_complete_sha256'][str(seed)]:
            raise ValueError('method and strict summaries reference different models')
        folder = root/'reports/D_method_operating'/run
        tables.append(read_checked_csv(folder/'per_seed_method_operating.csv',
                                      report['artifact_sha256']['per_seed_method_operating.csv']))
        snapshots[str((folder/'complete.json').relative_to(root))] = digest(folder/'complete.json')
    rows = pd.concat(tables, ignore_index=True)
    summary = five_seed_summary(rows)
    identity = dict(schema='D_METHOD_FIVE_SEED_IDENTITY_V1', backbone=backbone,
                    seeds=sorted(SEEDS), input_sha256=snapshots, script_sha256=digest(__file__))
    output = root/f'reports/D_method_operating/{backbone}/five_seed_summary'
    previous = _existing(output, identity)
    if previous:
        return previous
    output.mkdir(parents=True, exist_ok=True)
    _json(output/'identity.json', identity)
    rows.to_csv(output/'per_seed_methods.csv', index=False, float_format='%.17g')
    summary.to_csv(output/'five_seed_methods.csv', index=False, float_format='%.17g')
    for name, expected in snapshots.items():
        if digest(root/name) != expected:
            raise ValueError('method summary source changed during aggregation')
    result = dict(schema='D_METHOD_FIVE_SEED_COMPLETE_V1', status='METHOD_FIVE_SEED_COMPLETE',
        identity=identity, training_seeds=5, summary_rows=len(summary),
        scope='CAL means within each model, then sample SD across five training seeds; no new TEST CI.',
        artifact_sha256={name:digest(output/name) for name in
            ['identity.json', 'per_seed_methods.csv', 'five_seed_methods.csv']})
    _json(output/'complete.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run')
    selection.add_argument('--backbone', choices=BACKBONES)
    args = parser.parse_args()
    result = report_run(args.root, args.run) if args.run else report_backbone(args.root, args.backbone)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
