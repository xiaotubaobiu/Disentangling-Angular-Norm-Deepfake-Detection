"""Independently reduce saved model estimates after the original strict worker."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')
SEEDS = {7, 42, 123, 2026, 3407}
KEYS = ['backbone', 'domain', 'grouping', 'representation', 'aggregation', 'budget',
        'comparison', 'threshold_kind', 'calibration_kind', 'metric', 'method_id']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def check_reduction(raw, summary):
    checked = 0
    for _, row in summary.iterrows():
        mask = np.ones(len(raw), dtype=bool)
        for key in KEYS:
            if key in raw and key in row:
                mask &= (raw[key].isna() if pd.isna(row[key]) else raw[key].eq(row[key])).to_numpy()
        selected = raw[mask]
        assert len(selected) == 5 and set(selected.seed) == SEEDS and not selected.seed.duplicated().any()
        values = selected['estimate' if 'estimate' in raw else row.metric].to_numpy(dtype=float)
        valid = np.isfinite(values)
        assert valid.sum() in (0, 5), 'unexpected partial-seed availability'
        expected = [values.mean(), values.std(ddof=1)] if valid.all() else [np.nan, np.nan]
        for field, value in zip(['estimate_mean', 'estimate_sample_sd'], expected):
            assert np.isclose(row[field], value, rtol=0, atol=1e-12, equal_nan=True), (field, row.to_dict(), value)
            checked += 1
        if 'n_valid_seeds' in row:
            assert row.n_valid_seeds == int(valid.sum())
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', required=True, choices=['clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0'])
    parser.add_argument('--include-method', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'analysis/D' / args.backbone / 'five_seed_summary'
    marker = folder / 'complete.json'
    original = json.loads(marker.read_text())
    original_sha = sha(marker)
    assert original['schema'] == 'D_FIVE_SEED_SUMMARY_COMPLETE_V1' and original['status'] == 'SUMMARY_COMPLETE'
    assert original['training_seed_count'] == 5
    assert len(original['artifact_sha256']) == 15
    exit_path = ROOT / 'operations/cpu_jobs' / ('D_summary_' + args.backbone) / 'exit.json'
    terminal = json.loads(exit_path.read_text())
    assert terminal['status'] == 'SUCCEEDED' and terminal['exit_code'] == 0
    for name, digest in original['artifact_sha256'].items():
        assert sha(folder / name) == digest, name
    for seed in SEEDS:
        endpoint = ROOT / 'analysis/D' / f'D_formal_{args.backbone}_seed{seed}' / 'complete.json'
        source = json.loads(endpoint.read_text())
        assert source['seed'] == seed and source['backbone'] == args.backbone
        assert source['formal_trajectory_target_used'] is False
        assert len(source['artifact_sha256']) == 65
        for name, digest in source['artifact_sha256'].items():
            assert sha(endpoint.parent / name) == digest, name

    counts = {}
    for name in ['metrics', 'calibration', 'calibration_comparisons', 'oracles']:
        raw_table = read(folder / f'per_seed_{name}.csv')
        if name == 'oracles':
            raw_table = raw_table.rename(columns={'criterion': 'comparison'})
        counts[name] = check_reduction(raw_table, read(folder / f'five_seed_{name}.csv'))
    raw = read(folder / 'per_seed_effects.csv')
    summary = read(folder / 'five_seed_effects.csv')
    for row in summary.itertuples():
        q = raw[raw.backbone.eq(row.backbone) & raw.domain.eq(row.domain) & raw.aggregation.eq(row.aggregation)]
        assert len(q) == 5 and set(q.seed) == SEEDS
        for key in ['delta_correct', 'delta_collision', 'interaction']:
            assert np.isclose(getattr(row, key + '_mean'), q[key].mean(), rtol=0, atol=1e-12)
            assert np.isclose(getattr(row, key + '_sample_sd'), q[key].std(ddof=1), rtol=0, atol=1e-12)
        for key in ['reversal', 'reversal_correct_negative_collision_positive', 'reversal_correct_positive_collision_negative']:
            assert getattr(row, 'n_seed_' + ('reversals' if key == 'reversal' else key.replace('reversal_', 'reversals_'))) == int(q[key].sum())
    counts['effects'] = len(summary) * 6
    method_sha = None
    if args.include_method:
        method = ROOT / 'reports/D_method_operating' / args.backbone / 'five_seed_summary'
        receipt = json.loads((method / 'complete.json').read_text())
        assert receipt['status'] == 'METHOD_FIVE_SEED_COMPLETE'
        for name, digest in receipt['artifact_sha256'].items():
            assert sha(method / name) == digest, name
        wide = read(method / 'per_seed_methods.csv')
        raw_method = wide.melt(id_vars=[c for c in wide if c not in ['tpr_mean', 'ba_mean']],
            value_vars=['tpr_mean', 'ba_mean'], var_name='metric', value_name='estimate')
        raw_method['metric'] = raw_method.metric.map({'tpr_mean': 'tpr', 'ba_mean': 'balanced_accuracy'})
        mean_method = read(method / 'five_seed_methods.csv')
        counts['method'] = check_reduction(raw_method, mean_method)
        method_sha = sha(method / 'complete.json')
    assert sha(marker) == original_sha
    record = dict(status='PASS_STRICT_WORKER_AND_INDEPENDENT_TABLE_REDUCTIONS',
        created=datetime.now(timezone.utc).isoformat(), backbone=args.backbone, training_seeds=sorted(SEEDS),
        strict_summary_sha256=original_sha, strict_worker_exit_sha256=sha(exit_path),
        strict_summary_artifact_count=15, independently_reduced_mean_and_sd_values=counts,
        method_summary_sha256=method_sha, producer_sha256=sha(__file__),
        scope='Original strict worker owns complete live raw/cache validation; this receipt verifies committed outputs and independently recomputes model-summary statistics.',
        full_goal_complete=False)
    out = ROOT / 'operations' / ('D_' + args.backbone + '_strict_table_acceptance.json')
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
