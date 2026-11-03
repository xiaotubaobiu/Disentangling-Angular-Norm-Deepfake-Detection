"""Accept committed H outputs/context without replacing strict live-cache checks."""
from pathlib import Path
import argparse
import json

import pandas as pd

from journal_detector import analysis_h as analysis, cache_h, trainer, stats_h

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', required=True, choices=analysis.BACKBONES)
    parser.add_argument('--seed', required=True, type=int, choices=analysis.SEEDS)
    args = parser.parse_args()
    folder = ROOT / 'analysis/H' / args.backbone / f'seed{args.seed}'
    marker = folder / 'complete.json'
    snapshot = trainer.digest(marker)
    receipt = json.loads(marker.read_text())
    saved_identity = receipt['identity']
    config = Path(saved_identity['cache_contexts']['B']['config_path'])
    contexts, cache_markers = {}, {}
    for history in ['B', 'AB']:
        run = f'H_formal_{args.backbone}_{history}_seed{args.seed}'
        base, _, _ = cache_h._context(config, run)
        contexts[history] = base
        cache = ROOT / 'features/H' / run
        complete = json.loads((cache / 'complete.json').read_text())
        assert complete['status'] == 'CACHE_COMPLETE' and complete['identity'] == base
        assert set(complete['roles']) == set(cache_h.ROLES)
        cache_markers[history + '/complete'] = trainer.digest(cache / 'complete.json')
        for role in cache_h.ROLES:
            path = cache / role / 'complete.json'
            digest = trainer.digest(path)
            assert digest == complete['roles'][role]['complete_sha256']
            assert json.loads(path.read_text())['identity'] == cache_h._role_identity(base, role, ROOT / f'data/H/{role}.parquet')
            cache_markers[history + '/' + role] = digest
    expected = dict(saved_identity, cache_contexts=contexts, cache_completion_sha256=cache_markers,
        config_sha256=trainer.digest(config), data_freeze_sha256=trainer.digest(ROOT / 'data/H/freeze.json'),
        code_sha256=analysis._code_identities())
    assert saved_identity == expected
    assert receipt['backbone'] == args.backbone and receipt['seed'] == args.seed
    assert receipt['formal_trajectory_target_used'] is False and receipt['target_development_used'] is False
    assert analysis._verify_completed(folder, expected) == receipt
    job = ROOT / 'operations/cpu_jobs' / f'H_analysis_{args.backbone}_seed{args.seed}/exit.json'
    terminal = json.loads(job.read_text())
    assert terminal['status'] == 'SUCCEEDED' and terminal['exit_code'] == 0
    metrics, comparisons = [], []
    for aggregation in ['mean_logit', 'mean_probability']:
        cal = {h: analysis.io.read_scores(folder / aggregation / f'source_cal_{h}.csv') for h in ['B', 'AB']}
        for domain, methods in [('source_test', analysis.SOURCE_METHODS), ('target_test', analysis.TARGET_METHODS)]:
            for cohort, cell_names in [('all_B', ['H00', 'H10']), ('common', ['H00', 'H01', 'H10', 'H11'])]:
                base = folder / aggregation / domain / cohort
                cells = {cell: analysis.io.read_scores(base / f'{cell}_scores.csv') for cell in cell_names}
                normalized = stats_h.validate_cohort(cells, fixed_methods=methods)
                thresholds = stats_h._source_thresholds(cal, normalized['H00'], aggregation)[0]
                assert thresholds == json.loads((base / 'thresholds.json').read_text())
                m = pd.read_csv(base / 'metrics.csv', float_precision='round_trip')
                checks = analysis.verify_saved_metrics(normalized, m, thresholds, fixed_methods=methods)
                assert len(checks) == len(m)
                c = pd.read_csv(base / 'comparisons.csv', float_precision='round_trip')
                values = {cell: {kind: q.iloc[0].to_dict() for kind, q in frame.groupby('threshold_kind')}
                    for cell, frame in m.groupby('cell')}
                recomputed = pd.DataFrame(stats_h._comparison_rows(values, stats_h._comparison_terms(cohort == 'common')))
                keys = ['comparison', 'threshold_kind', 'metric']
                actual = c.set_index(keys).estimate
                wanted = recomputed.set_index(keys).estimate
                assert not c.duplicated(keys).any() and set(actual.index) == set(wanted.index)
                for key, value in wanted.items():
                    analysis.io._equal(actual.loc[key], value, 'H saved paired comparison')
                tags = dict(backbone=args.backbone, seed=args.seed, domain=domain, cohort=cohort, aggregation=aggregation)
                metrics.append(m.assign(**tags)); comparisons.append(c.assign(**tags))
    m, c = pd.concat(metrics, ignore_index=True), pd.concat(comparisons, ignore_index=True)
    assert len(m) == 48 and len(c) == 336
    assert trainer.digest(marker) == snapshot
    output = ROOT / 'reports/H_partial_models' / args.backbone / f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [('metrics.csv', m), ('comparisons.csv', c)]:
        frame.to_csv(output / name, index=False, float_format='%.17g')
    record = dict(status='PASS_COMMITTED_OUTPUTS_AND_PAIRED_MODEL_CONTEXT', backbone=args.backbone, seed=args.seed,
        complete_sha256=snapshot, artifact_count=len(receipt['artifact_sha256']), metrics_rows=len(m), comparison_rows=len(c),
        formal_trajectory_target_used=False, target_development_used=False, paired_epoch=receipt['paired_epoch'],
        scope='Committed output hashes, current paired model/data/cache headers and independent saved-score metrics/comparisons. Original strict three-seed summary retains live raw/cache-payload validation.',
        artifact_sha256={name: trainer.digest(output / name) for name in ['metrics.csv', 'comparisons.csv']},
        full_three_seed_summary_claimed=False)
    trainer.save_json(ROOT / 'operations' / f'H_{args.backbone}_seed{args.seed}_committed_analysis_verification.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
