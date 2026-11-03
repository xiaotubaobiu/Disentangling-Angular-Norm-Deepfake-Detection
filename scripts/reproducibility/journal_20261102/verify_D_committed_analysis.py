"""Verify committed outputs/context; full live input rechecks stay in five-seed summary."""
from pathlib import Path
import argparse
import json
import pandas as pd
from journal_detector import trainer, cache_d, analysis_d, summary_d, scoring_d, bootstrap_fast

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')


def verify(run):
    folder = ROOT / 'analysis/D' / run
    receipt = json.loads((folder / 'complete.json').read_text())
    snapshot = trainer.digest(folder / 'complete.json')
    seed = receipt['seed']
    assert receipt['status'] in ['ANALYSIS_COMPLETE', 'ANALYSIS_COMPLETE_WITH_NUMERIC_NA']
    assert receipt['run_id'] == run and receipt['formal_trajectory_target_used'] is False
    assert receipt['independent_recompute'] == 'PASSED_FOR_ALL_REPORTABLE_CELLS'
    assert set(receipt['artifact_sha256']) == summary_d._input_inventory()
    assert summary_d._entries(folder, ROOT) - {'complete.json'} == summary_d._input_inventory()
    for name, sha in receipt['artifact_sha256'].items():
        assert trainer.digest(folder / name) == sha, name
    base, config, task = cache_d._context(ROOT / 'configs/execution.json', run)
    assert task['seed'] == seed and task['backbone'] == receipt['backbone']
    cache = ROOT / 'features/D' / run
    cache_receipt = json.loads((cache / 'complete.json').read_text())
    assert cache_receipt['status'] == 'CACHE_COMPLETE' and set(cache_receipt['roles']) == set(cache_d.ROLES)
    expected = dict(cache_context=base, cache_completion_sha256=trainer.digest(cache / 'complete.json'),
        D_freeze_sha256=trainer.digest(ROOT / 'data/D/freeze.json'),
        analysis_code_sha256=trainer.digest(analysis_d.__file__), scoring_code_sha256=trainer.digest(scoring_d.__file__),
        bootstrap_code_sha256=trainer.digest(bootstrap_fast.__file__), calibration_repeats=200,
        test_bootstrap_repeats=2000, resampling_seed=20261023)
    assert receipt['identity'] == expected and json.loads((folder / 'identity.json').read_text()) == expected
    for role in cache_d.ROLES:
        path = cache / role / 'complete.json'
        assert trainer.digest(path) == cache_receipt['roles'][role]['complete_sha256']
        assert json.loads(path.read_text())['identity'] == cache_d._role_identity(base, role, ROOT / f'data/D/{role}.parquet')
    results = {}
    for domain in ['source_test', 'target_full']:
        metrics = pd.read_csv(folder / domain / 'metrics.csv', float_precision='round_trip')
        effects = pd.read_csv(folder / domain / 'effects.csv', float_precision='round_trip')
        recomputed = scoring_d.d_effects(metrics, seed=seed).set_index('aggregation')
        saved = effects.set_index('aggregation')
        for agg in saved.index:
            for key in ['delta_correct', 'delta_collision', 'interaction']:
                assert abs(saved.loc[agg, key] - recomputed.loc[agg, key]) < 1e-14
            for key in ['reversal', 'reversal_correct_negative_collision_positive', 'reversal_correct_positive_collision_negative']:
                assert saved.loc[agg, key] == recomputed.loc[agg, key]
        results[domain] = dict(metrics=metrics[['grouping', 'representation', 'aggregation', 'pooled_auc',
            'macro_auc', 'n_real', 'n_fake']].to_dict('records'), effects=effects.to_dict('records'))
    calibration = {}
    for rep in ['raw', 'l2']:
        for agg in ['mean_logit', 'mean_probability']:
            base_flow = folder / 'rq4' / f'{rep}_{agg}'
            table = pd.read_csv(base_flow / 'calibration.csv', float_precision='round_trip')
            checks = json.loads((base_flow / 'independent_calibration_recompute.json').read_text())
            assert len(table) == 802 and table.status.eq('ok').sum() == 602
            assert len(checks) == 602 and all(x['status'] == 'PASSED' for x in checks)
            valid = table[table.status.eq('ok')]
            assert table.test_id_hash.nunique() == 1 and valid.n_real.eq(93).all() and valid.n_fake.eq(12037).all()
            selected = table[(table.calibration_kind == 'target_real') & (table.budget == 20)]
            calibration[f'{rep}_{agg}'] = dict(source=table[table.calibration_kind == 'source_real'][
                ['fpr', 'tpr', 'pooled_ba']].to_dict('records')[0],
                target_n20_mean=selected[['fpr', 'tpr', 'pooled_ba']].mean().to_dict())
    assert trainer.digest(folder / 'complete.json') == snapshot
    record = dict(status='PASS_COMMITTED_OUTPUTS_AND_MODEL_CONTEXT',
        scope='Committed artifacts, current model/config and cache-role headers only; original five-seed summary must still recheck all live raw inputs and cache payloads.',
        run_id=run, complete_sha256=snapshot, artifact_count=len(receipt['artifact_sha256']),
        cache_complete_sha256=expected['cache_completion_sha256'], formal_trajectory_target_used=False,
        target_development_used=receipt['target_development_used'], full_five_seed_summary_complete=False,
        results=results, calibration=calibration)
    # Wrong-group macro is intentionally NA; use the scientific serializer.
    record = analysis_d._jsonable(record)
    trainer.save_json(ROOT / 'operations' / (run + '_committed_analysis_verification.json'), record)
    return dict(status=record['status'], artifact_count=record['artifact_count'],
        effects={domain: item['effects'] for domain, item in results.items()}, calibration=calibration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(analysis_d._jsonable(verify(args.run)), indent=2, allow_nan=False))
