"""Assemble accepted full-scope tables without changing scientific estimates."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import argparse
import shutil
import itertools
import json
import sys

import numpy as np
import pandas as pd

ROOT = OLD = None
CODE = Path(__file__).resolve().parents[3]
BONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')
sys.path.insert(0, str(CODE / 'scripts/figures'))
from formal_D_figure_data import load_verified as load_d
from formal_H_figure_data import load_verified_roots as load_h


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def revised_model_selection(previous, root, checked_json):
    """Replace both revised H backbones, retaining all other registered models."""
    expected = {f'D_formal_{b}_B_seed{s}' for b, s in itertools.product(BONES, [7,42,123,2026,3407])}
    expected |= {f'H_formal_{b}_{h}_seed{s}' for b, h, s in itertools.product(BONES, ['B','AB'], [7,42,123])}
    normalized = {run.replace('_seed', '_B_seed') if run.startswith('D_') and '_B_seed' not in run else run for run in previous.run_id}
    if len(previous) != 33 or not previous.run_id.is_unique or normalized != expected:
        raise ValueError('complete33 registered models required before replacement')
    revised = ['dinov2_vit_b14', 'efficientnet_b0']
    retained = previous[~(previous.route.eq('H') & previous.backbone.isin(revised))].copy()
    fresh, epochs = [], {}
    for bone, seed, history in itertools.product(revised, [7,42,123], ['B','AB']):
        run_id = f'H_formal_{bone}_{history}_seed{seed}'
        folder = Path(root) / 'runs' / run_id
        complete = checked_json(folder / 'complete.json')
        task = checked_json(folder / 'resolved_config.json')
        trace = checked_json(folder / 'history.json')
        if not (complete['status'] == 'TRAINING_COMPLETE' and
                complete['epochs'] == complete['max_epochs'] == len(trace) == task['epochs'] == 10 and
                task['fixed_epoch_budget'] is True and complete['early_stopped'] is False and
                1 <= complete['best_epoch'] <= 10):
            raise ValueError('fresh fixed10 training required: ' + run_id)
        epochs[bone, seed, history] = complete['best_epoch']
        best = trace[complete['best_epoch'] - 1]
        fresh.append(dict(run_id=run_id, route='H', backbone=bone, history=history, seed=seed,
            lr=task['lr'], max_epochs=10, actual_epochs=10, best_epoch=complete['best_epoch'], early_stopped=False,
            source_own_mean_probability_video_auc=best['dev']['pooled_auc'],
            source_selection_best_score=best['selection']['best_score'], success=complete['success'], amp_skips=complete['skip'],
            formal_trajectory_target_used=complete['target_used'], epoch_recipe_sha256=complete['epoch_recipe_sha256'],
            best_model_sha256=complete['best_sha256']))
    if any(epochs[b,s,'B'] != epochs[b,s,'AB'] for b,s in itertools.product(revised,[7,42,123])):
        raise ValueError('the two histories must use the same paired epoch')
    return pd.concat([retained, pd.DataFrame(fresh)], ignore_index=True).sort_values('run_id').reset_index(drop=True)


def main():
    global ROOT, OLD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--revision-root', type=Path, required=True)
    args = parser.parse_args()
    OLD, ROOT = args.old_root.resolve(), args.revision_root.resolve()
    roots = {'clip_vit_b16': str(OLD), 'dinov2_vit_b14': str(ROOT), 'efficientnet_b0': str(ROOT)}
    budgets = dict.fromkeys(BONES, 10)
    sources = {}

    def checked_csv(path):
        digest = sha(path)
        frame = pd.read_csv(path, float_precision='round_trip')
        assert sha(path) == digest
        sources[str(path.resolve())] = digest
        return frame

    def checked_json(path):
        digest = sha(path)
        record = json.loads(path.read_text())
        assert sha(path) == digest
        sources[str(path.resolve())] = digest
        return record

    # Guard before creating any final output: every original worker and its
    # independent reducer must have accepted all registered training seeds.
    for route, count in [('D', 5), ('H', 3)]:
        for bone in BONES:
            science_root = Path(roots[bone]) if route == 'H' else OLD
            receipt = checked_json(science_root / 'operations' / f'{route}_{bone}_strict_table_acceptance.json')
            expected = ('PASS_STRICT_WORKER_AND_INDEPENDENT_TABLE_REDUCTIONS' if route == 'D'
                else 'PASS_STRICT_H_WORKER_AND_INDEPENDENT_ALL_TABLE_REDUCTIONS')
            assert receipt['status'] == expected
            marker = science_root / f'analysis/{route}/{bone}' / ('five_seed_summary' if route == 'D' else 'three_seed_summary') / 'complete.json'
            assert sha(marker) == receipt['strict_summary_sha256']
            if route == 'D':
                method = OLD / f'reports/D_method_operating/{bone}/five_seed_summary/complete.json'
                assert sha(method) == receipt['method_summary_sha256']
    d, h = load_d(OLD), load_h(roots, epoch_budgets=budgets)
    sources.update({str(OLD / p): v for p, v in d['sources'].items()}); sources.update(h['sources'])
    out = ROOT / 'reports/final_formal_33'
    assert not out.exists(), 'Preserve any existing final-table version'
    frames = {}
    for name in ['metrics', 'effects', 'calibration', 'calibration_comparisons', 'oracles', 'numerical']:
        for level in ['per_seed', 'five_seed']:
            frames[f'D_{level}_{name}.csv'] = pd.concat([
                checked_csv(OLD / f'analysis/D/{bone}/five_seed_summary/{level}_{name}.csv')
                for bone in BONES], ignore_index=True)
    for name in ['metrics', 'comparisons']:
        for level in ['per_seed', 'three_seed']:
            frames[f'H_{level}_{name}.csv'] = pd.concat([
                checked_csv(Path(roots[bone]) / f'analysis/H/{bone}/three_seed_summary/{level}_{name}.csv')
                for bone in BONES], ignore_index=True)
    for level in ['per_seed', 'five_seed']:
        frames[f'D_{level}_methods.csv'] = pd.concat([
            checked_csv(OLD / f'reports/D_method_operating/{bone}/five_seed_summary/{level}_methods.csv')
            for bone in BONES], ignore_index=True)
    old_selection = checked_csv(OLD / 'reports/final_formal_33/formal_model_selection.csv')
    frames['formal_model_selection.csv'] = revised_model_selection(old_selection, ROOT, checked_json)
    assert len(frames['formal_model_selection.csv']) == 33
    assert len(frames['D_per_seed_metrics.csv']) == 240
    assert len(frames['D_per_seed_effects.csv']) == 60
    assert len(frames['H_per_seed_metrics.csv']) == 432
    assert len(frames['H_per_seed_comparisons.csv']) == 3024
    assert len(frames['D_five_seed_methods.csv']) == 3024

    m = frames['D_five_seed_metrics.csv']
    primary = m[m.metric.isin(['pooled_auc', 'macro_auc'])].copy()
    counts = d['metrics'].groupby(['backbone', 'domain', 'grouping', 'representation', 'aggregation'])[['n_real', 'n_fake']].first()
    keys = list(counts.index.names)
    primary = primary.merge(counts.reset_index(), on=keys, how='left', validate='many_to_one')
    primary['mean_percent'] = 100 * primary.estimate_mean
    primary['sample_sd_percent'] = 100 * primary.estimate_sample_sd
    assert primary.n_real.notna().all() and len(primary[primary.metric.eq('pooled_auc')]) == 48
    assert primary[primary.grouping.eq('collision') & primary.metric.eq('macro_auc')].estimate_mean.isna().all()
    frames['D_absolute_AUC_summary.csv'] = primary
    effects = frames['D_five_seed_effects.csv'].copy()
    for effect in ['delta_correct', 'delta_collision', 'interaction']:
        for suffix in ['mean', 'sample_sd']:
            effects[effect + '_' + suffix + '_pp'] = 100 * effects[effect + '_' + suffix]
    assert len(effects) == 12 and effects.n_seeds.eq(5).all()
    frames['D_effects_and_directional_reversals.csv'] = effects
    cal = frames['D_five_seed_calibration.csv']
    index = ['backbone', 'domain', 'representation', 'aggregation', 'budget', 'comparison', 'threshold_kind']
    selected = cal[cal.metric.isin(['pooled_auc', 'fpr', 'tpr', 'pooled_ba'])]
    wide = selected.pivot(index=index, columns='metric', values=['estimate_mean', 'estimate_sample_sd'])
    wide.columns = [metric + '_' + value for value, metric in wide.columns]
    wide = wide.reset_index()
    assert len(wide) == len(selected[index].drop_duplicates())
    frames['D_operating_points_all_budgets_wide.csv'] = wide
    frames['D_target_n20_minus_source.csv'] = frames['D_five_seed_calibration_comparisons.csv'].query('budget == 20').copy()

    hm = frames['H_three_seed_metrics.csv']
    frames['H_primary_AUC_summary.csv'] = hm[hm.metric.eq('pooled_auc') & hm.threshold_kind.eq('default')].copy()
    hc = frames['H_three_seed_comparisons.csv']
    frames['H_primary_paired_effects.csv'] = hc[hc.metric.eq('pooled_auc') & hc.threshold_kind.eq('default')].copy()
    assert len(frames['H_primary_AUC_summary.csv']) == 72
    h_terms = {'all_B': ['H10-H00'], 'common': ['H10-H00', 'H01-H00', 'H11-H10',
        '(H11-H10)-(H01-H00)', 'H11-H01']}
    h_keys = ['backbone', 'domain', 'aggregation', 'cohort', 'comparison']
    expected_h = {(bone, domain, agg, cohort, term)
        for bone, domain, agg in itertools.product(BONES, ['source_test', 'target_test'],
            ['mean_logit', 'mean_probability']) for cohort, terms in h_terms.items() for term in terms}
    actual_h = list(frames['H_primary_paired_effects.csv'][h_keys].itertuples(index=False, name=None))
    assert len(actual_h) == len(set(actual_h)) and set(actual_h) == expected_h

    # Exploratory cross-backbone comparisons retain their own identical video
    # cohort. These are recipe-specific gap changes, not architecture causality.
    base = h['metrics'][h['metrics'].threshold_kind.eq('default')]
    rows = []
    for a, b in itertools.combinations(BONES, 2):
        for domain, aggregation, seed in itertools.product(['source_test', 'target_test'],
                ['mean_logit', 'mean_probability'], [7, 42, 123]):
            for cohort in ['all_B', 'common']:
                q = base[base.domain.eq(domain) & base.aggregation.eq(aggregation)
                    & base.seed.eq(seed) & base.cohort.eq(cohort) & base.backbone.isin([a, b])]
                assert q.groupby('backbone').size().to_dict() == {a: 2 if cohort == 'all_B' else 4, b: 2 if cohort == 'all_B' else 4}
                assert q[['n_real', 'n_fake']].drop_duplicates().shape[0] == 1
                cell = q.pivot(index='backbone', columns='cell', values='pooled_auc')
                cases = [('training_history_at_fixed_B', 'H00', 'H10')] if cohort == 'all_B' else [
                    ('test_input_M1', 'H00', 'H01'), ('test_input_M2', 'H10', 'H11')]
                for effect, first, last in cases:
                    before = float(cell.loc[a, first] - cell.loc[b, first])
                    after = float(cell.loc[a, last] - cell.loc[b, last])
                    rows.append(dict(backbone_a=a, backbone_b=b, domain=domain, aggregation=aggregation,
                        seed=seed, cohort=cohort, effect=effect, first_cell=first, last_cell=last,
                        n_real=int(q.n_real.iloc[0]), n_fake=int(q.n_fake.iloc[0]),
                        gap_before=before, gap_after=after, gap_change=after-before,
                        strict_rank_flip=bool(before * after < 0)))
    cross = pd.DataFrame(rows)
    assert len(cross) == 108
    # Reconstruct the selected contrast from the separately saved paired
    # within-backbone comparisons; do not retain an obsolete15-epoch value.
    witness = cross[cross.backbone_a.eq('clip_vit_b16') & cross.backbone_b.eq('efficientnet_b0')
        & cross.domain.eq('target_test') & cross.aggregation.eq('mean_logit')
        & cross.seed.eq(7) & cross.effect.eq('test_input_M1')]
    independent = h['comparisons']
    independent = independent[independent.domain.eq('target_test') & independent.cohort.eq('common')
        & independent.aggregation.eq('mean_logit') & independent.seed.eq(7)
        & independent.threshold_kind.eq('default') & independent.metric.eq('pooled_auc')
        & independent.comparison.eq('H01-H00')].set_index('backbone')
    assert len(witness) == 1
    np.testing.assert_allclose(witness.gap_change.iloc[0],
        independent.loc['clip_vit_b16', 'estimate'] - independent.loc['efficientnet_b0', 'estimate'], rtol=0, atol=1e-12)
    summaries = []
    group = ['backbone_a', 'backbone_b', 'domain', 'aggregation', 'cohort', 'effect', 'first_cell', 'last_cell', 'n_real', 'n_fake']
    for identity, q in cross.groupby(group, sort=False):
        assert len(q) == 3 and set(q.seed) == {7, 42, 123}
        row = dict(zip(group, identity)); row.update(n_seed_pairs=3, n_strict_rank_flips=int(q.strict_rank_flip.sum()))
        for field in ['gap_before', 'gap_after', 'gap_change']:
            row[field + '_mean_pp'] = 100 * float(q[field].mean())
            row[field + '_sample_sd_pp'] = 100 * float(q[field].std(ddof=1))
        row['scope'] = 'Descriptive registered-recipe comparison on one fixed cohort; not architecture-only causality'
        summaries.append(row)
    frames['H_cross_backbone_gap_changes_per_seed.csv'] = cross
    frames['H_cross_backbone_gap_changes_summary.csv'] = pd.DataFrame(summaries)
    assert len(summaries) == 36

    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    out.mkdir()
    hashes = {}
    for name, frame in frames.items():
        path = out / name
        frame.to_csv(path, index=False, float_format='%.17g')
        if name.startswith('D_'):
            original = OLD / 'reports/final_formal_33' / name
            assert sha(path) == sha(original), 'Unchanged D tables must be byte-identical'
            shutil.copyfile(original, path)
        reread = pd.read_csv(path, float_precision='round_trip')
        # CSV null text and dtype inference do not alter any numeric estimate.
        assert len(reread) == len(frame) and set(reread.columns) == set(frame.columns)
        for column in frame.select_dtypes(include=[np.number]).columns:
            np.testing.assert_allclose(reread[column].to_numpy(float), frame[column].to_numpy(float), rtol=0, atol=0, equal_nan=True)
        hashes[name] = sha(path)
    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    record = dict(schema='FINAL_FORMAL_TABLES_COMPLETE_V1', status='ACCEPTED_FULL_33_MODEL_TABLE_ASSEMBLY',
        created=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), D_models=15, H_models=18,
        input_sha256=sources, output_sha256=hashes, output_row_counts={name:len(frame) for name,frame in frames.items()},
        original_estimates_preserved_exactly=True, all_registered_seeds_retained=True,
        cross_backbone_scope='Exploratory gap/strict-rank changes within identical cohorts and registered recipes',
        cross_backbone_witness_independently_reconstructed=True, training_or_inference=False,
        producer_sha256=sha(__file__), H_roots=roots, expected_H_epoch_budgets=budgets,
        replaced_H_DINO_models=6, replaced_H_EfficientNet_models=6, retained_D_models=15, retained_H_models=6,
        post_initial_results_fixed_budget_revision=True, final_manuscript=False, full_goal_complete=False)
    (out / 'complete.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'operations/final_formal_table_assembly_acceptance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['input_sha256','output_sha256']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
