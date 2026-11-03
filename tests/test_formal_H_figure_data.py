"""Preprocessing figures must preserve cohorts and matched training pairs."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/figures/formal_H_figure_data.py'
    assert path.is_file(), 'Verified H figure data loader is not implemented'
    spec = importlib.util.spec_from_file_location('formal_H_figure_data', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def tables():
    """Abstract test values only; these are never used to render research figures."""
    rows, comparisons = [], []
    for si, seed in enumerate([7, 42, 123]):
        for domain, aggregation in itertools.product(['source_test', 'target_test'], ['mean_logit', 'mean_probability']):
            shift = .02 * si - (.08 if domain == 'target_test' else 0)
            for cohort in ['all_B', 'common']:
                values = (dict(H00=.70 + shift, H10=.70 + shift + [-.01, 0, .01][si]) if cohort == 'all_B'
                    else dict(H00=.60 + shift, H01=.58 + shift, H10=.61 + shift, H11=.62 + shift + si * .005))
                counts = {('source_test', 'all_B'): (139, 558), ('source_test', 'common'): (136, 543),
                    ('target_test', 'all_B'): (162, 1977), ('target_test', 'common'): (90, 1271)}[domain, cohort]
                for cell, auc in values.items():
                    for threshold_kind in ['default', 'source_real']:
                        rows.append(dict(backbone='efficientnet_b0', seed=seed, domain=domain, cohort=cohort,
                            aggregation=aggregation, cell=cell, threshold_kind=threshold_kind,
                            model_history='B' if cell in ['H00', 'H01'] else 'AB',
                            pooled_auc=auc, ranking_usable=True, n_real=counts[0], n_fake=counts[1]))
                comparison = 'H10-H00' if cohort == 'all_B' else '(H11-H10)-(H01-H00)'
                estimate = (values['H10'] - values['H00'] if cohort == 'all_B'
                    else (values['H11'] - values['H10']) - (values['H01'] - values['H00']))
                comparisons.append(dict(backbone='efficientnet_b0', seed=seed, domain=domain, cohort=cohort,
                    aggregation=aggregation, comparison=comparison, threshold_kind='default',
                    metric='pooled_auc', estimate=estimate))
    metrics, paired = pd.DataFrame(rows), pd.DataFrame(comparisons)
    keys = [c for c in paired if c not in ['seed', 'estimate']]
    official = paired.groupby(keys, as_index=False).estimate.agg(estimate_mean='mean', estimate_sample_sd='std')
    official['complete_three_seeds'] = True
    official['n_valid_seeds'] = 3
    return metrics, paired, official


def test_primary_contrasts_use_their_own_cohort_and_paired_sd():
    m = module()
    metrics, comparisons, official = tables()
    result = m.validate_tables(metrics, comparisons, official, 'efficientnet_b0')
    assert len(result) == 24
    source = result[result.domain.eq('source_test') & result.aggregation.eq('mean_logit')]
    fixed = source[source.effect.eq('fixed_B')].estimate.to_numpy()
    interaction = source[source.effect.eq('interaction')].estimate.to_numpy()
    assert fixed.mean() == pytest.approx(0., abs=1e-12)
    assert fixed.std(ddof=1) == pytest.approx(.01)
    assert interaction.mean() == pytest.approx(.035)
    assert interaction.std(ddof=1) == pytest.approx(.005)
    assert set(result.seed) == {7, 42, 123}


def test_missing_model_seed_is_not_silently_averaged():
    m = module()
    metrics, comparisons, official = tables()
    with pytest.raises(ValueError, match='complete matched'):
        m.validate_tables(metrics[metrics.seed.ne(7)], comparisons, official, 'efficientnet_b0')


def test_replacing_a_setting_with_a_duplicate_is_rejected():
    m = module()
    metrics, comparisons, official = tables()
    metrics.iloc[-1] = metrics.iloc[0]
    with pytest.raises(ValueError, match='complete matched'):
        m.validate_tables(metrics, comparisons, official, 'efficientnet_b0')


def test_all_B_and_common_video_counts_cannot_be_exchanged():
    m = module()
    metrics, comparisons, official = tables()
    metrics.loc[0, 'n_real'] = 136
    with pytest.raises(ValueError, match='cohort'):
        m.validate_tables(metrics, comparisons, official, 'efficientnet_b0')


def test_saved_comparison_must_reconstruct_from_same_cohort_cells():
    m = module()
    metrics, comparisons, official = tables()
    comparisons.loc[0, 'estimate'] = .25
    with pytest.raises(ValueError, match='reconstruct'):
        m.validate_tables(metrics, comparisons, official, 'efficientnet_b0')


def test_official_error_bar_must_match_three_paired_differences():
    m = module()
    metrics, comparisons, official = tables()
    official.loc[0, 'estimate_sample_sd'] = .2
    with pytest.raises(ValueError, match='official'):
        m.validate_tables(metrics, comparisons, official, 'efficientnet_b0')


def test_numeric_NA_cannot_drop_a_model_from_the_figure():
    m = module()
    metrics, comparisons, official = tables()
    metrics.loc[0, 'pooled_auc'] = np.nan
    with pytest.raises(ValueError, match='reportable'):
        m.validate_tables(metrics, comparisons, official, 'efficientnet_b0')


def test_three_finished_model_markers_do_not_replace_the_strict_summary(tmp_path):
    m = module()
    with pytest.raises(ValueError, match='strict three-seed summary'):
        m.load_verified(tmp_path)
