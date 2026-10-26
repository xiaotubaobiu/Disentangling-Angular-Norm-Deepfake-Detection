from pathlib import Path
import sys
import pandas as pd
import pytest

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]


def fixture_rows(scores):
    return pd.DataFrame({'video_key': ['real::a', 'real::b', 'simswap::c', 'simswap::d'],
        'video_id': list('abcd'), 'method_id': ['real', 'real', 'simswap', 'simswap'],
        'label': [0, 0, 1, 1], 'prob_fake': scores, 'domain': ['ffpp'] * 4,
        'seed': [42] * 4, 'num_frames': [4] * 4})


def test_paired_bootstrap_has_hand_calculated_perfect_ranking_gain():
    from experiment_statistics import paired_bootstrap
    result = paired_bootstrap(fixture_rows([.5] * 4), fixture_rows([0, 0, 1, 1]), 'auroc', n_resamples=50)
    assert result['point'] == result['lower'] == result['upper'] == .5
    assert result['n_videos'] == 4
    assert result['n_resamples'] == 50


def test_bootstrap_rejects_unpaired_video_populations():
    from experiment_statistics import paired_bootstrap
    with pytest.raises(ValueError, match='video'):
        paired_bootstrap(fixture_rows([.5] * 4), fixture_rows([0, 0, 1, 1]).iloc[:-1], 'auroc', n_resamples=10)


def test_detection_metrics_separates_auroc_ba_and_real_recall():
    from experiment_statistics import detection_metrics
    values = detection_metrics(fixture_rows([.6, .7, .8, .9]))
    assert values['auroc'] == 1
    assert values['balanced_accuracy'] == .5
    assert values['real_recall'] == 0
    assert values['fake_recall'] == 1
    assert values['brier'] == pytest.approx(.225)


def semantic_rows():
    rows = fixture_rows([.2, .3, .7, .8])
    rows = pd.concat([rows.assign(variant_id='v1', question_id='q1'),
                      rows.assign(variant_id='v2', question_id='q2')], ignore_index=True)
    return rows.assign(task_id='T2', condition='heldout', question_yes_means='fake',
                       question_pair_id='', method_family=['camera', 'camera', 'swap', 'swap'] * 2,
                       target_semantic=['no', 'no', 'yes', 'yes'] * 2)


@pytest.mark.parametrize('damage', ['duplicate', 'variant', 'frames'])
def test_bootstrap_rejects_misaligned_language_variants(damage):
    from experiment_statistics import paired_bootstrap
    left = semantic_rows()
    right = left.copy()
    if damage == 'duplicate':
        right = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    elif damage == 'variant':
        right.loc[0, 'variant_id'] = 'unexpected'
    else:
        right.loc[0, 'num_frames'] = 1
    with pytest.raises(ValueError, match='duplicate|identity|frame'):
        paired_bootstrap(left, right, 'T2', n_resamples=10)


def test_bootstrap_pairs_reordered_rows_without_counting_variants_as_videos():
    from experiment_statistics import paired_bootstrap
    left = semantic_rows()
    result = paired_bootstrap(left, left.iloc[::-1], 'T2', n_resamples=10)
    assert result['point'] == result['lower'] == result['upper'] == 0
    assert result['n_videos'] == 4
    assert result['baseline_variant_rows'] == 8
