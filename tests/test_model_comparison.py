from __future__ import annotations

import numpy as np

from evaluation.model_comparison import compare_paired_video_predictions


def _predictions(scores):
    labels = [0, 0, 1, 1]
    return [
        {
            "video_key": f"m::{index}",
            "label": label,
            "prob_fake": score,
        }
        for index, (label, score) in enumerate(zip(labels, scores))
    ]


def test_paired_bootstrap_is_zero_for_identical_predictions():
    predictions = _predictions([0.1, 0.2, 0.8, 0.9])

    result = compare_paired_video_predictions(
        predictions,
        predictions,
        bootstrap_samples=200,
        seed=17,
    )

    assert result["n_videos"] == 4
    assert result["auc_a"] == 1.0
    assert result["auc_b"] == 1.0
    assert result["delta_auc_b_minus_a"] == 0.0
    assert result["delta_auc_ci95"] == [0.0, 0.0]


def test_comparison_uses_one_percentage_point_gate():
    baseline = _predictions([0.1, 0.4, 0.6, 0.9])
    degraded = _predictions([0.1, 0.7, 0.6, 0.9])

    result = compare_paired_video_predictions(
        baseline,
        degraded,
        bootstrap_samples=200,
        seed=17,
        noninferiority_margin=0.01,
    )

    assert result["delta_auc_b_minus_a"] < -0.01
    assert result["passes_point_estimate_gate"] is False


def test_comparison_rejects_mismatched_video_sets():
    baseline = _predictions([0.1, 0.2, 0.8, 0.9])
    candidate = baseline[:-1]

    with np.testing.assert_raises_regex(ValueError, "video-key sets differ"):
        compare_paired_video_predictions(baseline, candidate, bootstrap_samples=10)
