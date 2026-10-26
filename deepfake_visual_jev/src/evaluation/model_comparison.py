from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def _index_predictions(rows: Iterable[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        key = str(row["video_key"])
        if key in indexed:
            raise ValueError(f"Duplicate video key: {key}")
        indexed[key] = row
    return indexed


def compare_paired_video_predictions(
    baseline_rows: Iterable[dict],
    candidate_rows: Iterable[dict],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 20260921,
    noninferiority_margin: float = 0.01,
) -> dict:
    """Compare video AUROC with a label-stratified paired bootstrap.

    The same resampled video indices are applied to A and B in every replicate.
    Sampling real and fake videos separately prevents undefined single-class AUC
    replicates while retaining the original class counts.
    """
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    baseline = _index_predictions(baseline_rows)
    candidate = _index_predictions(candidate_rows)
    if set(baseline) != set(candidate):
        missing_from_b = sorted(set(baseline) - set(candidate))[:5]
        missing_from_a = sorted(set(candidate) - set(baseline))[:5]
        raise ValueError(
            "A/B video-key sets differ; "
            f"missing_from_b={missing_from_b}, missing_from_a={missing_from_a}"
        )

    keys = sorted(baseline)
    labels = np.asarray([int(baseline[key]["label"]) for key in keys], dtype=np.int64)
    candidate_labels = np.asarray([int(candidate[key]["label"]) for key in keys], dtype=np.int64)
    if not np.array_equal(labels, candidate_labels):
        raise ValueError("A/B labels differ for matched video keys")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Paired AUROC comparison requires both classes")

    scores_a = np.asarray([float(baseline[key]["prob_fake"]) for key in keys], dtype=np.float64)
    scores_b = np.asarray([float(candidate[key]["prob_fake"]) for key in keys], dtype=np.float64)
    auc_a = float(roc_auc_score(labels, scores_a))
    auc_b = float(roc_auc_score(labels, scores_b))
    delta = auc_b - auc_a

    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = np.concatenate((
            rng.choice(class_zero, size=len(class_zero), replace=True),
            rng.choice(class_one, size=len(class_one), replace=True),
        ))
        sampled_labels = labels[sampled]
        bootstrap_deltas[index] = (
            roc_auc_score(sampled_labels, scores_b[sampled])
            - roc_auc_score(sampled_labels, scores_a[sampled])
        )

    ci_low, ci_high = np.quantile(bootstrap_deltas, [0.025, 0.975])
    return {
        "n_videos": int(len(keys)),
        "n_real_videos": int(len(class_zero)),
        "n_fake_videos": int(len(class_one)),
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc_b_minus_a": float(delta),
        "delta_auc_ci95": [float(ci_low), float(ci_high)],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "noninferiority_margin": float(noninferiority_margin),
        "passes_point_estimate_gate": bool(delta >= -noninferiority_margin),
    }
