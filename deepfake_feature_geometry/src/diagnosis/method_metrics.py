"""Per-method evaluation.

For FF++ and CDF, for each fake method we form a binary evaluation set:

    positives = fake videos with method_id == method (same domain)
    negatives = real videos in the same domain (method_id == 'real')

and report AUC, balanced accuracy at 0.5, optimal balanced accuracy, optimal
threshold, threshold regret, and the fake / real score distributions.

DFD is treated as a single 'dfd_native' method bucket per spec section VIII:
no per-method split, just the overall DFD metrics.

All thresholding assumes higher score = more fake (which holds for every
inference mode: raw/direction/length all increase with fake-ness, except for
real-predicting models where AUC will simply be < 0.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def balanced_accuracy_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Balanced accuracy when prediction = (score >= threshold)."""
    preds = (scores >= threshold).astype(int)
    return float(balanced_accuracy_score(labels, preds))


def optimal_balanced_threshold_and_acc(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Brute-force sweep over unique scores to find the threshold maximizing
    balanced accuracy.

    Returns (threshold, balanced_accuracy).
    """
    if len(scores) == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(scores)
    best_thr = float(thresholds[0])
    best_ba = -1.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        ba = balanced_accuracy_score(labels, preds)
        if ba > best_ba:
            best_ba = float(ba)
            best_thr = float(thr)
    return best_thr, best_ba


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC; returns NaN if only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def evaluate_method_binary(
    fake_scores: np.ndarray, real_scores: np.ndarray
) -> dict:
    """Compute all per-method metrics for one (fake method, real negatives) set.

    Args:
        fake_scores: video_prob_fake for fake videos of this method.
        real_scores: video_prob_fake for real videos in the same domain.
    """
    labels = np.concatenate([np.ones(len(fake_scores)), np.zeros(len(real_scores))]).astype(int)
    scores = np.concatenate([fake_scores, real_scores]).astype(float)

    auc = _safe_auc(labels, scores)
    ba_05 = balanced_accuracy_at_threshold(labels, scores, 0.5)
    opt_thr, opt_ba = optimal_balanced_threshold_and_acc(labels, scores)
    threshold_regret = opt_ba - ba_05 if not np.isnan(ba_05) else float("nan")

    return {
        "auc_method": auc,
        "ba_at_0.5": ba_05,
        "optimal_ba": opt_ba,
        "optimal_threshold": opt_thr,
        "threshold_regret": threshold_regret,
        "fake_score_mean": float(np.mean(fake_scores)) if len(fake_scores) else float("nan"),
        "fake_score_std": float(np.std(fake_scores)) if len(fake_scores) else float("nan"),
        "real_score_mean": float(np.mean(real_scores)) if len(real_scores) else float("nan"),
        "real_score_std": float(np.std(real_scores)) if len(real_scores) else float("nan"),
        "n_fake": int(len(fake_scores)),
        "n_real": int(len(real_scores)),
    }


def evaluate_all_methods(
    video_df: pd.DataFrame, domain: str
) -> pd.DataFrame:
    """For one domain, evaluate every fake method against the domain's reals.

    Iterates over each (checkpoint_id, infer_mode) combination present in
    video_df, then over each fake method_id in that domain. For DFD the only
    fake method is 'dfd_native'.

    Returns a DataFrame with one row per (checkpoint, infer_mode, domain,
    method_id). Length-only AUC is filled only for the length infer_mode and
    NaN otherwise — callers can drop the column when not relevant.
    """
    rows: list[dict] = []
    domain_df = video_df[video_df["domain"] == domain]
    if domain_df.empty:
        return pd.DataFrame()

    for (ckpt, infer_mode), grp in domain_df.groupby(
        ["checkpoint_id", "infer_mode"], sort=False
    ):
        # Real negatives within this (checkpoint, infer_mode) group only.
        reals = grp[grp["method_id"] == "real"]
        real_scores = reals["video_prob_fake"].to_numpy()
        # Tag columns for output.
        tag = {
            "checkpoint_id": ckpt,
            "backbone": grp["backbone"].iloc[0],
            "seed": int(grp["seed"].iloc[0]),
            "train_mode": grp["train_mode"].iloc[0],
            "infer_mode": infer_mode,
            "domain": domain,
        }

        # Each fake method_id in this group becomes one row.
        fake_methods = sorted(
            m for m in grp["method_id"].unique() if m != "real"
        )
        for method in fake_methods:
            fake_scores = grp[grp["method_id"] == method]["video_prob_fake"].to_numpy()
            metrics = evaluate_method_binary(fake_scores, real_scores)
            rows.append({**tag, "method_id": method, **metrics})

        # Also produce an "all_fake" summary row (all fakes pooled against
        # the same reals) for headline numbers.
        all_fake_scores = grp[grp["method_id"] != "real"]["video_prob_fake"].to_numpy()
        all_metrics = evaluate_method_binary(all_fake_scores, real_scores)
        rows.append({**tag, "method_id": "all_fake", **all_metrics})

    out = pd.DataFrame(rows)
    return out
