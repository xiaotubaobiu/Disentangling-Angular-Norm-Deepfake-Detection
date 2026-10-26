"""Video-level paired bootstrap confidence intervals (spec section IX).

Bootstrap unit MUST be video, never frame. We resample (with replacement) from
the set of videos, recompute the metric (AUC by default) for both modes on the
resampled set, take the difference, and report the 2.5 / 97.5 percentile
interval over many resamples.

For paired comparison (direction vs raw on the same checkpoint), the same
resampled video indices are used for both modes — this is essential for the
paired structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auc_safe(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def paired_bootstrap_auc_delta(
    video_df: pd.DataFrame,
    mode_a: str,
    mode_b: str,
    domain: str,
    backbone: str,
    seed: int,
    n_bootstrap: int = 10000,
    rng_seed: int = 42,
) -> dict:
    """Paired bootstrap of AUC(mode_a) - AUC(mode_b) at the video level.

    video_df must contain rows for both mode_a and mode_b for the given
    (backbone, seed, domain). The video set is taken from mode_a (the two
    modes share the same underlying videos since they come from the same
    checkpoint).

    Returns dict with delta_mean, ci_low, ci_high, n_bootstrap, n_videos.
    """
    rng = np.random.default_rng(rng_seed)

    sub_a = video_df[
        (video_df["backbone"] == backbone)
        & (video_df["seed"] == seed)
        & (video_df["domain"] == domain)
        & (video_df["infer_mode"] == mode_a)
    ].sort_values("video_id").reset_index(drop=True)
    sub_b = video_df[
        (video_df["backbone"] == backbone)
        & (video_df["seed"] == seed)
        & (video_df["domain"] == domain)
        & (video_df["infer_mode"] == mode_b)
    ].sort_values("video_id").reset_index(drop=True)

    if sub_a.empty or sub_b.empty:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": n_bootstrap,
            "n_videos": 0,
        }

    # Align by video_id (defensive; should already be aligned since both modes
    # share the same checkpoint and domain).
    merged = sub_a[["video_id", "label", "video_prob_fake"]].merge(
        sub_b[["video_id", "video_prob_fake"]],
        on="video_id",
        suffixes=("_a", "_b"),
    )
    labels = merged["label"].to_numpy()
    scores_a = merged["video_prob_fake_a"].to_numpy()
    scores_b = merged["video_prob_fake_b"].to_numpy()
    n = len(labels)

    if len(np.unique(labels)) < 2:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": n_bootstrap,
            "n_videos": n,
        }

    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        l = labels[idx]
        auc_a = _auc_safe(l, scores_a[idx])
        auc_b = _auc_safe(l, scores_b[idx])
        if np.isnan(auc_a) or np.isnan(auc_b):
            deltas[i] = np.nan
        else:
            deltas[i] = auc_a - auc_b

    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.percentile(valid, 2.5))
        ci_high = float(np.percentile(valid, 97.5))

    return {
        "backbone": backbone,
        "seed": int(seed),
        "domain": domain,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "delta_mean": float(np.nanmean(deltas)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "n_videos": n,
    }


def bootstrap_all_pairs(
    video_df: pd.DataFrame,
    backbones: list[str],
    seeds: list[int],
    domains: list[str],
    mode_pairs: list[tuple[str, str]],
    n_bootstrap: int = 10000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Run paired_bootstrap_auc_delta for every (backbone, seed, domain,
    mode_pair) and stack into a DataFrame."""
    rows = []
    for backbone in backbones:
        for seed in seeds:
            for domain in domains:
                for mode_a, mode_b in mode_pairs:
                    rows.append(
                        paired_bootstrap_auc_delta(
                            video_df=video_df,
                            mode_a=mode_a,
                            mode_b=mode_b,
                            domain=domain,
                            backbone=backbone,
                            seed=seed,
                            n_bootstrap=n_bootstrap,
                            rng_seed=rng_seed,
                        )
                    )
    return pd.DataFrame(rows)
