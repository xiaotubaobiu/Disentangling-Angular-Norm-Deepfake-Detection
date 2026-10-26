"""Diagnostic statistics (spec section VII).

All six required diagnostics plus the feature-norm analysis. Each statistic is
computed per-seed first, then summarized as mean ± std over the 5 seeds (per
spec section IX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Per-seed primitives
# ---------------------------------------------------------------------------

def _per_method_mean_fake_score(
    method_df: pd.DataFrame, domain: str
) -> pd.Series:
    """For one (checkpoint, infer_mode, domain), return a Series indexed by
    method_id giving mean fake score (excluding 'all_fake' and 'real')."""
    sub = method_df[
        (method_df["domain"] == domain)
        & (method_df["method_id"].isin(method_df["method_id"].unique()))
        & (method_df["method_id"] != "all_fake")
    ]
    # method_metrics already reports fake_score_mean per method; reuse it.
    return sub.set_index("method_id")["fake_score_mean"]


def d_domain_shift(method_df: pd.DataFrame, infer_mode: str) -> float:
    """D_domain (spec VII.1): mean over methods of |Δ mean_fake_score CDF-FF++|.

    Computed for one checkpoint/seed.
    """
    sub = method_df[
        (method_df["infer_mode"] == infer_mode)
        & (method_df["method_id"] != "all_fake")
    ]
    ff = sub[sub["domain"] == "ffpp"].set_index("method_id")["fake_score_mean"]
    cdf = sub[sub["domain"] == "cdf"].set_index("method_id")["fake_score_mean"]
    common = ff.index.intersection(cdf.index)
    if len(common) == 0:
        return float("nan")
    delta = (cdf.loc[common] - ff.loc[common]).abs()
    return float(delta.mean())


def d_method_dispersion(method_df: pd.DataFrame, domain: str, infer_mode: str) -> float:
    """D_method (spec VII.2): std over methods of mean_fake_score.

    Computed for one checkpoint/seed.
    """
    sub = method_df[
        (method_df["domain"] == domain)
        & (method_df["infer_mode"] == infer_mode)
        & (method_df["method_id"] != "all_fake")
    ]
    if len(sub) < 2:
        return float("nan")
    return float(np.std(sub["fake_score_mean"].to_numpy(), ddof=0))


def t_method_threshold_dispersion(
    method_df: pd.DataFrame, domain: str, infer_mode: str
) -> float:
    """T_method (spec VII.3): std over methods of optimal_threshold."""
    sub = method_df[
        (method_df["domain"] == domain)
        & (method_df["infer_mode"] == infer_mode)
        & (method_df["method_id"] != "all_fake")
    ]
    if len(sub) < 2:
        return float("nan")
    return float(np.std(sub["optimal_threshold"].to_numpy(), ddof=0))


def delta_auc_per_method(
    method_df: pd.DataFrame, domain: str
) -> pd.DataFrame:
    """DeltaAUC (spec VII.5): direction AUC - raw AUC per method.

    Returns a DataFrame with columns [method_id, seed, delta_auc, raw_auc,
    direction_auc]. One row per (method, seed).
    """
    sub = method_df[
        (method_df["domain"] == domain)
        & (method_df["method_id"] != "all_fake")
    ]
    # Index MUST include backbone: raw and direction rows exist for every
    # (backbone, method_id, seed). Omitting backbone makes the join a
    # cartesian product across backbones (3x3 = 9 rows per method×seed instead
    # of 3), pairing one backbone's raw AUC with another backbone's direction
    # AUC and inflating counts (720 vs the correct 240).
    key = ["backbone", "method_id", "seed"]
    raw = sub[sub["infer_mode"] == "raw"].set_index(key)[
        ["auc_method"]
    ].rename(columns={"auc_method": "raw_auc"})
    direction = sub[sub["infer_mode"] == "direction"].set_index(
        key
    )[["auc_method"]].rename(columns={"auc_method": "direction_auc"})
    joined = raw.join(direction, how="inner").reset_index()
    if joined.empty:
        return joined
    joined["delta_auc"] = joined["direction_auc"] - joined["raw_auc"]
    joined["domain"] = domain
    return joined


# ---------------------------------------------------------------------------
# Feature-norm diagnostics
# ---------------------------------------------------------------------------

def feature_norm_diagnostics(frame_df: pd.DataFrame) -> dict:
    """H5-related feature-norm diagnostics on per-frame data.

    Computes:
      - corr(feature_norm, prob_fake): does norm predict score magnitude?
      - corr(feature_norm, prediction_correct): does norm predict right/wrong?
      - bucketed AUC and error rate by feature_norm quartile (per
        checkpoint × infer_mode).

    Correlations are computed across all (checkpoint, infer_mode, domain)
    frames and also per infer_mode.
    """
    out: dict = {}

    # Overall correlations.
    out["corr_norm_prob_fake"] = float(
        frame_df["feature_norm"].corr(frame_df["prob_fake"])
    )
    out["corr_norm_correct"] = float(
        frame_df["feature_norm"].corr(frame_df["prediction_correct"])
    )

    # Per infer_mode correlations.
    per_mode: dict[str, dict[str, float]] = {}
    for mode, grp in frame_df.groupby("infer_mode"):
        per_mode[mode] = {
            "corr_norm_prob_fake": float(grp["feature_norm"].corr(grp["prob_fake"])),
            "corr_norm_correct": float(
                grp["feature_norm"].corr(grp["prediction_correct"])
            ),
        }
    out["per_mode"] = per_mode

    return out


def feature_norm_quartile_buckets(
    frame_df: pd.DataFrame, checkpoint_id: str, infer_mode: str
) -> pd.DataFrame:
    """For one (checkpoint, infer_mode), bucket frames by feature_norm quartile
    and report AUC + error rate per bucket (spec VII.6)."""
    sub = frame_df[
        (frame_df["checkpoint_id"] == checkpoint_id)
        & (frame_df["infer_mode"] == infer_mode)
    ].copy()
    if len(sub) < 4 or sub["label"].nunique() < 2:
        return pd.DataFrame()

    try:
        sub["norm_quartile"] = pd.qcut(
            sub["feature_norm"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
    except ValueError:
        # Non-unique bin edges — fall back to rank-based equal buckets.
        sub["norm_quartile"] = pd.qcut(
            sub["feature_norm"].rank(method="first"), q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        )

    rows = []
    for q, qg in sub.groupby("norm_quartile", observed=True):
        labels = qg["label"].to_numpy()
        scores = qg["prob_fake"].to_numpy()
        preds = (scores >= 0.5).astype(int)
        error_rate = float(1.0 - (preds == labels).mean())
        auc = (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2
            else float("nan")
        )
        rows.append({
            "checkpoint_id": checkpoint_id,
            "infer_mode": infer_mode,
            "norm_quartile": str(q),
            "n_frames": int(len(qg)),
            "auc": auc,
            "error_rate": error_rate,
            "norm_min": float(qg["feature_norm"].min()),
            "norm_max": float(qg["feature_norm"].max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

def _mean_std_over_seeds(values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(len(arr)),
    }


def aggregate_diagnostics_over_seeds(
    method_df: pd.DataFrame, infer_modes: list[str]
) -> pd.DataFrame:
    """For each (backbone, infer_mode), compute D_domain / D_method / T_method
    per seed, then aggregate to mean ± std.

    Returns a tidy DataFrame.
    """
    rows = []
    for backbone in sorted(method_df["backbone"].unique()):
        bb_df = method_df[method_df["backbone"] == backbone]
        for infer_mode in infer_modes:
            mode_df = bb_df[bb_df["infer_mode"] == infer_mode]
            if mode_df.empty:
                continue
            for seed in sorted(mode_df["seed"].unique()):
                seed_df = mode_df[mode_df["seed"] == seed]

                d_dom = d_domain_shift(seed_df, infer_mode)
                d_m_ff = d_method_dispersion(seed_df, "ffpp", infer_mode)
                d_m_cdf = d_method_dispersion(seed_df, "cdf", infer_mode)
                t_m_ff = t_method_threshold_dispersion(seed_df, "ffpp", infer_mode)
                t_m_cdf = t_method_threshold_dispersion(seed_df, "cdf", infer_mode)
                # mean threshold regret over methods, in each domain.
                regret_ff = float(
                    seed_df[(seed_df["domain"] == "ffpp") & (seed_df["method_id"] != "all_fake")][
                        "threshold_regret"
                    ].mean()
                )
                regret_cdf = float(
                    seed_df[(seed_df["domain"] == "cdf") & (seed_df["method_id"] != "all_fake")][
                        "threshold_regret"
                    ].mean()
                )

                rows.append({
                    "backbone": backbone,
                    "infer_mode": infer_mode,
                    "seed": int(seed),
                    "D_domain": d_dom,
                    "D_method_ffpp": d_m_ff,
                    "D_method_cdf": d_m_cdf,
                    "T_method_ffpp": t_m_ff,
                    "T_method_cdf": t_m_cdf,
                    "threshold_regret_ffpp": regret_ff,
                    "threshold_regret_cdf": regret_cdf,
                })

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed

    # Aggregate over seeds.
    summary = (
        per_seed.groupby(["backbone", "infer_mode"])
        .agg(
            D_domain_mean=("D_domain", "mean"),
            D_domain_std=("D_domain", "std"),
            D_method_ffpp_mean=("D_method_ffpp", "mean"),
            D_method_ffpp_std=("D_method_ffpp", "std"),
            D_method_cdf_mean=("D_method_cdf", "mean"),
            D_method_cdf_std=("D_method_cdf", "std"),
            T_method_ffpp_mean=("T_method_ffpp", "mean"),
            T_method_ffpp_std=("T_method_ffpp", "std"),
            T_method_cdf_mean=("T_method_cdf", "mean"),
            T_method_cdf_std=("T_method_cdf", "std"),
            threshold_regret_ffpp_mean=("threshold_regret_ffpp", "mean"),
            threshold_regret_cdf_mean=("threshold_regret_cdf", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return summary
