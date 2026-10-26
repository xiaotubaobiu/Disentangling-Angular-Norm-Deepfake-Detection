"""Hypothesis summary (spec section XII).

Builds a structured verdict for H1..H5 from the diagnostic outputs. Each
hypothesis gets:
  - question
  - verdict: "supported" | "partially_supported" | "refuted" | "inconclusive"
  - evidence: numeric dict
  - reasoning: short text

Verdict thresholds are intentionally conservative; the goal is to surface the
evidence, not to over-claim. The user reads the reasoning + evidence to make
the final call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_lookup(df: pd.DataFrame, **filters) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for k, v in filters.items():
        mask &= df[k].astype(str) == str(v)
    return df[mask]


def summarize_h1(
    aggregate_df: pd.DataFrame, delta_auc_df: pd.DataFrame
) -> dict:
    """H1: Does Direction reduce FF++ -> CDF domain shift on most methods?

    We compare D_domain for raw vs direction (lower = less shift), and also
    check how many methods have positive delta_auc on CDF.
    """
    h: dict = {"question": (
        "Direction 是否在大多数方法上降低了 FF++ → CDF 的域偏移?"
    )}

    raw_row = _safe_lookup(aggregate_df, infer_mode="raw")
    dir_row = _safe_lookup(aggregate_df, infer_mode="direction")

    d_raw = float(raw_row["D_domain_mean"].mean()) if not raw_row.empty else float("nan")
    d_dir = float(dir_row["D_domain_mean"].mean()) if not dir_row.empty else float("nan")

    cdf_delta = delta_auc_df[delta_auc_df["domain"] == "cdf"] if (not delta_auc_df.empty and "domain" in delta_auc_df.columns) else delta_auc_df
    if not cdf_delta.empty and "delta_auc" in cdf_delta.columns:
        n_pos = int((cdf_delta["delta_auc"] > 0).sum())
        n_total = int(len(cdf_delta))
    else:
        n_pos = 0
        n_total = 0

    if np.isnan(d_raw) or np.isnan(d_dir):
        verdict = "inconclusive"
    elif d_dir < d_raw and n_total > 0 and n_pos > n_total / 2:
        verdict = "supported"
    elif d_dir < d_raw:
        verdict = "partially_supported"
    else:
        verdict = "refuted"

    h["verdict"] = verdict
    h["evidence"] = {
        "D_domain_raw_mean": d_raw,
        "D_domain_direction_mean": d_dir,
        "D_domain_reduction": d_raw - d_dir,
        "n_methods_direction_higher_cdf_auc": n_pos,
        "n_methods_total": n_total,
        "pct_methods_direction_higher": (n_pos / n_total) if n_total else float("nan"),
    }
    h["reasoning"] = (
        f"D_domain(raw)={d_raw:.4f}, D_domain(direction)={d_dir:.4f} "
        f"(lower = less shift). 在 CDF 上,direction 的 AUC 比 raw 高的方法有 "
        f"{n_pos}/{n_total} 个。"
    )
    return h


def summarize_h2(
    method_df: pd.DataFrame, aggregate_df: pd.DataFrame
) -> dict:
    """H2: Is Direction's ~0.48 threshold genuinely low-regret, or just a
    softmax-collapse artifact?

    Compares mean threshold_regret (lower = better) for direction vs raw on
    CDF, and reports the mean optimal threshold for direction.
    """
    h: dict = {"question": (
        "Direction 的 0.48 阈值是否同时具有较小 threshold regret,还是仅因概率"
        "收缩到 0.5 附近?"
    )}

    cdf = method_df[(method_df["domain"] == "cdf") & (method_df["method_id"] != "all_fake")]
    raw_regret = cdf[cdf["infer_mode"] == "raw"]["threshold_regret"]
    dir_regret = cdf[cdf["infer_mode"] == "direction"]["threshold_regret"]
    raw_reg = float(raw_regret.mean()) if not raw_regret.empty else float("nan")
    dir_reg = float(dir_regret.mean()) if not dir_regret.empty else float("nan")

    dir_thr = cdf[cdf["infer_mode"] == "direction"]["optimal_threshold"]
    dir_thr_mean = float(dir_thr.mean()) if not dir_thr.empty else float("nan")
    dir_thr_std = float(dir_thr.std()) if not dir_thr.empty else float("nan")

    if np.isnan(dir_reg):
        verdict = "inconclusive"
    elif dir_reg < 0.02:
        verdict = "supported"
    elif dir_reg < raw_reg:
        verdict = "partially_supported"
    else:
        verdict = "refuted"

    h["verdict"] = verdict
    h["evidence"] = {
        "direction_threshold_regret_mean": dir_reg,
        "raw_threshold_regret_mean": raw_reg,
        "direction_optimal_threshold_mean": dir_thr_mean,
        "direction_optimal_threshold_std": dir_thr_std,
    }
    h["reasoning"] = (
        f"direction 的 mean threshold regret={dir_reg:.4f}, raw={raw_reg:.4f}。"
        f"direction 的 optimal threshold mean={dir_thr_mean:.4f}, std={dir_thr_std:.4f}。"
        "若 regret 小,则阈值稳定不是 softmax 收缩的副产物。"
    )
    return h


def summarize_h3(
    aggregate_df: pd.DataFrame, method_df: pd.DataFrame
) -> dict:
    """H3: Do different forgery methods have significantly different direction
    score distributions? Is this difference larger than seed variance?"""
    h: dict = {"question": (
        "不同伪造方法的 direction score 分布是否显著不同?"
        "这种方法差异是否大于 seed 波动?"
    )}

    dir_agg = _safe_lookup(aggregate_df, infer_mode="direction")
    d_method_cdf = (
        float(dir_agg["D_method_cdf_mean"].mean()) if not dir_agg.empty else float("nan")
    )

    # Seed-to-seed variation of mean fake score per method as a noise floor.
    dir_cdf = method_df[
        (method_df["domain"] == "cdf")
        & (method_df["infer_mode"] == "direction")
        & (method_df["method_id"] != "all_fake")
    ]
    if not dir_cdf.empty:
        # Seed noise = std of fake_score_mean across the 5 seeds WITHIN each
        # (backbone, method), then averaged. Grouping by method_id alone would
        # pool the 3 backbones' means into the std and contaminate the
        # seed-to-seed noise floor with backbone variance.
        per_method_seed_std = (
            dir_cdf.groupby(["backbone", "method_id"])["fake_score_mean"].std().mean()
        )
        seed_noise = float(per_method_seed_std)
    else:
        seed_noise = float("nan")

    if np.isnan(d_method_cdf) or np.isnan(seed_noise):
        verdict = "inconclusive"
    elif d_method_cdf > 2 * seed_noise and seed_noise > 0:
        verdict = "supported"
    elif d_method_cdf > seed_noise:
        verdict = "partially_supported"
    else:
        verdict = "refuted"

    h["verdict"] = verdict
    h["evidence"] = {
        "direction_D_method_cdf": d_method_cdf,
        "direction_seed_noise_floor": seed_noise,
        "ratio_method_over_seed": (d_method_cdf / seed_noise) if seed_noise else float("nan"),
    }
    h["reasoning"] = (
        f"D_method(CDF, direction)={d_method_cdf:.4f}, "
        f"seed-level noise floor={seed_noise:.4f}。"
        "若方法间离散显著大于 seed 间离散,说明方法分布确实不同。"
    )
    return h


def summarize_h4(aggregate_df: pd.DataFrame) -> dict:
    """H4: Does NormTrain reduce some methods' error but increase method-level
    score/threshold dispersion?"""
    h: dict = {"question": (
        "Normtrain 是否降低了部分方法的分类错误,但增加了方法间 score/threshold 离散?"
    )}

    dir_agg = _safe_lookup(aggregate_df, infer_mode="direction")
    norm_agg = _safe_lookup(aggregate_df, infer_mode="normtrain_norm")

    d_method_dir_cdf = (
        float(dir_agg["D_method_cdf_mean"].mean()) if not dir_agg.empty else float("nan")
    )
    d_method_norm_cdf = (
        float(norm_agg["D_method_cdf_mean"].mean()) if not norm_agg.empty else float("nan")
    )
    t_method_dir_cdf = (
        float(dir_agg["T_method_cdf_mean"].mean()) if not dir_agg.empty else float("nan")
    )
    t_method_norm_cdf = (
        float(norm_agg["T_method_cdf_mean"].mean()) if not norm_agg.empty else float("nan")
    )

    if any(np.isnan([d_method_dir_cdf, d_method_norm_cdf])):
        verdict = "inconclusive"
    elif d_method_norm_cdf > d_method_dir_cdf:
        verdict = "supported"
    else:
        verdict = "refuted"

    h["verdict"] = verdict
    h["evidence"] = {
        "D_method_cdf_direction": d_method_dir_cdf,
        "D_method_cdf_normtrain": d_method_norm_cdf,
        "T_method_cdf_direction": t_method_dir_cdf,
        "T_method_cdf_normtrain": t_method_norm_cdf,
    }
    h["reasoning"] = (
        f"D_method(CDF): direction={d_method_dir_cdf:.4f}, normtrain={d_method_norm_cdf:.4f}。"
        f"T_method(CDF): direction={t_method_dir_cdf:.4f}, normtrain={t_method_norm_cdf:.4f}。"
        "normtrain 若方法离散更大,则 H4 supported。"
    )
    return h


def summarize_h5(norm_diagnostics: dict) -> dict:
    """H5: Can feature norm predict frame correctness? If it only inflates
    confidence without improving correctness, label it as 'confidence
    polarizer' rather than 'reliability estimator'."""
    h: dict = {"question": (
        "Feature norm 是否能预测帧的正确性?"
        "若只能提高置信度而不能提高正确率,则标记为 'confidence polarizer'。"
    )}

    corr_norm_correct = norm_diagnostics.get("corr_norm_correct", float("nan"))
    corr_norm_prob = norm_diagnostics.get("corr_norm_prob_fake", float("nan"))

    if np.isnan(corr_norm_correct):
        verdict = "inconclusive"
        tag = "unknown"
    elif abs(corr_norm_correct) < 0.05 and abs(corr_norm_prob) > 0.1:
        verdict = "supported"
        tag = "confidence_polarizer"
    elif abs(corr_norm_correct) > 0.15:
        verdict = "refuted"
        tag = "reliability_estimator"
    else:
        verdict = "partially_supported"
        tag = "mixed"

    h["verdict"] = verdict
    h["evidence"] = {
        "corr_norm_prob_fake": corr_norm_prob,
        "corr_norm_correct": corr_norm_correct,
        "norm_role_tag": tag,
    }
    h["reasoning"] = (
        f"corr(norm, prob_fake)={corr_norm_prob:.4f}, "
        f"corr(norm, prediction_correct)={corr_norm_correct:.4f}。"
        f"判定: {tag}。"
    )
    return h


def build_hypothesis_summary(
    *,
    aggregate_df: pd.DataFrame,
    method_df: pd.DataFrame,
    delta_auc_df: pd.DataFrame,
    norm_diagnostics: dict,
) -> dict:
    """Compose the full H1..H5 summary dict (will be JSON-dumped)."""
    return {
        "H1": summarize_h1(aggregate_df, delta_auc_df),
        "H2": summarize_h2(method_df, aggregate_df),
        "H3": summarize_h3(aggregate_df, method_df),
        "H4": summarize_h4(aggregate_df),
        "H5": summarize_h5(norm_diagnostics),
    }
