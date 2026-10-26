"""Apply frozen source-domain decisions to the target domain.

Nothing in this module fits anything: it consumes a `Calibrator` and a threshold
dict produced on FF++ validation and reports what happens on CDF.

Definitions
-----------
``R_transfer = BA_target(oracle threshold) - BA_target(source threshold)``
    How much balanced accuracy the frozen threshold gives up relative to the best
    threshold *on the target domain itself*. ``>= 0`` by construction; 0 means the
    source threshold is already optimal for CDF.

``fixed-FPR transfer error = FPR_target(t_FPRx_source) - x``
    Whether a "1% false positives" promise made on the source domain survives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from study_threshold.calibration import Calibrator, probability_metrics
from study_threshold.io import score_column
from study_threshold.thresholds import auc_mann_whitney, confusion, oracle_ba

FIXED_FPR_TARGETS = {"t_FPR1": 0.01, "t_FPR5": 0.05}


def evaluate_at_threshold(videos: pd.DataFrame, mode: str, threshold: float,
                          score_col: str | None = None) -> dict:
    col = score_col or score_column(mode)
    return confusion(videos[col].to_numpy(float), videos["label"].to_numpy(int), threshold)


def transfer_rows(source_videos: pd.DataFrame, target_videos: pd.DataFrame,
                  mode: str, thresholds: dict, calibrator: Calibrator,
                  score_col: str | None = None,
                  rule_scope: str = "pooled") -> list[dict]:
    """One row per threshold rule: source behaviour, target behaviour, regret."""
    col = score_col or score_column(mode)
    s_src = source_videos[col].to_numpy(float)
    y_src = source_videos["label"].to_numpy(int)
    s_tgt = target_videos[col].to_numpy(float)
    y_tgt = target_videos["label"].to_numpy(int)

    auc_src = auc_mann_whitney(s_src, y_src)
    auc_tgt = auc_mann_whitney(s_tgt, y_tgt)
    t_oracle, ba_oracle = oracle_ba(s_tgt, y_tgt)
    prob_metrics = probability_metrics(s_tgt, y_tgt, calibrator)
    prob_metrics_src = probability_metrics(s_src, y_src, calibrator)

    rows = []
    for rule, t in thresholds.items():
        src = confusion(s_src, y_src, t)
        tgt = confusion(s_tgt, y_tgt, t)
        row = {
            "mode": mode,
            "score": col,
            "scope": rule_scope,
            "rule": rule,
            "source_threshold": float(t),
            "source_auc": auc_src,
            "source_ba": src["balanced_accuracy"],
            "source_tpr": src["tpr"],
            "source_fpr": src["fpr"],
            "target_auc": auc_tgt,
            "target_ba": tgt["balanced_accuracy"],
            "target_accuracy": tgt["accuracy"],
            "target_tpr": tgt["tpr"],
            "target_tnr": tgt["tnr"],
            "target_fpr": tgt["fpr"],
            "target_fnr": tgt["fnr"],
            "target_precision": tgt["precision"],
            "target_f1": tgt["f1"],
            "target_predicted_fake_rate": tgt["predicted_fake_rate"],
            "target_oracle_threshold": float(t_oracle),
            "target_oracle_ba": float(ba_oracle),
            "r_transfer": float(ba_oracle - tgt["balanced_accuracy"]),
            "threshold_shift_oracle_minus_source": float(t_oracle - t),
            "n_target_fake": tgt["n_fake"],
            "n_target_real": tgt["n_real"],
            "n_source_fake": src["n_fake"],
            "n_source_real": src["n_real"],
            "temperature": calibrator.temperature,
            "platt_a": calibrator.platt_a,
            "platt_b": calibrator.platt_b,
        }
        if rule in FIXED_FPR_TARGETS:
            row["fixed_fpr_target"] = FIXED_FPR_TARGETS[rule]
            row["fixed_fpr_error"] = float(tgt["fpr"] - FIXED_FPR_TARGETS[rule])
        else:
            row["fixed_fpr_target"] = np.nan
            row["fixed_fpr_error"] = np.nan
        row.update({f"target_{k}": v for k, v in prob_metrics.items()})
        row.update({f"source_{k}": v for k, v in prob_metrics_src.items()})
        rows.append(row)
    return rows


def baseline_mean_prob_rows(source_videos: pd.DataFrame, target_videos: pd.DataFrame,
                            mode: str) -> list[dict]:
    """The mis-specified baseline: uncalibrated ``mean_t(sigmoid(s))`` at p >= 0.5.

    Reported to show why the study's primary score is `mean_logit`; never used for
    conclusions about calibration or transfer quality.
    """
    col = {"raw": "video_prob_raw", "direction": "video_prob_dir"}[mode]
    src = confusion(source_videos[col].to_numpy(float),
                    source_videos["label"].to_numpy(int), 0.5)
    tgt = confusion(target_videos[col].to_numpy(float),
                    target_videos["label"].to_numpy(int), 0.5)
    t_oracle, ba_oracle = oracle_ba(target_videos[col].to_numpy(float),
                                    target_videos["label"].to_numpy(int))
    return [{
        "mode": mode,
        "score": col,
        "scope": "pooled",
        "rule": "baseline_mean_prob@0.5",
        "source_threshold": 0.5,
        "source_ba": src["balanced_accuracy"],
        "source_fpr": src["fpr"],
        "target_ba": tgt["balanced_accuracy"],
        "target_tpr": tgt["tpr"],
        "target_tnr": tgt["tnr"],
        "target_fpr": tgt["fpr"],
        "target_predicted_fake_rate": tgt["predicted_fake_rate"],
        "target_oracle_threshold": float(t_oracle),
        "target_oracle_ba": float(ba_oracle),
        "r_transfer": float(ba_oracle - tgt["balanced_accuracy"]),
        "n_target_fake": tgt["n_fake"],
        "n_target_real": tgt["n_real"],
    }]
