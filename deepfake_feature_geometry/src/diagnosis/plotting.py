"""Diagnostic figures (spec section XI).

Six figures:
  1. CDF direction-raw AUC gain heatmap (16 methods × 5 seeds).
  2. FF++ -> CDF score shift per method.
  3. Optimal threshold distribution by infer_mode (boxplot).
  4. D_method comparison across raw / direction / normtrain_norm.
  5. Feature norm quartile vs error rate.
  6. BA@0.5 vs optimal_BA scatter.

All figures are written to <output_dir>/figures/*.png. We use Agg backend so
no display is required.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_cdf_delta_auc_heatmap(
    delta_auc_df: pd.DataFrame, output_path: Path
) -> None:
    """Figure 1: 16 methods × 5 seeds heatmap of (direction_auc - raw_auc) on CDF."""
    if delta_auc_df.empty:
        return
    cdf = delta_auc_df[delta_auc_df["domain"] == "cdf"]
    if cdf.empty:
        return
    pivot = cdf.pivot_table(
        index="method_id", columns="seed", values="delta_auc", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Δ AUC (direction - raw)"},
        ax=ax,
    )
    ax.set_title("CDF: Direction - Raw AUC gain per method × seed")
    ax.set_ylabel("Method")
    ax.set_xlabel("Seed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_score_shift_per_method(
    method_df: pd.DataFrame, infer_mode: str, output_path: Path
) -> None:
    """Figure 2: FF++ -> CDF shift in mean fake score per method."""
    sub = method_df[
        (method_df["infer_mode"] == infer_mode)
        & (method_df["method_id"] != "all_fake")
    ]
    if sub.empty:
        return
    ff = sub[sub["domain"] == "ffpp"].groupby("method_id")["fake_score_mean"].mean()
    cdf = sub[sub["domain"] == "cdf"].groupby("method_id")["fake_score_mean"].mean()
    common = ff.index.intersection(cdf.index)
    shift = (cdf.loc[common] - ff.loc[common]).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in shift.values]
    ax.barh(range(len(shift)), shift.values, color=colors)
    ax.set_yticks(range(len(shift)))
    ax.set_yticklabels(shift.index)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ mean fake score  (CDF - FF++)")
    ax.set_title(f"FF++ → CDF score shift per method  (infer_mode={infer_mode})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_optimal_threshold_distribution(
    method_df: pd.DataFrame, domain: str, output_path: Path
) -> None:
    """Figure 3: optimal threshold distribution per infer_mode."""
    sub = method_df[
        (method_df["domain"] == domain)
        & (method_df["method_id"] != "all_fake")
    ]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=sub,
        x="infer_mode",
        y="optimal_threshold",
        hue="backbone",
        ax=ax,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5")
    ax.set_title(f"Optimal threshold distribution by infer_mode  (domain={domain})")
    ax.set_ylabel("Optimal balanced-accuracy threshold")
    ax.set_xlabel("Inference mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_d_method_comparison(
    aggregate_df: pd.DataFrame, domain: str, output_path: Path
) -> None:
    """Figure 4: D_method bar chart across raw / direction / normtrain_norm."""
    if aggregate_df.empty:
        return
    col = "D_method_cdf_mean" if domain == "cdf" else "D_method_ffpp_mean"
    err_col = "D_method_cdf_std" if domain == "cdf" else "D_method_ffpp_std"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=aggregate_df,
        x="infer_mode",
        y=col,
        hue="backbone",
        ax=ax,
    )
    ax.set_title(f"D_method (method-level score dispersion) on {domain.upper()}")
    ax.set_ylabel("D_method  (std over methods of mean fake score)")
    ax.set_xlabel("Inference mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_norm_quartile_error(
    norm_bucket_df: pd.DataFrame, output_path: Path
) -> None:
    """Figure 5: feature norm quartile vs error rate."""
    if norm_bucket_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    # Average over checkpoints/infer_modes for a clean per-quartile view.
    agg = (
        norm_bucket_df.groupby(["infer_mode", "norm_quartile"])
        .agg(mean_error=("error_rate", "mean"), mean_auc=("auc", "mean"))
        .reset_index()
    )
    sns.lineplot(
        data=agg,
        x="norm_quartile",
        y="mean_error",
        hue="infer_mode",
        marker="o",
        ax=ax,
    )
    ax.set_title("Error rate by feature-norm quartile")
    ax.set_xlabel("Feature-norm quartile  (Q1 = lowest norm)")
    ax.set_ylabel("Mean frame error rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ba_vs_optimal_ba(
    method_df: pd.DataFrame, domain: str, output_path: Path
) -> None:
    """Figure 6: BA@0.5 vs optimal_BA scatter per (method, seed)."""
    sub = method_df[
        (method_df["domain"] == domain)
        & (method_df["method_id"] != "all_fake")
    ]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        data=sub,
        x="ba_at_0.5",
        y="optimal_ba",
        hue="infer_mode",
        style="backbone",
        ax=ax,
        alpha=0.7,
    )
    lims = [
        min(sub["ba_at_0.5"].min(), sub["optimal_ba"].min()),
        max(sub["ba_at_0.5"].max(), sub["optimal_ba"].max()),
    ]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8, label="y = x")
    ax.set_title(f"BA@0.5 vs optimal BA  (domain={domain})")
    ax.set_xlabel("Balanced accuracy @ threshold=0.5")
    ax.set_ylabel("Optimal balanced accuracy")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_all_figures(
    *,
    output_dir: Path,
    method_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    delta_auc_df: pd.DataFrame,
    norm_bucket_df: pd.DataFrame,
) -> list[str]:
    """Generate all six figures and return the list of written paths."""
    fig_dir = output_dir / "figures"
    _ensure_dir(fig_dir)

    written: list[str] = []

    # Figure 1: CDF delta-AUC heatmap.
    p = fig_dir / "01_cdf_delta_auc_heatmap.png"
    plot_cdf_delta_auc_heatmap(delta_auc_df, p)
    written.append(str(p))

    # Figure 2: score shift per method (direction mode is the diagnostic one).
    p = fig_dir / "02_ffpp_to_cdf_score_shift_direction.png"
    plot_score_shift_per_method(method_df, "direction", p)
    written.append(str(p))

    # Figure 3: optimal threshold distribution on CDF.
    p = fig_dir / "03_optimal_threshold_distribution_cdf.png"
    plot_optimal_threshold_distribution(method_df, "cdf", p)
    written.append(str(p))

    # Figure 4: D_method comparison on CDF.
    p = fig_dir / "04_d_method_comparison_cdf.png"
    plot_d_method_comparison(aggregate_df, "cdf", p)
    written.append(str(p))

    # Figure 5: norm quartile vs error.
    p = fig_dir / "05_norm_quartile_vs_error.png"
    plot_norm_quartile_error(norm_bucket_df, p)
    written.append(str(p))

    # Figure 6: BA@0.5 vs optimal BA on CDF.
    p = fig_dir / "06_ba_at_05_vs_optimal_ba_cdf.png"
    plot_ba_vs_optimal_ba(method_df, "cdf", p)
    written.append(str(p))

    return written
