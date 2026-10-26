#!/usr/bin/env python3
"""Render descriptive paper figures from existing result tables only.

Run: python /absolute/path/to/render_figures.py
No model fitting, resampling, training, or source-table mutation is performed.
All AUC differences are percentage points (100 times the stored AUC difference).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
POOLING = PROJECT / "experiments/2026-07-27_pooling_saturation/results/tables"
THRESHOLDS = PROJECT / "experiments/2026-07-30_threshold_transfer/results/tables"
INPUTS = {
    "aggregation": POOLING / "Q_pooling_aggregation_auc.csv",
    "calibrated": POOLING / "Q_pooling_calibrated_auc.csv",
    "gamma": POOLING / "Q_pooling_gamma_scan.csv",
    "threshold": THRESHOLDS / "04_transfer_pooled.csv",
}
PROB, LOGIT, TEMP = "#1877A8", "#D16C23", "#16836A"
RAW, DIRECTION = "#C6692B", "#267BA3"
METHODS = ["MRAA", "blendface", "e4s", "facedancer", "faceswap", "facevid2vid",
           "fomm", "fsgan", "inswap", "lia", "mobileswap", "pirender",
           "sadtalker", "simswap", "tpsm", "wav2lip"]
NAMES = {"MRAA": "MRAA", "blendface": "BlendFace", "e4s": "E4S",
         "facedancer": "FaceDancer", "faceswap": "FaceSwap", "facevid2vid": "FaceVid2Vid",
         "fomm": "FOMM", "fsgan": "FSGAN", "inswap": "InSwap", "lia": "LIA",
         "mobileswap": "MobileSwap", "pirender": "PIRender", "sadtalker": "SadTalker",
         "simswap": "SimSwap", "tpsm": "TPSM", "wav2lip": "Wav2Lip"}
KEY = ["domain", "backbone", "seed", "method"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7A838B", "axes.linewidth": 0.7,
        "xtick.color": "#35424D", "ytick.color": "#35424D",
        "text.color": "#23313D", "axes.labelcolor": "#23313D",
        "grid.color": "#DDE3E8", "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight",
                metadata={"Title": stem, "Creator": "render_figures.py / Matplotlib"})
    plt.close(fig)


def validate(a: pd.DataFrame, c: pd.DataFrame, g: pd.DataFrame) -> dict:
    assert len(a) == 4800 and len(c) == 480 and len(g) == 10560
    for frame, extra in [(a, ["aggregation"]), (c, []), (g, ["aggregation", "gamma"])]:
        assert not frame.duplicated(KEY + extra).any()
        assert set(frame.domain) == {"ffpp", "cdf"}
        assert set(frame.method) == set(METHODS)
        assert set(frame.backbone) == {"clip", "dinov2", "efficientnet"}
        assert set(frame.seed) == {7, 42, 123, 999, 2048}
    assert a.groupby(["domain", "method", "aggregation"]).size().eq(15).all()
    assert c.groupby(["domain", "method"]).size().eq(15).all()
    assert g.groupby(["domain", "method", "aggregation", "gamma"]).size().eq(15).all()
    assert np.allclose(sorted(g.gamma.unique()), np.linspace(0, 1, 11))
    assert np.allclose(a.delta_auc, a.auc_direction - a.auc_raw)
    assert np.allclose(c.delta_calib, c.auc_dir_calib - c.auc_raw_calib)
    # Audit all endpoints. Do not alter small inconsistencies in existing tables.
    endpoint_audit = []
    for gamma_value, view in [(0.0, "direction"), (1.0, "raw")]:
        z = g[g.gamma.eq(gamma_value)].merge(a, on=KEY + ["aggregation"], validate="one_to_one")
        assert len(z) == 960
        z["gamma_endpoint_minus_main_table_pp"] = 100 * (z.video_auc - z[f"auc_{view}"])
        mismatch = z[z.gamma_endpoint_minus_main_table_pp.ne(0)]
        endpoint_audit.append({"gamma": gamma_value, "view": view, "n_compared": len(z),
                               "n_not_exactly_equal": len(mismatch),
                               "maximum_absolute_difference_pp": float(z.gamma_endpoint_minus_main_table_pp.abs().max()),
                               "differences": mismatch[KEY + ["aggregation", "gamma_endpoint_minus_main_table_pp"]].to_dict("records")})
        plotted = z[z.domain.eq("cdf") & z.method.isin(["fsgan", "sadtalker"])]
        assert plotted.gamma_endpoint_minus_main_table_pp.eq(0).all()
    return {"gamma_endpoint_audit": endpoint_audit,
            "plotted_gamma_endpoints_match_main_table_exactly": True}


def normalization_effects(a: pd.DataFrame, c: pd.DataFrame) -> dict:
    z = a[a.domain.eq("cdf") & a.aggregation.isin(["mean_prob", "mean_logit"])].copy()
    z["effect_pp"] = 100 * (z.auc_direction - z.auc_raw)
    effect = z.groupby(["method", "aggregation"]).effect_pp.mean().unstack().reindex(METHODS)
    cz = c[c.domain.eq("cdf")].copy()
    cz["effect_pp"] = 100 * (cz.auc_dir_calib - cz.auc_raw_calib)
    effect["source_temperature_mean_prob"] = cz.groupby("method").effect_pp.mean()
    fig, ax = plt.subplots(figsize=(7.7, 7.6))
    y = np.arange(len(METHODS))
    for i in range(len(METHODS)):
        if i % 2 == 0:
            ax.axhspan(i - .47, i + .47, color="#F3F6F8", zorder=0)
    ax.axvline(0, color="#5A6772", lw=1, zorder=1)
    series = [("mean_prob", "Probability mean", PROB, "o", -.22),
              ("mean_logit", "Logit mean", LOGIT, "s", 0),
              ("source_temperature_mean_prob", "Source-temperature probability mean", TEMP, "D", .22)]
    for col, label, color, marker, offset in series:
        ax.scatter(effect[col], y + offset, c=color, marker=marker, s=43,
                   label=label, zorder=3, edgecolors="white", linewidths=.55)
    ax.set_yticks(y, [NAMES[m] for m in METHODS])
    for label in ax.get_yticklabels():
        if label.get_text() in ("FSGAN", "SadTalker"):
            label.set_fontweight("bold")
    ax.set_ylim(len(METHODS) - .45, -.65)
    ax.set_xlim(-5.3, 3.55)
    ax.set_xticks(np.arange(-5, 4))
    ax.set_xlabel("AUC(direction) − AUC(raw)  [percentage points]", labelpad=9)
    ax.grid(axis="x", zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("CDF: normalization effects across 16 generation methods", loc="left", pad=84)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
              fontsize=9.6, borderaxespad=0, labelspacing=.7)
    ax.text(0, 1.165, "Equal-weight means over 15 fixed models (3 backbones × 5 seeds)",
            transform=ax.transAxes, fontsize=9.4, color="#52616D")
    fig.subplots_adjust(left=.2, right=.97, top=.77, bottom=.1)
    save(fig, "cdf_normalization_effects")
    absolute = z.groupby(["method", "aggregation"])[["auc_raw", "auc_direction"]].mean().mul(100)
    cal_absolute = cz.groupby("method")[["auc_raw_calib", "auc_dir_calib"]].mean().mul(100)
    return {"unit": "AUC percentage points", "models_per_mean": 15,
            "method_effects_pp": effect.reset_index().to_dict("records"),
            "representative_absolute_auc_percent": absolute.loc[["fsgan", "sadtalker"]].reset_index().to_dict("records"),
            "representative_temperature_absolute_auc_percent": cal_absolute.loc[["fsgan", "sadtalker"]].reset_index().to_dict("records")}


def aggregation_sensitivity(a: pd.DataFrame) -> dict:
    piv = a[a.aggregation.isin(["mean_prob", "mean_logit"])].pivot(
        index=KEY, columns="aggregation", values=["auc_raw", "auc_direction"])
    fig, axs = plt.subplots(2, 2, figsize=(7.8, 6.6), sharey="row")
    metrics = []
    rng = np.random.default_rng(20260928)
    for row, (view, color) in enumerate([("raw", RAW), ("direction", DIRECTION)]):
        for col, (domain, display) in enumerate([("ffpp", "FF++"), ("cdf", "CDF")]):
            ax = axs[row, col]
            p = piv.loc[domain]
            values = 100 * (p[(f"auc_{view}", "mean_logit")] - p[(f"auc_{view}", "mean_prob")])
            assert len(values) == 240
            mean, mean_abs = float(values.mean()), float(values.abs().mean())
            metrics.append({"domain": domain, "view": view, "n_cells": len(values),
                            "mean_pp": mean, "mean_absolute_pp": mean_abs,
                            "median_pp": float(values.median()),
                            "min_pp": float(values.min()), "max_pp": float(values.max()),
                            "cells_absolute_at_least_1pp": int(values.abs().ge(1).sum())})
            ax.axhline(0, color="#697783", lw=.8, zorder=1)
            # Horizontal jitter is visual only and uses a fixed seed.
            ax.scatter(rng.uniform(.52, .96, len(values)), values,
                       color=color, s=12, alpha=.4, edgecolors="none", zorder=2)
            ax.boxplot([values.to_numpy()], positions=[.3], widths=.18, whis=(0, 100),
                       showfliers=False, patch_artist=True,
                       boxprops={"facecolor": color, "alpha": .22, "edgecolor": color},
                       whiskerprops={"color": color, "linewidth": 1},
                       capprops={"color": color, "linewidth": 1},
                       medianprops={"color": "#23313D", "linewidth": 1.5})
            ax.set_xlim(.08, 1.1)
            ax.set_xticks([])
            ax.set_title(f"({chr(97 + row * 2 + col)})  {display} / {view}", loc="left", pad=10)
            mean_label = f"{mean:+.6f}" if abs(mean) < .00005 else f"{mean:+.4f}"
            ax.text(.98, .95, f"Mean |Δ| = {mean_abs:.4f} pp\nMean Δ = {mean_label} pp",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
            ax.grid(axis="y", zorder=0)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if col == 0:
                ax.set_ylabel("Logit mean − probability mean\n[AUC percentage points]")
    for ax in axs[0]:
        ax.set_ylim(-10, 8.5)
    for ax in axs[1]:
        ax.set_ylim(-.18, .24)
    fig.suptitle("Fixed-view sensitivity to video aggregation", x=.12, y=.98, ha="left", fontweight="bold", fontsize=13)
    fig.text(.12, .925, "240 model × method cells per panel; vertical scales differ between rows",
             fontsize=9.5, color="#52616D")
    fig.text(.12, .025, "Dots: individual cells. Boxes: median and interquartile range; whiskers: minimum–maximum.\n"
             "These distributions describe configuration variation, not confidence intervals.",
             fontsize=8.5, color="#52616D", linespacing=1.5)
    fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.86, hspace=.38, wspace=.12)
    save(fig, "aggregation_sensitivity")
    return {"unit": "AUC percentage points", "definition": "AUC(mean_logit) - AUC(mean_prob) at fixed view",
            "groups": metrics}


def gamma_curves(g: pd.DataFrame) -> dict:
    z = g[g.domain.eq("cdf") & g.method.isin(["fsgan", "sadtalker"])]
    curve = z.groupby(["method", "aggregation", "gamma"]).video_auc.mean().mul(100)
    fig, axs = plt.subplots(1, 2, figsize=(8.3, 4.55))
    maxima = []
    for ax, method, letter in zip(axs, ["fsgan", "sadtalker"], ["a", "b"]):
        for agg, label, color, marker in [("mean_prob", "Probability mean", PROB, "o"),
                                          ("mean_logit", "Logit mean", LOGIT, "s")]:
            c = curve.loc[method, agg].sort_index()
            best_gamma = float(c.idxmax())  # First maximum on the mean curve, not per-model selection.
            maxima.append({"method": method, "aggregation": agg, "gamma_at_first_maximum": best_gamma,
                           "maximum_mean_auc_percent": float(c.max()),
                           "direction_endpoint_auc_percent": float(c.loc[0]),
                           "raw_endpoint_auc_percent": float(c.loc[1])})
            ax.plot(c.index, c, color=color, marker=marker, markersize=4.7, lw=1.9, label=label)
            ax.scatter([best_gamma], [c.loc[best_gamma]], s=110, facecolors="none", edgecolors=color,
                       linewidths=1.35, zorder=4)
        ax.set_title(f"({letter})  CDF / {NAMES[method]}", loc="left", pad=14)
        ax.set_xlim(-.04, 1.04)
        ax.set_xticks([0, .2, .4, .6, .8, 1], ["0\n(direction)", "0.2", "0.4", "0.6", "0.8", "1\n(raw)"])
        ax.set_xlabel("Radial exponent γ", labelpad=9)
        ax.set_ylabel("Mean video AUC (%)")
        ax.grid(alpha=.9)
    axs[0].set_ylim(91.4, 95.35)
    axs[1].set_ylim(75.2, 81.25)
    axs[0].set_yticks([92, 93, 94, 95])
    axs[1].set_yticks([76, 77, 78, 79, 80, 81])
    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                          markeredgecolor="#52616D", markersize=9))
    labels.append("Maximum of the mean curve")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3,
               frameon=False, fontsize=9, columnspacing=1.7)
    fig.text(.1, .015, "Each point averages the same 15 models. No uncertainty interval is shown.\n"
             "Target-label optima are descriptive; the two panels use different vertical ranges.",
             fontsize=8.7, color="#52616D", linespacing=1.5)
    fig.subplots_adjust(left=.1, right=.98, top=.83, bottom=.25, wspace=.3)
    save(fig, "gamma_curves")
    return {"unit": "absolute AUC percent", "models_per_gamma": 15,
            "selection": "first maximum after averaging 15 models at each gamma",
            "curves": curve.rename("mean_auc_percent").reset_index().to_dict("records"),
            "mean_curve_maxima": maxima}


def threshold_points(t: pd.DataFrame) -> dict:
    z = t[t.rule.eq("t_BA") & t.scope.eq("pooled")].copy()
    assert len(z) == 30 and z.groupby("mode").size().eq(15).all()
    assert set(z.score) == {"s_raw", "s_dir"}
    cols = ["source_auc", "source_ba", "source_fpr", "target_auc", "target_ba", "target_fpr"]
    avg = z.groupby("mode")[cols].mean().mul(100)
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 4.45), sharey=True)
    for ax, metric, title, letter in zip(axs, ["ba", "fpr"], ["Balanced accuracy", "False-positive rate"], ["a", "b"]):
        for mode, color, offset, marker in [("raw", RAW, -.04, "o"), ("direction", DIRECTION, .04, "s")]:
            vals = [avg.loc[mode, f"source_{metric}"], avg.loc[mode, f"target_{metric}"]]
            xs = np.array([0., 1.]) + offset
            ax.plot(xs, vals, c=color, marker=marker, ms=6.2, lw=1.8, label=mode)
            for x, value in zip(xs, vals):
                shift = 10 if mode == "raw" else -18
                offset, align = (0, shift), "center"
                if metric == "fpr" and x < .5:
                    offset = (-8, 12) if mode == "raw" else (8, 12)
                    align = "right" if mode == "raw" else "left"
                ax.annotate(f"{value:.2f}", (x, value), xytext=offset, textcoords="offset points",
                            ha=align, color=color, fontsize=9,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1})
        ax.set_xlim(-.22, 1.22)
        ax.set_ylim(-6, 106)
        ax.set_xticks([0, 1], ["FF++ validation\n(source)", "CDF\n(target)"])
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(f"({letter})  {title}", loc="left", pad=15)
        ax.grid(axis="y")
    axs[0].set_ylabel("Rate (%)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False)
    fig.text(.1, .03, "Mean-logit scores; each model/view uses its own source max-BA threshold, frozen on CDF.\n"
             "Means over 15 models per view. Source results use the threshold-selection validation set.\n"
             "CDF pools all fake methods; no uncertainty interval is shown.",
             fontsize=8.6, color="#52616D", linespacing=1.5)
    fig.subplots_adjust(left=.1, right=.98, bottom=.27, top=.8, wspace=.18)
    save(fig, "threshold_operating_points")
    return {"unit": "percent", "rule": "t_BA", "aggregation": "mean_logit",
            "target_scope": "pooled fake methods", "models_per_view": 15,
            "view_means_percent": avg.reset_index().to_dict("records"),
            "all_30_checkpoint_view_mean_percent": z[cols].mean().mul(100).to_dict()}


def main() -> None:
    style()
    before = {name: sha256(path) for name, path in INPUTS.items()}
    tables = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    audit = validate(tables["aggregation"], tables["calibrated"], tables["gamma"])
    summary = {
        "protocol": "Descriptive re-expression of existing tables; no fitting, training, resampling, or confidence intervals",
        "model_scope": "15 raw-trained checkpoints: 3 backbones x 5 seeds",
        "cross_table_audit": audit,
        "inputs": {name: {"path": str(path), "sha256": before[name], "rows": len(tables[name])}
                   for name, path in INPUTS.items()},
        "figures": {
            "cdf_normalization_effects": normalization_effects(tables["aggregation"], tables["calibrated"]),
            "aggregation_sensitivity": aggregation_sensitivity(tables["aggregation"]),
            "gamma_curves": gamma_curves(tables["gamma"]),
            "threshold_operating_points": threshold_points(tables["threshold"]),
        },
        "software": {"python": __import__("platform").python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    after = {name: sha256(path) for name, path in INPUTS.items()}
    assert before == after, "An input table changed during rendering."
    (OUT / "figure_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT / "input_sha256.json").write_text(json.dumps(summary["inputs"], ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_directory": str(OUT), "input_hashes_unchanged": before == after,
                      "figures": list(summary["figures"])}, indent=2))


if __name__ == "__main__":
    main()
