#!/usr/bin/env python3
"""Stage 05 — why the frozen threshold fails: score-shift decomposition.

Inputs : results/val_cache/source_video_scores.parquet   (stage 01)
         results/tables/04_target_video_scores.parquet   (stage 04)
         results/tables/04_transfer_pooled.csv           (stage 04, for R_transfer)
Outputs: results/tables/05_score_shift.csv
         results/tables/05_class_distributions.csv
         figures/05_score_distributions_<cell>.png
         figures/05_shift_decomposition.png
         figures/05_ba_vs_threshold_<cell>.png
GPU    : no.
Target-domain labels: yes (descriptive statistics of the target distribution
         only; nothing here selects a threshold or a model).

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/05_analyze_score_shift.py
    python .../05_analyze_score_shift.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PHASE / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from study_threshold.io import MODES, score_column  # noqa: E402
from study_threshold.reporting import (  # noqa: E402
    cell_id,
    new_figure,
    phase_paths,
    save_figure,
    write_table,
)
from study_threshold.score_shift import class_stats, classify_failure, shift_metrics  # noqa: E402
from study_threshold.thresholds import rate_curves  # noqa: E402

_stage01 = import_module("01_prepare_ffpp_validation")
_stage04 = import_module("04_evaluate_transfer")


def _plot_distributions(src, tgt, mode, cell, out_dir, thresholds):
    col = score_column(mode)
    fig, axes = new_figure(figsize=(11, 4))
    import matplotlib.pyplot as plt

    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (df, name) in zip(axes, ((src, "FF++ val (source)"), (tgt, "CDF (target)"))):
        for label, cls, color in ((0, "real", "tab:blue"), (1, "fake", "tab:red")):
            vals = df.loc[df.label == label, col].to_numpy(float)
            ax.hist(vals, bins=40, alpha=0.55, label=f"{cls} (n={vals.size})",
                    color=color, density=True)
        for rule, style in (("t_BA", "-"), ("t_FPR5", ":")):
            if rule in thresholds:
                ax.axvline(thresholds[rule], color="black", linestyle=style, linewidth=1.2,
                           label=f"{rule} (source)")
        ax.set_title(f"{name} — {cell} / {mode}")
        ax.set_xlabel("video score  mean_t(signed_logit)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"05_score_distributions_{cell}_{mode}.png")


def _plot_ba_vs_threshold(src, tgt, mode, cell, out_dir, thresholds):
    col = score_column(mode)
    fig, ax = new_figure(figsize=(7, 4.5))
    for df, name, color in ((src, "FF++ val (source)", "tab:blue"),
                            (tgt, "CDF (target)", "tab:red")):
        cands, tpr, tnr = rate_curves(df[col].to_numpy(float), df["label"].to_numpy(int))
        order = np.argsort(cands)
        ax.plot(cands[order], ((tpr + tnr) / 2.0)[order], label=name, color=color)
    for rule, style in (("t_BA", "-"), ("t_equal", "--"), ("t_FPR1", "-."), ("t_FPR5", ":")):
        if rule in thresholds:
            ax.axvline(thresholds[rule], linestyle=style, color="black", linewidth=1.0,
                       label=f"{rule} (source)")
    ax.set_xlabel("threshold on mean_t(signed_logit)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"BA vs threshold — {cell} / {mode}")
    ax.legend(fontsize=7)
    return save_figure(fig, out_dir / f"05_ba_vs_threshold_{cell}_{mode}.png")


def _plot_shift_decomposition(shift_df, out_dir):
    fig, ax = new_figure(figsize=(9, 4.5))
    width = 0.2
    modes = list(shift_df["mode"].unique())
    metrics = ["shift_real", "shift_fake", "common_shift", "differential_shift"]
    x = np.arange(len(metrics))
    for i, mode in enumerate(modes):
        sub = shift_df[shift_df["mode"] == mode]
        means = [sub[m].mean() for m in metrics]
        errs = [sub[m].std(ddof=1) if len(sub) > 1 else 0.0 for m in metrics]
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=mode)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylabel("FF++ val -> CDF shift (video score units)")
    ax.set_title("Score-shift decomposition (mean ± sd over checkpoints)")
    ax.legend()
    return save_figure(fig, out_dir / "05_shift_decomposition.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-figure-cells", type=int, default=3,
                    help="how many (checkpoint, mode) cells get per-cell figures")
    args = ap.parse_args()

    cfg = _stage01.load_cfg(Path(args.config) if args.config else None)
    paths = phase_paths(PHASE, cfg)
    backbones, seeds, _ = _stage01.resolve_dims(cfg, args.dry_run)
    ba_drop = float(cfg["protocol"]["ba_drop_threshold"])

    source = pd.read_parquet(paths["val_cache"] / _stage01.OUT_NAME)
    target = pd.read_parquet(paths["tables"] / _stage04.TARGET_CACHE)
    pooled = pd.read_csv(paths["tables"] / "04_transfer_pooled.csv")
    thresholds_tbl = pd.read_csv(paths["tables"] / "03_source_thresholds.csv")

    shift_rows, dist_rows = [], []
    figures_made = 0
    for backbone in backbones:
        for seed in seeds:
            ckpt = cell_id(backbone, seed)
            src = source[source.checkpoint_id == ckpt]
            tgt = target[target.checkpoint_id == ckpt]
            for mode in MODES:
                shift = shift_metrics(src, tgt, mode)
                ba_row = pooled[(pooled.checkpoint_id == ckpt) & (pooled["mode"] == mode)
                                & (pooled.rule == "t_BA")].iloc[0]
                shift.update({
                    "checkpoint_id": ckpt, "backbone": backbone, "seed": seed,
                    "r_transfer": float(ba_row.r_transfer),
                    "target_auc": float(ba_row.target_auc),
                    "target_ba_at_source_t": float(ba_row.target_ba),
                    "target_oracle_ba": float(ba_row.target_oracle_ba),
                    "target_fpr_at_source_t": float(ba_row.target_fpr),
                    "target_ece_uncalibrated": float(ba_row.target_ece_uncalibrated),
                    "target_ece_temperature": float(ba_row.target_ece_temperature),
                    "target_ece_platt": float(ba_row.target_ece_platt),
                })
                shift["dominant_cause"] = classify_failure(shift, ba_drop_threshold=ba_drop)
                shift_rows.append(shift)

                for domain, df in (("source", src), ("target", tgt)):
                    row = {"checkpoint_id": ckpt, "backbone": backbone, "seed": seed,
                           "mode": mode, "domain": domain}
                    row.update(class_stats(df, mode))
                    dist_rows.append(row)

                th_map = dict(zip(
                    thresholds_tbl[(thresholds_tbl.checkpoint_id == ckpt)
                                   & (thresholds_tbl["mode"] == mode)]["rule"],
                    thresholds_tbl[(thresholds_tbl.checkpoint_id == ckpt)
                                   & (thresholds_tbl["mode"] == mode)]["threshold"]))
                if figures_made < args.max_figure_cells * len(MODES):
                    _plot_distributions(src, tgt, mode, ckpt, paths["figures"], th_map)
                    _plot_ba_vs_threshold(src, tgt, mode, ckpt, paths["figures"], th_map)
                    figures_made += 1

                print(f"[shift] {ckpt} {mode:9s} shift_real={shift['shift_real']:+.2f} "
                      f"shift_fake={shift['shift_fake']:+.2f} "
                      f"common={shift['common_shift']:+.2f} "
                      f"diff={shift['differential_shift']:+.2f} "
                      f"scale_real={shift['scale_real']:.2f} "
                      f"W_real={shift['wasserstein_real']:.2f} -> {shift['dominant_cause']}")

    shift_df = pd.DataFrame(shift_rows)
    write_table(shift_df, paths["tables"] / "05_score_shift.csv")
    write_table(pd.DataFrame(dist_rows), paths["tables"] / "05_class_distributions.csv")
    _plot_shift_decomposition(shift_df, paths["figures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
