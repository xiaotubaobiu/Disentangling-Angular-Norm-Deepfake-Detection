"""Assemble rebuttal figures from EXISTING artifacts (no new experiments).

Figure 1: CLIP feature-norm distributions, CDF vs DFD (side-by-side).
  Source: figures/frame_level_bw/norm_histogram/clip_{cdf,dfd}_norm_hist.png
  (seed-42 raw-train checkpoint, frame-level, real vs fake)

Figure 2: per-seed paired AUC deltas from results/auc_video_level_60rows.csv.
  Left:  CDF, direction - raw   (claim: direction wins on CDF)
  Right: DFD, raw - direction   (claim: raw wins on DFD)

Run:  python scripts/make_rebuttal_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
FIGS = HERE / "figures"
OUT = FIGS / "rebuttal"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 42, 123, 999, 2048]
STYLE = {"CLIP": dict(marker="o", color="black"),
         "DINOv2": dict(marker="^", color="0.35"),
         "EfficientNet": dict(marker="s", color="0.6")}


def fig_norm_hist_pair():
    src = [FIGS / "frame_level_bw" / "norm_histogram" / name
           for name in ("clip_cdf_norm_hist.png", "clip_dfd_norm_hist.png")]
    imgs = [Image.open(p) for p in src]
    h = min(im.height for im in imgs)
    imgs = [im.resize((round(im.width * h / im.height), h)) for im in imgs]
    gap = 24
    canvas = Image.new("RGB", (sum(im.width for im in imgs) + gap, h), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    dst = OUT / "fig_norm_hist_clip_cdf_vs_dfd.png"
    canvas.save(dst)
    print(f"saved {dst}")


def fig_seed_deltas():
    auc = pd.read_csv(HERE / "results" / "auc_video_level_60rows.csv")
    raw = auc[auc.train_mode == "raw"].pivot_table(
        index=["backbone", "seed"], columns="inference_mode",
        values=["cdf_auc", "dfd_auc"])
    panels = [("cdf_auc", "direction", "raw",
               "CDF:  direction $-$ raw"),
              ("dfd_auc", "raw", "direction",
               "DFD:  raw $-$ direction")]
    pos = list(range(len(SEEDS)))
    dodge = {"CLIP": -0.16, "DINOv2": 0.0, "EfficientNet": 0.16}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    all_vals = []
    for ax, (domain, a, b, title) in zip(axes, panels):
        delta = raw[(domain, a)] - raw[(domain, b)]
        for backbone, st in STYLE.items():
            y = delta.loc[backbone].reindex(SEEDS)
            all_vals += list(y)
            ax.plot([p + dodge[backbone] for p in pos], y, linestyle="none",
                    markersize=7, markeredgecolor="white", **st)
            ax.hlines(y.mean(), -0.55, -0.28, color=st["color"], linewidth=2.5)
        ax.axhline(0, color="0.2", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("training seed")
        ax.set_xticks(pos)
        ax.set_xticklabels(SEEDS)
        ax.set_xlim(-0.75, 4.45)
    axes[0].set_ylabel("video-level AUC difference")
    lo = min(min(all_vals), 0)
    axes[0].set_ylim(lo - 0.003, max(all_vals) * 1.1)
    # the DFD DINOv2 seed-2048 difference is positive but negligible; label it
    # so the marker on the zero line is not misread as a tie/negative
    ax = axes[1]
    dv = raw[("dfd_auc", "raw")] - raw[("dfd_auc", "direction")]
    v = dv.loc["DINOv2", 2048]
    ax.annotate(f"{v:+.4f}", (pos[SEEDS.index(2048)], v),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
    handles = [plt.Line2D([], [], linestyle="none", markersize=7, **st)
               for st in STYLE.values()]
    fig.legend(handles, list(STYLE), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    dst = OUT / "fig_paired_seed_deltas.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    print(f"saved {dst}")


if __name__ == "__main__":
    fig_norm_hist_pair()
    fig_seed_deltas()
