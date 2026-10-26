#!/usr/bin/env python
"""Class-conditional feature-norm distributions across domains (reviewer response).

Demonstrates — rather than asserts — the paper's claim that the feature norm
behaves as a dataset-dependent scaling factor:

  1. Per (backbone, seed, domain, label): video-level mean-feature-norm
     moments, Cohen's d, and the norm-only AUC  P(||z||_fake > ||z||_real)
     computed as the Mann-Whitney rank statistic on per-video mean norms.
  2. Per-seed table showing the norm<->label association INVERTS on DFD
     (AUC < 0.5 in 8/15 seed x backbone cells) while it is consistently
     positive on FF++ and CDF — stronger evidence for the thesis than the
     ~0.50 DFD average previously reported.
  3. Figure: class-conditional norm distributions per backbone x domain
     (seed 42), for the revised manuscript.

Inputs (repo-relative):
  experiments/2026-07-21_frame_midterm_diagnosis/results/
      parallel_20260721_213914/per_video_scores.parquet   (mean_feature_norm)
  experiments/2026-07-21_frame_midterm_diagnosis/results/
      parallel_20260721_213914/per_frame_scores.parquet   (frame-level moments)
  results/auc_video_level_60rows.csv                      (phase-1 length AUC)

Outputs:
  results/norm_distribution_stats.csv
  results/norm_only_auc_per_seed.csv
  figures/norm_distribution_by_domain.png

Run:  python scripts/norm_distribution_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

RUN_DIR = ROOT / "experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914"
VIDEO_PQ = RUN_DIR / "per_video_scores.parquet"
FRAME_PQ = RUN_DIR / "per_frame_scores.parquet"
P1_AUC = PHASE / "results/auc_video_level_60rows.csv"
TAB = PHASE / "results"
FIG = PHASE / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

BACKBONES = ["clip", "dinov2", "efficientnet"]
DOMAINS = ["ffpp", "cdf", "dfd"]
PRETTY = {"clip": "CLIP", "dinov2": "DINOv2", "efficientnet": "EfficientNet",
          "ffpp": "FF++", "cdf": "CDF", "dfd": "DFD"}

con = duckdb.connect()


def video_table() -> pd.DataFrame:
    """One row per (checkpoint, domain, video): mean feature norm."""
    return con.execute(
        f"""
        SELECT backbone, seed, domain, method_id, video_id, label,
               mean_feature_norm AS video_norm, num_frames
        FROM '{VIDEO_PQ}'
        WHERE train_mode = 'raw' AND infer_mode = 'raw'
        """
    ).df()


def auc_mw(x_fake: np.ndarray, x_real: np.ndarray) -> float:
    """Mann-Whitney AUC = P(x_fake > x_real) + 0.5 P(equal)."""
    x = np.concatenate([x_fake, x_real])
    r = rankdata(x)
    n_f, n_r = len(x_fake), len(x_real)
    r_fake = r[:n_f].sum()
    return float((r_fake - n_f * (n_f + 1) / 2.0) / (n_f * n_r))


def cohens_d(m_f, s_f, n_f, m_r, s_r, n_r) -> float:
    s_pooled = np.sqrt(((n_f - 1) * s_f**2 + (n_r - 1) * s_r**2) / (n_f + n_r - 2))
    return float((m_f - m_r) / s_pooled) if s_pooled > 0 else float("nan")


def group_stats(v: pd.DataFrame) -> pd.DataFrame:
    """Video-level moments + norm-only AUC per (backbone, seed, domain)."""
    rows = []
    for (bb, sd, dom), g in v.groupby(["backbone", "seed", "domain"]):
        f = g[g.label == 1].video_norm.to_numpy()
        r = g[g.label == 0].video_norm.to_numpy()
        rows.append(dict(
            backbone=bb, seed=sd, domain=dom, level="video",
            n_fake=len(f), n_real=len(r),
            mean_norm_fake=f.mean(), sd_norm_fake=f.std(ddof=1),
            mean_norm_real=r.mean(), sd_norm_real=r.std(ddof=1),
            cohens_d=cohens_d(f.mean(), f.std(ddof=1), len(f),
                              r.mean(), r.std(ddof=1), len(r)),
            norm_auc=auc_mw(f, r),
        ))
    return pd.DataFrame(rows)


def frame_stats() -> pd.DataFrame:
    """Frame-level moments + AUC per (backbone, seed, domain) via SQL."""
    moments = con.execute(
        f"""
        SELECT backbone, seed, domain, label,
               count(*) AS n, avg(feature_norm) AS m, stddev_samp(feature_norm) AS s
        FROM '{FRAME_PQ}'
        WHERE train_mode = 'raw' AND infer_mode = 'raw'
        GROUP BY 1, 2, 3, 4
        """
    ).df()
    rows = []
    for (bb, sd, dom), g in moments.groupby(["backbone", "seed", "domain"]):
        fk = g[g.label == 1].iloc[0]
        rk = g[g.label == 0].iloc[0]
        rows.append(dict(
            backbone=bb, seed=sd, domain=dom, level="frame",
            n_fake=int(fk.n), n_real=int(rk.n),
            mean_norm_fake=fk.m, sd_norm_fake=fk.s,
            mean_norm_real=rk.m, sd_norm_real=rk.s,
            cohens_d=cohens_d(fk.m, fk.s, fk.n, rk.m, rk.s, rk.n),
            norm_auc=np.nan,   # filled below for seed 42 only (rank stat needs arrays)
        ))
    out = pd.DataFrame(rows)

    # Frame-level AUC for the seed-42 cells used in the figure.
    for bb in BACKBONES:
        for dom in DOMAINS:
            g = con.execute(
                f"""
                SELECT label, feature_norm FROM '{FRAME_PQ}'
                WHERE train_mode = 'raw' AND infer_mode = 'raw'
                  AND backbone = '{bb}' AND seed = 42 AND domain = '{dom}'
                """
            ).df()
            f = g[g.label == 1].feature_norm.to_numpy()
            r = g[g.label == 0].feature_norm.to_numpy()
            out.loc[(out.backbone == bb) & (out.seed == 42)
                    & (out.domain == dom), "norm_auc"] = auc_mw(f, r)
    return out


def figure(v: pd.DataFrame, stats: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REAL, FAKE = "#2a78d6", "#eb6834"   # categorical slots 1-2 (CVD-validated)
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"

    s42 = v[v.seed == 42]
    st42 = stats[(stats.seed == 42) & (stats.level == "video")]

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2))
    for i, bb in enumerate(BACKBONES):
        for j, dom in enumerate(DOMAINS):
            ax = axes[i, j]
            g = s42[(s42.backbone == bb) & (s42.domain == dom)]
            f = g[g.label == 1].video_norm.to_numpy()
            r = g[g.label == 0].video_norm.to_numpy()
            lo = min(f.min(), r.min())
            hi = max(f.max(), r.max())
            bins = np.linspace(lo, hi, 41)
            ax.hist(r, bins=bins, density=True, histtype="stepfilled",
                    facecolor=REAL, alpha=0.15, edgecolor=REAL, linewidth=1.8,
                    label="real")
            ax.hist(f, bins=bins, density=True, histtype="stepfilled",
                    facecolor=FAKE, alpha=0.15, edgecolor=FAKE, linewidth=1.8,
                    label="fake")
            row = st42[(st42.backbone == bb) & (st42.domain == dom)].iloc[0]
            flip = row.norm_auc < 0.5
            ax.text(0.97, 0.95,
                    f"AUC {row.norm_auc:.3f}\nd {row.cohens_d:+.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color=INK2, fontweight="bold" if flip else "normal")
            ax.set_title(f"{PRETTY[bb]}  ·  {PRETTY[dom]}", fontsize=10,
                         color=INK, loc="left")
            ax.tick_params(labelsize=8, colors=INK2)
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            if j == 0:
                ax.set_ylabel("video density", fontsize=9, color=INK2)
            if i == 2:
                ax.set_xlabel("per-video mean feature norm  ||z||", fontsize=9,
                              color=INK2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = FIG / "norm_distribution_by_domain.png"
    fig.savefig(out, dpi=200)
    plt.close("all")
    print(f"[norm] wrote {out}")


def main():
    v = video_table()
    print(f"[norm] video rows: {len(v)}; "
          f"cells: {v.groupby(['backbone','seed','domain']).ngroups}")

    stats_v = group_stats(v)
    stats_f = frame_stats()
    stats = pd.concat([stats_v, stats_f], ignore_index=True)

    # attach the phase-1 paper-table length AUC for cross-validation
    p1 = pd.read_csv(P1_AUC)
    p1["backbone"] = p1.backbone.map(
        {"CLIP": "clip", "DINOv2": "dinov2", "EfficientNet": "efficientnet"})
    p1_len = (p1[p1.inference_mode == "length"]
              [["backbone", "train_mode", "seed", "cdf_auc", "dfd_auc", "ffpp_auc"]])
    p1_len = p1_len[p1_len.train_mode == "raw"].drop(columns="train_mode")
    stats = stats.merge(p1_len, on=["backbone", "seed"], how="left")
    stats["phase1_length_auc"] = stats.apply(
        lambda r: r[f"{r.domain}_auc"] if r.level == "video" else np.nan, axis=1)
    stats = stats.drop(columns=["ffpp_auc", "cdf_auc", "dfd_auc"])

    stats["abs_d_phase1_minus_auc"] = (
        (stats.phase1_length_auc - stats.norm_auc).abs().where(stats.level == "video"))
    stats.to_csv(TAB / "norm_distribution_stats.csv", index=False)
    print(f"[norm] wrote {TAB / 'norm_distribution_stats.csv'}")

    # per-seed video-level AUC pivot (the reviewer-facing table)
    pv = (stats[stats.level == "video"]
          .pivot_table(index=["backbone", "seed"], columns="domain",
                       values="norm_auc")
          .round(4))
    pv["ffpp_sign"] = np.where(pv.ffpp > 0.5, "+", "FLIP")
    pv["cdf_sign"] = np.where(pv.cdf > 0.5, "+", "FLIP")
    pv["dfd_sign"] = np.where(pv.dfd > 0.5, "+", "FLIP")
    pv.to_csv(TAB / "norm_only_auc_per_seed.csv")
    print(f"[norm] wrote {TAB / 'norm_only_auc_per_seed.csv'}")

    print("\n=== video-level norm-only AUC: P(||z||_fake > ||z||_real) ===")
    print(pv.to_string())
    n_flip = {d: int((pv[f"{d}_sign"] == "FLIP").sum()) for d in DOMAINS}
    print(f"\nAUC<0.5 (sign flip) counts: {n_flip}")

    print("\n=== validation vs phase-1 length AUC (video level) ===")
    chk = stats[stats.level == "video"][["backbone", "seed", "domain",
                                         "norm_auc", "phase1_length_auc"]]
    chk["abs_diff"] = (chk.norm_auc - chk.phase1_length_auc).abs()
    print(chk.sort_values("abs_diff").round(4).to_string(index=False))

    figure(v, stats)


if __name__ == "__main__":
    main()
