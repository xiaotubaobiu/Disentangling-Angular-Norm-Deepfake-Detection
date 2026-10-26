"""Balanced CDF direction - raw AUC heatmap, per backbone.

For each (backbone, seed, method), DOWNSAMPLE fake videos to match the real
count (178 per ckpt) with a fixed RNG seed, so every method is evaluated on
exactly the same number of fake and real videos. This removes any sample-size
imbalance artefact.

Produces:
  - 3 per-backbone heatmaps (16 methods × 5 seeds) of delta AUC
  - 1 pooled bar chart
  - cdf_balanced_delta_auc.csv (the numbers behind the heatmaps)

Usage:
    python scripts/cdf_balanced_dir_raw.py \\
        experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


METHOD_ORDER = [
    "simswap", "inswap", "blendface", "faceswap",
    "fsgan", "mobileswap", "e4s", "facedancer",
    "fomm", "facevid2vid", "wav2lip", "sadtalker",
    "MRAA", "pirender", "tpsm", "lia",
]
SEED_ORDER = [7, 42, 123, 999, 2048]
BACKBONES = ["clip", "dinov2", "efficientnet"]
RNG_SEED = 20260722  # fixed for reproducibility


def _paired_balanced_auc(base: pd.DataFrame,
                         n_per_class: int,
                         rng: np.random.Generator,
                         method: str | None = None) -> tuple[float, float, int, int]:
    """Paired balanced AUC for raw and direction on the SAME video subsample.

    raw and direction rows of one (backbone, seed) cover the identical set of
    (method_id, video_id) pairs, so after sorting by (method_id, video_id) row
    i of each mode is the same video. We choose ONE downsampled index set (per
    class) and apply it to both modes — this keeps the delta paired. Passing
    the rng separately to two independent draws (the previous behaviour) picked
    different fake subsets for raw vs direction and broke the pairing.

    method=None pools all fake methods; otherwise restricts fakes to `method`.
    Returns (auc_raw, auc_direction, n_fake_used, n_real_used).
    """
    raw = base[base["infer_mode"] == "raw"].sort_values(
        ["method_id", "video_id"]).reset_index(drop=True)
    dire = base[base["infer_mode"] == "direction"].sort_values(
        ["method_id", "video_id"]).reset_index(drop=True)
    if len(raw) == 0 or len(raw) != len(dire):
        return float("nan"), float("nan"), 0, 0

    is_real = (raw["method_id"] == "real").to_numpy()
    is_fake = (raw["method_id"] != "real").to_numpy() if method is None \
        else (raw["method_id"] == method).to_numpy()
    fake_idx = np.where(is_fake)[0]
    real_idx = np.where(is_real)[0]
    if len(fake_idx) == 0 or len(real_idx) == 0:
        return float("nan"), float("nan"), len(fake_idx), len(real_idx)

    if len(fake_idx) > n_per_class:
        fake_idx = rng.choice(fake_idx, size=n_per_class, replace=False)
    if len(real_idx) > n_per_class:
        real_idx = rng.choice(real_idx, size=n_per_class, replace=False)

    sel = np.concatenate([fake_idx, real_idx])
    labels = np.concatenate([np.ones(len(fake_idx)), np.zeros(len(real_idx))])
    raw_scores = raw["video_prob_fake"].to_numpy()[sel]
    dir_scores = dire["video_prob_fake"].to_numpy()[sel]
    return (
        float(roc_auc_score(labels, raw_scores)),
        float(roc_auc_score(labels, dir_scores)),
        len(fake_idx),
        len(real_idx),
    )


def compute_balanced(pv: pd.DataFrame, n_per_class: int = 178) -> pd.DataFrame:
    """Per-method balanced delta AUC (direction - raw) on CDF."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    pv_raw = pv[(pv["domain"] == "cdf")
                & (pv["train_mode"] == "raw")
                & (pv["infer_mode"].isin(["raw", "direction"]))]

    for backbone in BACKBONES:
        for seed in SEED_ORDER:
            base = pv_raw[(pv_raw["backbone"] == backbone)
                          & (pv_raw["seed"] == seed)]

            for method in METHOD_ORDER:
                if base[(base["infer_mode"] == "raw")
                        & (base["method_id"] == method)].empty:
                    continue
                # Shared subsample applied to both raw and direction (paired).
                auc_raw, auc_dir, n_fake, n_real = _paired_balanced_auc(
                    base, n_per_class, rng, method=method
                )
                rows.append({
                    "backbone": backbone,
                    "seed": seed,
                    "method_id": method,
                    "n_fake_used": n_fake,
                    "n_real_used": n_real,
                    "auc_raw": auc_raw,
                    "auc_direction": auc_dir,
                    "delta": auc_dir - auc_raw,
                })
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, backbone: str, out: Path) -> None:
    sub = df[df["backbone"] == backbone]
    pivot = sub.pivot_table(index="method_id", columns="seed",
                            values="delta", aggfunc="mean")
    pivot = pivot.reindex(index=[m for m in METHOD_ORDER if m in pivot.index])
    pivot = pivot.reindex(columns=SEED_ORDER)
    vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 0.02)

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        pivot, annot=True, fmt="+.4f",
        cmap="RdBu_r", center=0.0, vmin=-vmax, vmax=+vmax,
        cbar_kws={"label": "Δ AUC (direction − raw)"},
        ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_title(f"CDF (balanced {int(sub['n_real_used'].iloc[0])}v{int(sub['n_fake_used'].iloc[0])}) · "
                 f"{backbone.upper()} · Direction − Raw AUC", fontsize=11)
    ax.set_ylabel("Forgery method")
    ax.set_xlabel("Seed")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    vals = pivot.values
    print(f"  {out.name}: direction>raw = {int((vals>0).sum())}/{vals.size}, "
          f"Δ∈[{np.nanmin(vals):+.4f},{np.nanmax(vals):+.4f}], "
          f"mean={np.nanmean(vals):+.4f}")


def plot_pooled(pv: pd.DataFrame, out: Path, n_per_class: int = 178) -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    pv_raw = pv[(pv["domain"] == "cdf")
                & (pv["train_mode"] == "raw")
                & (pv["infer_mode"].isin(["raw", "direction"]))]
    for backbone in BACKBONES:
        for seed in SEED_ORDER:
            base = pv_raw[(pv_raw["backbone"] == backbone) & (pv_raw["seed"] == seed)]
            # Pooled (all fakes), paired subsample shared by both modes.
            auc_raw, auc_dir, _, _ = _paired_balanced_auc(
                base, n_per_class, rng, method=None
            )
            rows.append({"backbone": backbone, "seed": seed, "mode": "raw", "auc": auc_raw})
            rows.append({"backbone": backbone, "seed": seed, "mode": "direction", "auc": auc_dir})
    df = pd.DataFrame(rows)
    raw = df[df["mode"] == "raw"].set_index(["backbone", "seed"])["auc"]
    dire = df[df["mode"] == "direction"].set_index(["backbone", "seed"])["auc"]
    delta = (dire - raw).reset_index()
    delta.columns = ["backbone", "seed", "delta"]

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = delta.pivot_table(index="backbone", columns="seed", values="delta")
    pivot = pivot.reindex(index=BACKBONES, columns=SEED_ORDER)
    pivot.plot(kind="bar", ax=ax, edgecolor="white", width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ AUC (direction − raw)")
    ax.set_title(f"CDF (balanced {n_per_class}v{n_per_class}) · Pooled · Direction − Raw AUC")
    ax.set_xlabel("Backbone")
    ax.legend(title="Seed", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}: pooled delta by backbone:")
    for bb in BACKBONES:
        v = delta[delta["backbone"] == bb]["delta"]
        print(f"    {bb}: mean Δ = {v.mean():+.4f} (±{v.std():.4f})")


def main():
    if len(sys.argv) < 2:
        print("Usage: cdf_balanced_dir_raw.py <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    pv_path = run_dir / "per_video_scores.parquet"
    pv = pd.read_parquet(pv_path)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"=== Balanced (178v178) per-method delta AUC on CDF ===")
    df = compute_balanced(pv, n_per_class=178)
    df.to_csv(run_dir / "cdf_balanced_delta_auc.csv", index=False)
    print(f"  {len(df)} cells\n")
    for bb in BACKBONES:
        plot_heatmap(df, bb, fig_dir / f"balanced_heatmap_cdf_{bb}.png")
    print()
    print("=== Balanced pooled delta AUC ===")
    plot_pooled(pv, fig_dir / "balanced_heatmap_cdf_pooled.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
