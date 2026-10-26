"""Paired seed-level significance tests for the paper's main comparisons.

Addresses the reviewer point that the reported AUC gains (+0.016 CDF,
+0.011 DFD) are ~one seed-level standard deviation and that no statistical
test across seeds was provided.

All tests are PAIRED by training seed: both compared modes are evaluated
with the same seed and the same eval data, so the relevant dispersion is
the SD of per-seed deltas, not the SD of absolute AUCs.

Inputs (relative to this experiment dir):
  results/auc_video_level_60rows.csv
  results/dfd_balanced_thresholds_45rows.csv

Run:  python scripts/seed_significance_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
SEEDS = [7, 42, 123, 999, 2048]
BACKBONES = ["CLIP", "DINOv2", "EfficientNet"]


def paired_report(deltas: pd.Series, label: str):
    """deltas indexed by seed (positive = paper's claim direction)."""
    x = deltas.to_numpy(dtype=float)
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        t, p_t = stats.ttest_1samp(x, 0.0)
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sd / np.sqrt(n))
        dz = mean / sd
    else:
        t, p_t, ci, dz = float("nan"), float("nan"), (float("nan"),) * 2, float("nan")
    wins = int((x > 0).sum())
    losses = int((x < 0).sum())
    try:
        w, p_w = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided",
                                method="exact")
    except ValueError:
        w, p_w = float("nan"), float("nan")
    print(f"\n## {label}")
    print(f"per-seed deltas: " + ", ".join(f"s{int(k)}={v:+.4f}" for k, v in deltas.items()))
    print(f"n={n}  mean={mean:+.4f}  SD={sd:.4f}  Cohen's dz={dz:.2f}")
    print(f"95% CI (paired t): [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"paired t-test: t({n-1})={t:.2f}, p={p_t:.4g}")
    print(f"Wilcoxon signed-rank (exact, two-sided): p={p_w:.4g}")
    print(f"sign consistency: {wins}/{n} wins, {losses}/{n} losses")
    return {"label": label, "n": n, "mean": mean, "sd": sd, "t": t, "p_t": p_t,
            "p_wilcoxon": p_w, "wins": wins, "losses": losses,
            "ci_lo": ci[0], "ci_hi": ci[1]}


def wide(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """pivot one domain's AUC to columns per inference mode."""
    if isinstance(df.index, pd.MultiIndex) or "backbone" in df.index.names:
        df = df.reset_index(drop=True)
    return df.pivot_table(index="seed", columns="inference_mode", values=domain)


def main():
    auc = pd.read_csv(HERE / "results" / "auc_video_level_60rows.csv")
    bal = pd.read_csv(HERE / "results" / "dfd_balanced_thresholds_45rows.csv")
    rows = []

    print("=" * 72)
    print("Seed-level SD of ABSOLUTE AUC (what the reviewer compared against)")
    print("=" * 72)
    for backbone in BACKBONES:
        sub = auc[auc.backbone == backbone]
        raw = sub[sub.inference_mode == "raw"].set_index("seed")
        print(f"{backbone:13s} CDF raw AUC across seeds: "
              f"{raw.cdf_auc.mean():.4f} ± {raw.cdf_auc.std(ddof=1):.4f}   "
              f"DFD raw AUC: {raw.dfd_auc.mean():.4f} ± {raw.dfd_auc.std(ddof=1):.4f}")

    # --- paired comparisons, per backbone -----------------------------------
    for backbone in BACKBONES:
        sub = auc[auc.backbone == backbone]
        w_cdf = wide(sub, "cdf_auc")
        w_dfd = wide(sub, "dfd_auc")
        print("\n" + "=" * 72)
        print(f"BACKBONE: {backbone}")
        print("=" * 72)
        rows.append(paired_report((w_cdf.direction - w_cdf.raw).dropna(),
                                  "CDF AUC: direction - raw (claim: direction wins)"))
        rows.append(paired_report((w_dfd.raw - w_dfd.direction).dropna(),
                                  "DFD AUC: raw - direction (claim: raw wins)"))
        rows.append(paired_report((w_cdf.raw - w_cdf.length).dropna(),
                                  "CDF AUC: raw - length (ordering claim raw > length)"))
        rows.append(paired_report((w_cdf.raw - w_cdf.norm).dropna(),
                                  "CDF AUC: rawtrain raw - normtrain norm"))
        rows.append(paired_report((w_dfd.norm - w_dfd.raw).dropna(),
                                  "DFD AUC: normtrain norm - rawtrain raw"))

    # --- seed-level pooled across backbones (n = 5 seeds) -------------------
    print("\n" + "=" * 72)
    print("POOLED ACROSS BACKBONES at seed level (mean of 3 backbones per seed, n=5)")
    print("  (15 seed x backbone cells share the same 5 seeds, so the seed is")
    print("   the independent unit; do NOT treat 15 cells as 15 independent pairs)")
    print("=" * 72)
    w = auc.pivot_table(index=["backbone", "seed"], columns="inference_mode",
                        values=["cdf_auc", "dfd_auc"])
    seed_mean = w.groupby("seed").mean()
    rows.append(paired_report((seed_mean[("cdf_auc", "direction")]
                               - seed_mean[("cdf_auc", "raw")]).dropna(),
                              "CDF AUC: direction - raw, seed-level pooled"))
    rows.append(paired_report((seed_mean[("dfd_auc", "raw")]
                               - seed_mean[("dfd_auc", "direction")]).dropna(),
                              "DFD AUC: raw - direction, seed-level pooled"))
    # pooled cells as 15 pairs, for the sign test only
    d_cdf = (w[("cdf_auc", "direction")] - w[("cdf_auc", "raw")]).dropna()
    d_dfd = (w[("dfd_auc", "raw")] - w[("dfd_auc", "direction")]).dropna()
    print(f"\n15-cell sign counts: CDF direction>raw {int((d_cdf>0).sum())}/15 "
          f"(exact sign test p={stats.binomtest(int((d_cdf>0).sum()), 15, 0.5).pvalue:.2g}); "
          f"DFD raw>direction {int((d_dfd>0).sum())}/15 "
          f"(exact sign test p={stats.binomtest(int((d_dfd>0).sum()), 15, 0.5).pvalue:.2g})")

    # --- DFD balanced accuracy (45rows) --------------------------------------
    print("\n" + "=" * 72)
    print("DFD balanced accuracy (balanced-threshold protocol)")
    print("=" * 72)
    pv = bal.pivot_table(index=["backbone", "seed"], columns="variant",
                         values="balanced_acc")
    for backbone in BACKBONES:
        s = pv.loc[backbone]
        rows.append(paired_report((s["normtrain_norm"] - s["raw"]).dropna(),
                                  f"DFD bal-acc: normtrain_norm - raw ({backbone})"))

    # --- the seed-999 CDF detail ---------------------------------------------
    print("\n" + "=" * 72)
    print("SEED 999, CDF, all three inference modes (the 'ordering reversal' cell)")
    print("=" * 72)
    s999 = auc[(auc.seed == 999) & (auc.train_mode == "raw")]
    print(s999[["backbone", "inference_mode", "cdf_auc"]]
          .pivot(index="backbone", columns="inference_mode", values="cdf_auc")
          .to_string())

    out = pd.DataFrame(rows)
    out_path = HERE / "reports" / "seed_significance_tests.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved numeric summary -> {out_path}")


if __name__ == "__main__":
    main()
