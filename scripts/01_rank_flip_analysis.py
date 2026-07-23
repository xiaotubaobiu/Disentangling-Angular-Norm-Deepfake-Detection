# scripts/01_rank_flip_analysis.py
"""Q1 (spec §4): rank-flip identity self-check, frame_full + video_mean H/B,
flip dissection (rho), partial-norm gamma curve. Parquet-only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.auc_pairs import (auc_mannwhitney, flip_counts,
    tie_rate, identity_check, aggregate_video_scores)
from deepfake_detection.study.flip_dissect import (recover_wdotu_b, score_gamma,
    optimal_gamma)

CFG = load_config(ROOT/"configs"/"study_config.yaml")
ALL = sum(CFG.fake_methods.values(), [])
TAB = ROOT/CFG.paths["tables"]; FIG = ROOT/CFG.paths["figures"]
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
CAP = 5_000_000  # max fake*real pairs per broadcast

def _cap(a, b, seed):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size * b.size > CAP:
        rng = np.random.default_rng(seed)
        na = max(1, int(CAP // max(b.size, 1)))
        nb = max(1, int(CAP // max(a.size, 1)))
        if a.size > na: a = rng.choice(a, size=na, replace=False)
        if b.size > nb: b = rng.choice(b, size=nb, replace=False)
    return a, b

def _wide(df):
    raw = df[df.infer_mode=="raw"][["checkpoint_id","backbone","seed","domain",
        "method_id","label","video_id","source_video_id","image_path",
        "feature_norm","signed_logit"]].copy()
    raw = raw.rename(columns={"signed_logit":"s_raw"})
    d = df[df.infer_mode=="direction"][["image_path","signed_logit"]].rename(
        columns={"signed_logit":"s_dir"})
    return raw.merge(d, on="image_path")

def _metrics(fr_raw, rr_raw, fr_dir, rr_dir, seed):
    fr, rr = _cap(fr_raw, rr_raw, seed)
    fdr, rdr = _cap(fr_dir, rr_dir, seed)
    auc_raw = auc_mannwhitney(fr, rr); auc_dir = auc_mannwhitney(fdr, rdr)
    cc, ww, h, b = flip_counts(fr, rr, fdr, rdr)
    return dict(auc_raw=auc_raw, auc_dir=auc_dir, delta_auc=auc_dir-auc_raw,
                CC=cc, WW=ww, H=h, B=b, H_minus_B=h-b,
                identity_error=identity_check(auc_raw, auc_dir, h, b),
                tie_rate=tie_rate(fr, rr))

def run_one(bb, sd, fpath, methods):
    rows = []
    for method in methods:
        for dom in CFG.domains:
            df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                backbones=[bb], seeds=[sd], domains=[dom], methods=[method, "real"]))
            df_w = _wide(df)
            fake = df_w[df_w.label == 1]; real = df_w[df_w.label == 0]
            if fake.empty or real.empty:
                continue
            base = dict(backbone=bb, seed=sd, method=method, domain=dom)
            m = _metrics(fake.s_raw.to_numpy(), real.s_raw.to_numpy(),
                         fake.s_dir.to_numpy(), real.s_dir.to_numpy(), CFG.global_seed)
            rows.append({**base, "view": "frame_full", **m})
            vr = aggregate_video_scores(df_w.assign(score=df_w.s_raw), how="mean")
            vd = aggregate_video_scores(df_w.assign(score=df_w.s_dir), how="mean")
            m2 = _metrics(vr[vr.label == 1].video_score.to_numpy(),
                          vr[vr.label == 0].video_score.to_numpy(),
                          vd[vd.label == 1].video_score.to_numpy(),
                          vd[vd.label == 0].video_score.to_numpy(), CFG.global_seed + 1)
            rows.append({**base, "view": "video_mean", **m2})
    return pd.DataFrame(rows)

def dissect_and_gamma(bb, sd, fpath):
    rows = []
    for method in ALL:
        for dom in CFG.domains:
            df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                backbones=[bb], seeds=[sd], domains=[dom], methods=[method, "real"]))
            df_w = _wide(df)
            fake = df_w[df_w.label == 1]; real = df_w[df_w.label == 0]
            if fake.empty or real.empty:
                continue
            wu, bb_ = recover_wdotu_b(df_w.s_raw.to_numpy(), df_w.s_dir.to_numpy(),
                                      df_w.feature_norm.to_numpy())
            fmask = df_w.label.to_numpy().astype(bool)
            gbest, _ = optimal_gamma(wu, bb_, df_w.feature_norm.to_numpy(), fmask,
                                     CFG.gamma_grid, auc_mannwhitney)
            for g in CFG.gamma_grid:
                s = score_gamma(wu, bb_, df_w.feature_norm.to_numpy(), g)
                rows.append(dict(backbone=bb, seed=sd, method=method, domain=dom,
                                 gamma=g, auc=float(auc_mannwhitney(s[fmask], s[~fmask]))))
            # rho mean via linearity: E[log(real)] - E[log(fake)]
            rho_mean = float(np.log(real.feature_norm.to_numpy()+1e-12).mean()
                             - np.log(fake.feature_norm.to_numpy()+1e-12).mean())
            rows.append(dict(backbone=bb, seed=sd, method=method, domain=dom,
                             stat="rho_mean", val=rho_mean, gamma_best=gbest))
    return pd.DataFrame(rows)

def main():
    fpath = str(ROOT/CFG.parquet["frame"])
    # 1) self-check
    self_df = run_one(CFG.smoke["backbone"], CFG.smoke["seed"], fpath, CFG.smoke["methods"])
    print(self_df[["method", "domain", "view", "identity_error"]].to_string())
    worst = float(self_df.identity_error.max())
    assert worst < CFG.identity_tol, f"identity self-check FAILED: max err={worst}"
    print(f"[self-check] max identity_error={worst:.2e} < tol={CFG.identity_tol} -> OK")
    # 2) full expansion (all combos, all methods)
    frames = [run_one(bb, sd, fpath, ALL) for bb in CFG.backbones for sd in CFG.seeds]
    q1 = pd.concat(frames, ignore_index=True)
    q1.to_csv(TAB/"Q1_rank_flip.csv", index=False)
    # 3) dissection + gamma
    dg = pd.concat([dissect_and_gamma(bb, sd, fpath)
                    for bb in CFG.backbones for sd in CFG.seeds], ignore_index=True)
    dg[dg.gamma.notna()][["backbone","seed","method","domain","gamma","auc"]].to_csv(
        TAB/"Q1_partial_norm_curve.csv", index=False)
    dg[dg.stat.notna()].to_csv(TAB/"Q1_flip_dissection.csv", index=False)
    _plots(q1)
    print("Q1 done. rows:", len(q1))

def _plots(q1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, col in [("fsgan", "C0"), ("sadtalker", "C1")]:
        s = q1[(q1.method == m) & (q1.view == "frame_full") & (q1.domain == "cdf")]
        if not s.empty:
            ax.errorbar([s.H.mean()], [f"{m} H"], xerr=[s.H.std()], fmt="o", color=col)
            ax.errorbar([s.B.mean()], [f"{m} B"], xerr=[s.B.std()], fmt="s", color=col)
    ax.set_title("Q1 H/B FSGAN vs SadTalker (CDF)"); fig.tight_layout()
    fig.savefig(FIG/"Q1_H_B_FSGAN_SadTalker.png", dpi=120)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(q1.H_minus_B, q1.delta_auc, s=10)
    lo = min(q1.H_minus_B.min(), q1.delta_auc.min()); hi = max(q1.H_minus_B.max(), q1.delta_auc.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("H-B"); ax.set_ylabel("AUC_dir - AUC_raw"); ax.set_title("identity: H-B vs delta AUC")
    fig.tight_layout(); fig.savefig(FIG/"Q1_H_minus_B_vs_delta_auc.png", dpi=120)
    ff = q1[q1.view == "frame_full"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, col in [("fsgan", "C0"), ("sadtalker", "C1")]:
        s = ff[ff.method == m]
        ax.scatter(s[s.domain=="ffpp"].H_minus_B, s[s.domain=="ffpp"].delta_auc, color=col, marker="o", label=f"{m} ffpp")
        ax.scatter(s[s.domain=="cdf"].H_minus_B, s[s.domain=="cdf"].delta_auc, color=col, marker="x", label=f"{m} cdf")
    ax.set_xlabel("H-B"); ax.set_ylabel("delta AUC"); ax.set_title("Q1 FF++ vs CDF rank flip"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"Q1_FFPP_vs_CDF_rank_flip.png", dpi=120)
    plt.close("all")

if __name__ == "__main__":
    main()
