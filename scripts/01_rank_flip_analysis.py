# scripts/01_rank_flip_analysis.py
"""Q1 (spec §4): rank-flip identity + H/B anatomy + partial-norm gamma.

Two views:
  - frame: signed_logit per frame (rank-identical to prob_fake).
  - video: mean(sigmoid(signed)) per video == official video_prob_fake.
The direction>raw phenomenon lives at the VIDEO (mean-sigmoid) level; at frame
level raw wins. Both reported. Gamma evaluated at video level (primary).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.auc_pairs import auc_mannwhitney, flip_counts, tie_rate, identity_check
from deepfake_detection.study.flip_dissect import recover_wdotu_b, score_gamma

CFG = load_config(ROOT/"configs"/"study_config.yaml")
ALL = sum(CFG.fake_methods.values(), [])
TAB = ROOT/CFG.paths["tables"]; FIG = ROOT/CFG.paths["figures"]
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
CAP = 5_000_000

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x, float)))

def _cap(a, b, seed):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size*b.size > CAP:
        rng = np.random.default_rng(seed)
        na = max(1, int(CAP//max(b.size,1))); nb = max(1, int(CAP//max(a.size,1)))
        if a.size>na: a = rng.choice(a, size=na, replace=False)
        if b.size>nb: b = rng.choice(b, size=nb, replace=False)
    return a, b

def _flip_block(fr_raw, rr_raw, fr_dir, rr_dir, fnorm_fake, fnorm_real, seed):
    """Frame-level OR video-level flip analysis on aligned fake/real arrays.
    Returns auc_raw, auc_dir, H/B/CC/WW, identity, tie, plus H-class & B-class
    mean(fake_norm - real_norm) over the cartesian pair set."""
    fr, rr = _cap(fr_raw, rr_raw, seed); fdr, rdr = _cap(fr_dir, rr_dir, seed)
    auc_raw = auc_mannwhitney(fr, rr); auc_dir = auc_mannwhitney(fdr, rdr)
    cc, ww, h, b = flip_counts(fr, rr, fdr, rdr)
    # cartesian norm anatomy (subsampled consistently with cap is hard; use raw arrays, cap separately)
    nf = np.asarray(fnorm_fake, float)[:, None]; nr = np.asarray(fnorm_real, float)[None, :]
    rd = np.asarray(fr_raw, float)[:, None] - np.asarray(rr_raw, float)[None, :]
    dd = np.asarray(fr_dir, float)[:, None] - np.asarray(rr_dir, float)[None, :]
    raw_ok = rd > 0; dir_ok = dd > 0
    Hmask = dir_ok & ~raw_ok; Bmask = ~dir_ok & raw_ok
    ndiff = nf - nr
    h_norm = float(np.mean(ndiff[Hmask])) if Hmask.any() else float("nan")
    b_norm = float(np.mean(ndiff[Bmask])) if Bmask.any() else float("nan")
    return dict(auc_raw=auc_raw, auc_dir=auc_dir, delta_auc=auc_dir-auc_raw,
                CC=cc, WW=ww, H=h, B=b, H_minus_B=h-b,
                identity_error=identity_check(auc_raw, auc_dir, h, b),
                tie_rate=tie_rate(fr, rr), H_normdiff=h_norm, B_normdiff=b_norm)

def _vid_table(df_w, signed, norm):
    """Per-video mean(sigmoid(signed)) and mean(norm)."""
    p = _sigmoid(df_w[signed].to_numpy())
    n = df_w[norm].to_numpy(float)
    tmp = pd.DataFrame({"video_id": df_w.video_id.to_numpy(),
                        "label": df_w.label.to_numpy(), "p": p, "n": n})
    return tmp.groupby("video_id").agg(p=("p","mean"), n=("n","mean"),
                                       label=("label","first")).reset_index()

def _wide(df):
    raw = df[df.infer_mode=="raw"][["checkpoint_id","backbone","seed","domain","method_id",
        "label","video_id","image_path","feature_norm","source_video_id","signed_logit"]].copy()
    raw = raw.rename(columns={"signed_logit":"s_raw"})
    d = df[df.infer_mode=="direction"][["image_path","signed_logit"]].rename(columns={"signed_logit":"s_dir"})
    return raw.merge(d, on="image_path")

def run_one(bb, sd, fpath, methods):
    rows = []
    for method in methods:
        for dom in CFG.domains:
            df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                backbones=[bb], seeds=[sd], domains=[dom], methods=[method,"real"]))
            df_w = _wide(df)
            fake = df_w[df_w.label==1]; real = df_w[df_w.label==0]
            if fake.empty or real.empty: continue
            base = dict(backbone=bb, seed=sd, method=method, domain=dom)
            # FRAME view (signed_logit)
            fm = _flip_block(fake.s_raw.to_numpy(), real.s_raw.to_numpy(),
                             fake.s_dir.to_numpy(), real.s_dir.to_numpy(),
                             fake.feature_norm.to_numpy(), real.feature_norm.to_numpy(),
                             CFG.global_seed)
            rows.append({**base, "view":"frame", **fm})
            # VIDEO view (mean sigmoid) -- the phenomenon lives here
            vr = _vid_table(df_w, "s_raw", "feature_norm")
            vd = _vid_table(df_w, "s_dir", "feature_norm")
            m = vr.merge(vd, on="video_id", suffixes=("_r","_d"))
            fk = m[m.label_r==1]; rl = m[m.label_r==0]
            vm = _flip_block(fk.p_r.to_numpy(), rl.p_r.to_numpy(),
                             fk.p_d.to_numpy(), rl.p_d.to_numpy(),
                             fk.n_r.to_numpy(), rl.n_r.to_numpy(),
                             CFG.global_seed+1)
            rows.append({**base, "view":"video", **vm})
    return pd.DataFrame(rows)

def dissect_and_gamma(bb, sd, fpath):
    curve=[]; stat=[]
    for method in ALL:
        for dom in CFG.domains:
            df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                backbones=[bb], seeds=[sd], domains=[dom], methods=[method,"real"]))
            df_w = _wide(df)
            fake = df_w[df_w.label==1]; real = df_w[df_w.label==0]
            if fake.empty or real.empty: continue
            wu, bb_ = recover_wdotu_b(df_w.s_raw.to_numpy(), df_w.s_dir.to_numpy(),
                                      df_w.feature_norm.to_numpy())
            # gamma curve at VIDEO level (mean sigmoid)
            vt0 = _vid_table(df_w, "s_raw", "feature_norm")  # template for video grouping
            best_g, best_a = None, -1.0
            for g in CFG.gamma_grid:
                s = score_gamma(wu, bb_, df_w.feature_norm.to_numpy(), g)  # per-frame signed
                p = _sigmoid(s)
                tmp = pd.DataFrame({"video_id": df_w.video_id.to_numpy(),
                                    "label": df_w.label.to_numpy(), "p": p})
                v = tmp.groupby("video_id").agg(p=("p","mean"), label=("label","first")).reset_index()
                a = auc_mannwhitney(v[v.label==1].p.to_numpy(), v[v.label==0].p.to_numpy())
                curve.append(dict(backbone=bb,seed=sd,method=method,domain=dom,gamma=g,video_auc=a))
                if a > best_a: best_a, best_g = a, g
            rho_mean = float(np.log(real.feature_norm.to_numpy()+1e-12).mean()
                             - np.log(fake.feature_norm.to_numpy()+1e-12).mean())
            stat.append(dict(backbone=bb,seed=sd,method=method,domain=dom,
                             rho_mean=rho_mean, gamma_best_video=best_g, gamma_best_auc=best_a))
    return pd.DataFrame(curve), pd.DataFrame(stat)

def main():
    fpath = str(ROOT/CFG.parquet["frame"])
    self_df = run_one(CFG.smoke["backbone"], CFG.smoke["seed"], fpath, CFG.smoke["methods"])
    print(self_df[["method","domain","view","delta_auc","identity_error"]].to_string())
    worst = float(self_df.identity_error.max())
    assert worst < CFG.identity_tol, f"identity self-check FAILED: {worst}"
    # sanity: video view should show direction>raw for at least the aggregate trend on CDF
    print(f"[self-check] max identity_error={worst:.2e} < tol -> OK")
    frames = [run_one(bb, sd, fpath, ALL) for bb in CFG.backbones for sd in CFG.seeds]
    q1 = pd.concat(frames, ignore_index=True)
    q1.to_csv(TAB/"Q1_rank_flip.csv", index=False)
    curves=[]; stats=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            c,s = dissect_and_gamma(bb, sd, fpath); curves.append(c); stats.append(s)
    pd.concat(curves, ignore_index=True).to_csv(TAB/"Q1_partial_norm_curve.csv", index=False)
    pd.concat(stats, ignore_index=True).to_csv(TAB/"Q1_flip_dissection.csv", index=False)
    _plots(q1)
    print("Q1 done. rows:", len(q1))

def _plots(q1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    v = q1[q1.view=="video"]
    fig, ax = plt.subplots(figsize=(8,5))
    for m,col in [("fsgan","C0"),("sadtalker","C1")]:
        s = v[(v.method==m)&(v.domain=="cdf")]
        if not s.empty:
            ax.errorbar([s.H.mean()],[f"{m} H"],xerr=[s.H.std()],fmt="o",color=col)
            ax.errorbar([s.B.mean()],[f"{m} B"],xerr=[s.B.std()],fmt="s",color=col)
    ax.set_title("Q1 VIDEO H/B FSGAN vs SadTalker (CDF)"); fig.tight_layout()
    fig.savefig(FIG/"Q1_H_B_FSGAN_SadTalker.png", dpi=120)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(v.H_minus_B, v.delta_auc, s=12)
    lo=min(v.H_minus_B.min(),v.delta_auc.min()); hi=max(v.H_minus_B.max(),v.delta_auc.max())
    ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_xlabel("H-B"); ax.set_ylabel("AUC_dir-AUC_raw")
    ax.set_title("VIDEO identity: H-B vs delta AUC"); fig.tight_layout()
    fig.savefig(FIG/"Q1_H_minus_B_vs_delta_auc.png", dpi=120)
    ff=q1[q1.view=="video"]
    fig,ax=plt.subplots(figsize=(6,5))
    for m,col in [("fsgan","C0"),("sadtalker","C1")]:
        s=ff[ff.method==m]
        ax.scatter(s[s.domain=="ffpp"].H_minus_B, s[s.domain=="ffpp"].delta_auc,color=col,marker="o",label=f"{m} ffpp")
        ax.scatter(s[s.domain=="cdf"].H_minus_B, s[s.domain=="cdf"].delta_auc,color=col,marker="x",label=f"{m} cdf")
    ax.set_xlabel("H-B"); ax.set_ylabel("delta AUC"); ax.set_title("Q1 VIDEO FF++ vs CDF rank flip"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"Q1_FFPP_vs_CDF_rank_flip.png", dpi=120)
    plt.close("all")

if __name__=="__main__":
    main()
