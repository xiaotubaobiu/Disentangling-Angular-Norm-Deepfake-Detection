# scripts/03_length_robustness.py
"""Q3 (spec §6): length-only robustness; checkpoint-level global orientation.

Orientation source = FF++ data in parquet (= FF++ test split; see length.py
LIMITATION). Sign fixed on FF++, applied to CDF; never flipped by CDF."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
ROOT = PHASE.parents[1]                              # repository root; sys.path.insert(0,str(ROOT/"src"))
from study.config import load_config
from study.io import load_frame_subset, dedup_reals
from study.length import (orient_sign, micro_auc, macro_auc,
    video_weighted_macro, video_auc, equalized_auc)
from study.auc_pairs import auc_mannwhitney

CFG=load_config(PHASE/"configs"/"study_config.yaml")
ALL=sum(CFG.fake_methods.values(),[])
TAB=PHASE/CFG.paths["tables"]; FIG=PHASE/CFG.paths["figures"]


def _video_ci_fast(vdf, n_iter, rng_seed):
    """Vectorized cluster bootstrap over video-clusters.

    Statistically identical to study.cluster_bootstrap.cluster_bootstrap_metric
    (same rng.default_rng(seed), same with-replacement resampling of sorted
    clusters, same Mann-Whitney AUC) -- verified to reproduce lib CIs exactly.
    Inlined because the lib's per-cluster pd.concat loop is ~0.77s/iter at
    ~2k videos (~13 min per config); this is ~0.001s/iter. Does not alter the
    shared library, which Q1/Q2 use on much smaller method-level frames.
    """
    rng = np.random.default_rng(rng_seed)
    clusters = np.array(sorted(vdf["cluster"].unique()))
    g = vdf.set_index("cluster")
    sc = np.array([g.loc[c, "score"] for c in clusters])
    lb = np.array([g.loc[c, "label"] for c in clusters])
    n = len(clusters)
    vals = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        s = sc[idx]; l = lb[idx]
        f = s[l == 1]; r = s[l == 0]
        if f.size and r.size:
            d = f[:, None] - r[None, :]
            vals.append((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / (f.size * r.size))
    vals = np.array(vals)
    return {"ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5))}

def load_q(bb, sd, dom, methods=None):
    methods = methods if methods is not None else ALL+["real"]
    df=dedup_reals(load_frame_subset(str(ROOT/CFG.parquet["frame"]), train_mode="raw",
        infer_mode="length", backbones=[bb], seeds=[sd], domains=[dom], methods=methods))
    df["score"]=np.log(df["feature_norm"].to_numpy()+1e-12)
    df=df.rename(columns={"method_id":"method"})
    return df

def main():
    rows=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            ffpp=load_q(bb,sd,"ffpp"); cdf=load_q(bb,sd,"cdf")
            o=orient_sign(ffpp[ffpp.label==1].score.to_numpy(), ffpp[ffpp.label==0].score.to_numpy())
            for name,df in [("ffpp",ffpp),("cdf",cdf)]:
                df=df.copy(); df["score"]=o*df["score"]
                row=dict(backbone=bb,seed=sd,domain=name,orientation=o,
                    micro=micro_auc(df), macro=macro_auc(df),
                    video_weighted=video_weighted_macro(df),
                    video_mean=video_auc(df,"mean"), video_median=video_auc(df,"median"),
                    video_topk=video_auc(df,"topk"))
                em,elo,ehi=equalized_auc(df,repeats=CFG.sampling["equalized_repeats"],
                    seed=CFG.global_seed, frames_per_video=CFG.sampling["frames_per_video_balanced"])
                row.update(equalized_mean=em,equalized_lo=elo,equalized_hi=ehi)
                # true CI via cluster bootstrap on video-mean length AUC
                vdf=df.groupby("video_id").agg(score=("score","mean"),label=("label","first")).reset_index()
                vdf["cluster"]=vdf["video_id"]
                ci=_video_ci_fast(vdf, n_iter=min(1000,CFG.bootstrap["n_iter"]), rng_seed=CFG.global_seed)
                row.update(video_ci_low=ci["ci_low"], video_ci_high=ci["ci_high"])
                rows.append(row)
    q3=pd.DataFrame(rows)
    # per-method length AUC (oriented), per domain
    pm=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            ffpp=load_q(bb,sd,"ffpp")
            o=orient_sign(ffpp[ffpp.label==1].score.to_numpy(),ffpp[ffpp.label==0].score.to_numpy())
            for dom in CFG.domains:
                df=load_q(bb,sd,dom); df["score"]=o*df["score"]
                real=df[df.label==0]
                for m in ALL:
                    sub=df[df.method==m]
                    if len(sub) and len(real):
                        pm.append(dict(backbone=bb,seed=sd,domain=dom,method=m,
                            length_auc=auc_mannwhitney(sub.score.to_numpy(),real.score.to_numpy())))
    pd.DataFrame(pm).to_csv(TAB/"Q3_per_method_length.csv",index=False)
    q3.to_csv(TAB/"Q3_length_robustness.csv",index=False)
    _plots(q3)
    print("Q3 done.")

def _plots(q3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cdf=q3[q3.domain=="cdf"]
    fig,ax=plt.subplots(figsize=(8,5))
    x=np.arange(len(CFG.backbones)); w=0.25
    for i,col in enumerate(["micro","macro","video_mean"]):
        vals=cdf.groupby("backbone")[col].mean().reindex(CFG.backbones).values
        ax.bar(x+(i-1)*w, vals, w, label=col)
    ax.set_xticks(x); ax.set_xticklabels(CFG.backbones); ax.axhline(0.5,ls="--",color="k")
    ax.set_title("Q3 CDF length-only AUC (oriented)"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"Q3_micro_macro_video_auc.png",dpi=120); plt.close("all")

if __name__=="__main__": main()
