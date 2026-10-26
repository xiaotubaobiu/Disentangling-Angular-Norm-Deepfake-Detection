# scripts/02_radial_stability.py
"""Q2 (spec §5): radial stability per backbone, method-level units, STG/R_retain.
dir_minus_raw_cdf taken from Q1 VIDEO view (where the phenomenon lives)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
ROOT = PHASE.parents[1]                              # repository root; sys.path.insert(0,str(ROOT/"src"))
from study.config import load_config
from study.io import load_frame_subset, dedup_reals
from study.radial import delta_d, rtg_stg, length_auc, method_level_corr, lomo_corr

CFG=load_config(PHASE/"configs"/"study_config.yaml")
ALL=sum(CFG.fake_methods.values(),[])
TAB=PHASE/CFG.paths["tables"]; FIG=PHASE/CFG.paths["figures"]

def q_of(df):
    return np.log(df["feature_norm"].to_numpy()+1e-12)

def main():
    fpath=str(ROOT/CFG.parquet["frame"])
    rows=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            for m in ALL:
                per={}
                for dom in CFG.domains:
                    df=dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode="length",
                        backbones=[bb], seeds=[sd], domains=[dom], methods=[m,"real"]))
                    fake=df[df.label==1]; real=df[df.label==0]
                    if fake.empty or real.empty: continue
                    qf=q_of(fake); qr=q_of(real)
                    per[dom]=dict(d=delta_d(qf,qr), la=length_auc(qf,qr))
                if "ffpp" not in per or "cdf" not in per: continue
                ff=per["ffpp"]; cd=per["cdf"]; meta=rtg_stg(ff["d"], cd["d"])
                rows.append(dict(backbone=bb,seed=sd,method=m,
                    delta_ffpp=ff["d"],delta_cdf=cd["d"],RTG=meta["RTG"],
                    STG=meta["STG"],R_retain=meta["R_retain"],sign_flip=meta["sign_flip"],
                    length_auc_ffpp=ff["la"],length_auc_cdf=cd["la"]))
    q2=pd.DataFrame(rows)
    # attach dir_minus_raw_cdf from Q1 VIDEO view
    try:
        q1=pd.read_csv(TAB/"Q1_rank_flip.csv")
        d=q1[q1.view=="video"].groupby(["backbone","method"])["delta_auc"].mean().reset_index()
        d=d.rename(columns={"delta_auc":"dir_minus_raw_cdf"})
        q2=q2.merge(d,on=["backbone","method"],how="left")
    except FileNotFoundError:
        print("[warn] Q1 table missing; dir_minus_raw_cdf NaN")
    q2.to_csv(TAB/"Q2_radial_stability.csv",index=False)
    _corr(q2); _plots(q2)
    print("Q2 done.")

def _corr(q2):
    lines=["# Q2 method-level correlations (RTG vs dir_minus_raw_cdf)\n"]
    for bb in CFG.backbones:
        sub=q2[(q2.backbone==bb)&(q2.dir_minus_raw_cdf.notna())]
        if sub.empty:
            lines.append(f"## {bb}: no data\n"); continue
        c=method_level_corr(sub,"RTG","dir_minus_raw_cdf",n_perm=1000,rng_seed=CFG.global_seed)
        lo=lomo_corr(sub,"RTG","dir_minus_raw_cdf")
        lines.append(f"## backbone={bb}\n- pearson={c['pearson']:.3f} (p={c['pearson_p']:.3g}), "
                     f"spearman={c['spearman']:.3f} (p={c['spearman_p']:.3g}), perm_p={c['perm_p']:.3g}, "
                     f"n_methods={c['n_methods']}\n- LOMO pearson range=[{lo['min']:.3f},{lo['max']:.3f}]\n")
    (PHASE/CFG.paths["logs"]).mkdir(parents=True,exist_ok=True)
    (PHASE/CFG.paths["logs"]/"Q2_correlations.md").write_text("\n".join(lines))
    print("\n".join(lines))

def _plots(q2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,6))
    for bb,c in zip(CFG.backbones,["C0","C1","C2"]):
        s=q2[(q2.backbone==bb)&(q2.dir_minus_raw_cdf.notna())]
        if not s.empty:
            ax.scatter(s.RTG, s.dir_minus_raw_cdf, label=bb, color=c, s=25)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("RTG (radial instability)"); ax.set_ylabel("dir - raw AUC (CDF)")
    ax.set_title("Q2 RTG vs normalization gain"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"Q2_RTG_vs_normalization_gain.png",dpi=120)
    fig,ax=plt.subplots(figsize=(6,6))
    for bb,c in zip(CFG.backbones,["C0","C1","C2"]):
        s=q2[q2.backbone==bb]
        ax.scatter(s.delta_ffpp, s.delta_cdf, label=bb, color=c, s=20)
    ax.plot([-3,3],[-3,3],"k--",lw=1); ax.set_xlabel("delta FF++"); ax.set_ylabel("delta CDF")
    ax.set_title("Q2 delta FF++ vs CDF"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"Q2_delta_FFPP_vs_CDF.png",dpi=120); plt.close("all")

if __name__=="__main__": main()
