# src/study/radial.py
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, permutation_test

def delta_d(q_fake, q_real):
    qf=np.asarray(q_fake,float); qr=np.asarray(q_real,float)
    num = qf.mean()-qr.mean()
    den = np.sqrt((qf.var()+qr.var())/2.0)
    return float(num/den) if den>0 else 0.0

def rtg_stg(d_ffpp, d_cdf):
    return {"RTG":abs(d_cdf-d_ffpp), "STG":d_cdf-d_ffpp,
            "R_retain":np.sign(d_ffpp)*d_cdf,
            "sign_flip":bool(np.sign(d_ffpp)!=np.sign(d_cdf))}

def length_auc(q_fake, q_real):
    from study.auc_pairs import auc_mannwhitney
    return auc_mannwhitney(q_fake, q_real)

def method_level_corr(df, x_col, y_col, n_perm=1000, rng_seed=42):
    mdf = df.groupby("method")[[x_col,y_col]].mean().reset_index()
    x=mdf[x_col].to_numpy(); y=mdf[y_col].to_numpy()
    pr = pearsonr(x,y); sr = spearmanr(x,y)
    pt = permutation_test((x,y), lambda a,b: pearsonr(a,b)[0],
                          permutation_type="pairings", n_resamples=n_perm, random_state=rng_seed)
    return {"pearson":float(pr[0]), "pearson_p":float(pr[1]),
            "spearman":float(sr[0]), "spearman_p":float(sr[1]),
            "perm_p":float(pt.pvalue), "n_methods":int(len(mdf))}

def lomo_corr(df, x_col, y_col, **kw):
    rs=[]
    for m in df["method"].unique():
        sub=df[df.method!=m]
        if len(sub["method"].unique())>=3:
            rs.append(method_level_corr(sub,x_col,y_col)["pearson"])
    return {"min":float(min(rs)),"max":float(max(rs)),"all":rs}
