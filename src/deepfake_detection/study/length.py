# src/deepfake_detection/study/length.py
"""Length-only (q=log||z||) orientation + AUC views.

ORIENTATION LIMITATION: orientation uses the FF++ data available in the parquet,
which is the FF++ TEST split (train/val not stored). This is a 1-bit in-domain
sign choice; the macro/video/equalized robustness conclusions are unaffected.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from deepfake_detection.study.auc_pairs import auc_mannwhitney, sample_pairs_video_balanced, aggregate_video_scores

def orient_sign(q_fake_ffpp, q_real_ffpp):
    a=auc_mannwhitney(np.asarray(q_fake_ffpp), np.asarray(q_real_ffpp))
    return +1.0 if a>=0.5 else -1.0

def micro_auc(df):
    fake=df[df.label==1]; real=df[df.label==0]
    return auc_mannwhitney(fake.score.to_numpy(), real.score.to_numpy())

def macro_auc(df):
    aucs=[]; real=df[df.label==0]
    for m,sub in df[df.label==1].groupby("method"):
        aucs.append(auc_mannwhitney(sub.score.to_numpy(), real.score.to_numpy()))
    return float(np.mean(aucs)) if aucs else float("nan")

def video_weighted_macro(df):
    aucs=[]; ws=[]; real=df[df.label==0]
    for m,sub in df[df.label==1].groupby("method"):
        aucs.append(auc_mannwhitney(sub.score.to_numpy(), real.score.to_numpy()))
        ws.append(sub.video_id.nunique())
    ws=np.array(ws,float); aucs=np.array(aucs,float)
    return float(np.sum(ws*aucs)/np.sum(ws)) if ws.sum()>0 else float("nan")

def video_auc(df, how="mean", k=5):
    v=aggregate_video_scores(df, how=how, k=k)
    return auc_mannwhitney(v[v.label==1].video_score.to_numpy(), v[v.label==0].video_score.to_numpy())

def equalized_auc(df, repeats=100, seed=42, frames_per_video=8):
    vals=[]
    for i in range(repeats):
        ff,rr=sample_pairs_video_balanced(df, frames_per_video=frames_per_video, seed=seed+i)
        if len(ff) and len(rr):
            vals.append(auc_mannwhitney(ff,rr))
    vals=np.array(vals)
    return float(vals.mean()), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))
