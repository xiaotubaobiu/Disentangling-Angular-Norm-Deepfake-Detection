# src/deepfake_detection/study_pooling/video_eval.py
"""Aggregate frames->video, then AUC + rank-flip + identity for raw vs direction."""
from __future__ import annotations
import numpy as np, pandas as pd
from deepfake_detection.study.auc_pairs import auc_mannwhitney, flip_counts, identity_check, tie_rate
from deepfake_detection.study_pooling.aggregators import aggregate

def video_scores(df_frame, score_col, agg, **kw):
    out = []
    for vid, g in df_frame.groupby("video_id", sort=False):
        out.append({"video_id": vid, "label": int(g["label"].iloc[0]),
                    "score": aggregate(agg, g[score_col].to_numpy(), **kw)})
    return pd.DataFrame(out)

def eval_aggregator(df_frame, agg, **kw):
    vr_raw = video_scores(df_frame, "s_raw", agg, **kw)
    vr_dir = video_scores(df_frame, "s_dir", agg, **kw)
    f = vr_raw[vr_raw.label==1]; r = vr_raw[vr_raw.label==0]
    fd = vr_dir[vr_dir.label==1]; rd = vr_dir[vr_dir.label==0]
    fr = f["score"].to_numpy(); rr = r["score"].to_numpy()
    fdr = fd["score"].to_numpy(); rdr = rd["score"].to_numpy()
    auc_raw = auc_mannwhitney(fr, rr); auc_dir = auc_mannwhitney(fdr, rdr)
    cc,ww,h,b = flip_counts(fr, rr, fdr, rdr)
    return {"auc_raw":auc_raw, "auc_dir":auc_dir, "delta_auc":auc_dir-auc_raw,
            "CC":cc,"WW":ww,"H":h,"B":b,"H_minus_B":h-b,
            "identity_error":identity_check(auc_raw,auc_dir,h,b),
            "tie_rate":tie_rate(fr,rr), "n_fake":int(len(f)), "n_real":int(len(r))}
