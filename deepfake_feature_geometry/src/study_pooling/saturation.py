# src/study_pooling/saturation.py
from __future__ import annotations
import numpy as np

def frame_stats(s, p):
    s = np.asarray(s, float)
    return {"mean":float(s.mean()), "std":float(s.std()), "median":float(np.median(s)),
            "iqr":float(np.percentile(s,75)-np.percentile(s,25)),
            "q90":float(np.percentile(s,90)),"q95":float(np.percentile(s,95)),"q99":float(np.percentile(s,99)),
            "mean_abs":float(np.mean(np.abs(s))),"q95_abs":float(np.percentile(np.abs(s),95))}

def sat_ratios(p, thresholds):
    p = np.asarray(p, float)
    out = {}
    for t in thresholds:
        out[f"sat_{int(round(t*100)):03d}"] = float(np.mean((p<t)|(p>1-t)))
    return out

def sigmoid_grad_stats(p):
    p = np.asarray(p, float); g = p*(1-p)
    return {"mean":float(g.mean()), "median":float(np.median(g))}

def video_sat(df_frame, score_col, t=0.05):
    s = df_frame[score_col].to_numpy(float)
    p = 1.0/(1.0+np.exp(-s))
    return {"video_sat_ratio":float(np.mean((p<t)|(p>1-t))),
            "video_mean_abs_logit":float(np.mean(np.abs(s))),
            "video_logit_std":float(s.std()),
            "video_logit_iqr":float(np.percentile(s,75)-np.percentile(s,25))}
