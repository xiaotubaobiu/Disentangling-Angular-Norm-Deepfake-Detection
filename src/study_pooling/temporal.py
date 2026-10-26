# src/study_pooling/temporal.py
from __future__ import annotations
import re
import numpy as np

_FRAME_RE = re.compile(r"(\d+)\.png$")

def _frame_key(path):
    m = _FRAME_RE.search(path or "")
    return int(m.group(1)) if m else 0

def _frame_order(paths):
    keys = np.array([_frame_key(p) for p in paths])
    return np.argsort(keys, kind="stable")

def temporal_stats(df_frame, score_col):
    # Order frames by numeric frame key parsed from image_path (NOT lexicographic
    # image_path sort, which would order "100" before "20"). Index, then compute.
    order = _frame_order(df_frame["image_path"].to_numpy())
    s = df_frame.iloc[order][score_col].to_numpy(float)
    med = np.median(s); mad = np.median(np.abs(s-med))
    aMAD = mad*1.4826 if mad>0 else (s.std() or 1.0)
    q10,q90 = np.percentile(s,[10,90])
    diffs = np.abs(np.diff(s)) if s.size>1 else np.array([0.0])
    ac = float(np.corrcoef(s[:-1], s[1:])[0,1]) if s.size>=4 and np.std(s[:-1])>0 and np.std(s[1:])>0 else float("nan")
    return {"std":float(s.std()), "mad":float(mad),
            "iqr":float(np.percentile(s,75)-np.percentile(s,25)),
            "q90_q10":float(q90-q10), "p_pos":float(np.mean(s>0)),
            "adj_diff":float(np.mean(diffs)), "auto_corr":ac,
            "outlier_ratio":float(np.mean(np.abs(s-med)>3*aMAD))}
