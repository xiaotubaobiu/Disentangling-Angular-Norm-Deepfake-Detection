# src/deepfake_detection/study/auc_pairs.py
"""Mann-Whitney AUC (ties=0.5), rank-flip H/B/CC/WW, identity check.

AUC = P(s_f > s_r) + 0.5 P(s_f == s_r).  On the SAME pair set with no ties,
AUC_dir - AUC_raw == H - B exactly (machine precision with counting).
"""
from __future__ import annotations
import numpy as np

def auc_mannwhitney(fake_scores, real_scores):
    f = np.asarray(fake_scores, dtype=np.float64)
    r = np.asarray(real_scores, dtype=np.float64)
    diff = f[:, None] - r[None, :]
    wins = np.sum(diff > 0)
    ties = np.sum(diff == 0)
    n = f.size * r.size
    return float((wins + 0.5 * ties) / n)

def flip_counts(raw_f, raw_r, dir_f, dir_r):
    rf = np.asarray(raw_f, dtype=np.float64)[:, None] - np.asarray(raw_r)[None, :]
    df = np.asarray(dir_f, dtype=np.float64)[:, None] - np.asarray(dir_r)[None, :]
    n = rf.size
    raw_ok = rf > 0
    dir_ok = df > 0
    cc = np.sum(dir_ok & raw_ok) / n
    ww = np.sum((~dir_ok) & (~raw_ok)) / n
    h = np.sum(dir_ok & (~raw_ok)) / n
    b = np.sum((~dir_ok) & raw_ok) / n
    return float(cc), float(ww), float(h), float(b)

def tie_rate(f, r):
    f = np.asarray(f, dtype=np.float64)[:, None]; r = np.asarray(r, dtype=np.float64)[None, :]
    return float(np.mean(f == r))

def identity_check(auc_raw, auc_dir, h, b):
    return abs((auc_dir - auc_raw) - (h - b))

import pandas as pd

def _split(df):
    return df[df["label"] == 1], df[df["label"] == 0]

def aggregate_video_scores(df: pd.DataFrame, how: str = "mean", k: int = 5) -> pd.DataFrame:
    g = df.groupby("video_id", sort=False)
    if how == "mean":
        v = g["score"].mean()
    elif how == "median":
        v = g["score"].median()
    elif how == "topk":
        v = g["score"].apply(lambda s: np.sort(s)[-k:].mean() if len(s) else np.nan)
    else:
        raise ValueError(how)
    out = v.to_frame("video_score")
    out["label"] = g["label"].first()
    out["source_video_id"] = g["source_video_id"].first()
    return out.reset_index()

def sample_pairs_native(df, n_pairs, seed):
    rng = np.random.default_rng(seed)
    fake, real = _split(df)
    if fake.empty or real.empty:
        return np.array([]), np.array([])
    n = int(max(1, np.sqrt(n_pairs)))
    fs = fake.sample(min(n, len(fake)), random_state=int(rng.integers(1<<31)))
    rs = real.sample(min(n, len(real)), random_state=int(rng.integers(1<<31)))
    return fs["score"].to_numpy(), rs["score"].to_numpy()

def sample_pairs_video_balanced(df, frames_per_video, seed, n_videos_cap=None):
    rng = np.random.default_rng(seed)
    fake, real = _split(df)
    def _per(group):
        out = []
        for vid, sub in group.groupby("video_id", sort=False):
            s = sub.sample(min(frames_per_video, len(sub)),
                           random_state=int(rng.integers(1<<31)))
            out.append(s["score"].to_numpy())
        return np.concatenate(out) if out else np.array([])
    fv = fake.groupby("video_id", sort=False).size()
    rv = real.groupby("video_id", sort=False).size()
    m = min(len(fv), len(rv))
    if m == 0:
        return np.array([]), np.array([])
    fake_vids = np.array(list(fv.index))[:m] if n_videos_cap is None else \
        rng.choice(list(fv.index), size=min(n_videos_cap, len(fv)), replace=False)
    real_vids = np.array(list(rv.index))[:m] if n_videos_cap is None else \
        rng.choice(list(rv.index), size=min(n_videos_cap, len(rv)), replace=False)
    fake_s = fake[fake.video_id.isin(fake_vids)]
    real_s = real[real.video_id.isin(real_vids)]
    return _per(fake_s), _per(real_s)
