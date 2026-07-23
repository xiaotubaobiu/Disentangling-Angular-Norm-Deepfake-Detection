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
