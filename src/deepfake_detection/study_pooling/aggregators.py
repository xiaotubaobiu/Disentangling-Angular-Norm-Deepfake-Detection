# src/deepfake_detection/study_pooling/aggregators.py
"""Video-level aggregators over per-frame signed logits + Jensen gap.

s_frame: (T,) signed logits for one video. Returns a scalar video score.
mean_prob uses sigmoid(s); mean_logit uses raw s. Because sigmoid is strictly
monotone, AUC(mean_logit) == AUC(sigmoid(mean_logit)) exactly (rank identity)."""
from __future__ import annotations
import numpy as np

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x, dtype=np.float64)))

def mean_prob(s):     return float(np.mean(_sigmoid(s)))
def mean_logit(s):    return float(np.mean(np.asarray(s, dtype=np.float64)))
def sigmoid_mean_logit(s): return float(_sigmoid(np.mean(np.asarray(s, dtype=np.float64))))
def median_logit(s):  return float(np.median(np.asarray(s, dtype=np.float64)))
def clip_prob(s, c=6.0):
    s = np.asarray(s, dtype=np.float64)
    return float(np.mean(_sigmoid(np.clip(s, -c, c))))
def trimmed_logit(s, trim=0.10):
    s = np.asarray(s, dtype=np.float64)
    if s.size < 4: return float(np.mean(s))
    lo = int(np.floor(s.size*trim)); hi = int(np.ceil(s.size*(1-trim)))
    return float(np.mean(np.sort(s)[lo:hi]))
def topk_logit(s, frac=0.25):
    s = np.asarray(s, dtype=np.float64)
    k = max(1, int(np.ceil(s.size*frac)))
    return float(np.mean(np.sort(s)[-k:]))

NAMES = ["mean_prob","mean_logit","sigmoid_mean_logit","median_logit",
         "clip_prob","trimmed_logit","topk_logit"]
_FNS = {"mean_prob": mean_prob, "mean_logit": mean_logit,
        "sigmoid_mean_logit": sigmoid_mean_logit, "median_logit": median_logit,
        "clip_prob": clip_prob, "trimmed_logit": trimmed_logit, "topk_logit": topk_logit}

def aggregate(name, s_frame, **kw):
    fn = _FNS[name]
    if name == "clip_prob": return fn(s_frame, c=kw.get("c", 6.0))
    if name == "trimmed_logit": return fn(s_frame, trim=kw.get("trim", 0.10))
    if name == "topk_logit": return fn(s_frame, frac=kw.get("frac", 0.25))
    return fn(s_frame)

def jensen_gap(s) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float(np.mean(_sigmoid(s)) - _sigmoid(np.mean(s)))
