# src/study/flip_dissect.py
"""Flip-pair dissection + partial-normalization gamma curve from PARQUET only.

s_raw = ||z||(w.u) + b ;  s_dir = (w.u) + b  =>  recover per-frame (w.u), b
without needing z.  s_gamma = ||z||^gamma (w.u) + b ; gamma=1 raw, gamma=0 dir.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-6

def recover_wdotu_b(s_raw, s_dir, norm):
    """Return (w_dot_u (N,), b (N,)). Per-frame recovery; b is constant in theory."""
    s_raw = np.asarray(s_raw, dtype=np.float64)
    s_dir = np.asarray(s_dir, dtype=np.float64)
    norm = np.asarray(norm, dtype=np.float64)
    denom = norm - 1.0
    safe = np.abs(denom) > EPS
    wu = np.where(safe, (s_raw - s_dir) / np.where(safe, denom, 1.0), 0.0)
    b = s_dir - wu
    return wu, b

def score_gamma(w_dot_u, b, norm, gamma):
    norm = np.asarray(norm, dtype=np.float64)
    return (np.power(norm, gamma) * np.asarray(w_dot_u) + np.asarray(b))

def optimal_gamma(w_dot_u, b, norm, fake_mask, gamma_grid, auc_fn):
    """Pick gamma maximizing AUC over real/fake frames."""
    best_g, best_a = None, -1.0
    fake = fake_mask.astype(bool)
    for g in gamma_grid:
        s = score_gamma(w_dot_u, b, norm, g)
        a = auc_fn(s[fake], s[~fake])
        if a > best_a:
            best_a, best_g = a, g
    return best_g, best_a

def rho_log_ratio(fake_norm, real_norm):
    """rho = log(||z_real|| / ||z_fake||) for matched pairs (broadcastable)."""
    return np.log(np.asarray(real_norm) + 1e-12) - np.log(np.asarray(fake_norm) + 1e-12)
