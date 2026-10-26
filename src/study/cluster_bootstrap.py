# src/study/cluster_bootstrap.py
from __future__ import annotations
import numpy as np
import pandas as pd

def cluster_bootstrap_metric(df, metric, cluster_col, n_iter=10000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    clusters = np.array(sorted(df[cluster_col].unique()))
    point = metric(df)
    vals = []
    for _ in range(n_iter):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [df[df[cluster_col] == c] for c in sampled]
        sub = pd.concat(parts)
        v = metric(sub)
        if v == v:  # not nan
            vals.append(v)
    vals = np.array(vals)
    return {"point": float(point),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_iter": int(len(vals))}
