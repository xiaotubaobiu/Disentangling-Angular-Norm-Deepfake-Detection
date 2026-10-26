"""Video-cluster bootstrap with resamples shared between raw and direction.

The reference implementation is `study.cluster_bootstrap.cluster_bootstrap_metric`
(reused, not reimplemented). It rebuilds each resample with one boolean mask per
cluster, which is O(n_clusters * n_rows) per iteration — fine for the frame-level
tables it was written for, too slow for 2000 iterations over ~6.6k video rows
where the metric itself has to re-scan an oracle threshold.

So this module adds a **fast path** for the case this study always hits: one row
per cluster (a video-level table). Then resampling clusters is exactly resampling
rows, and the same `np.random.Generator.choice(size, replace=True)` draw can be
applied to positional indices. `Generator.choice` with `replace=True` and no `p`
samples positions via `integers(0, len(a))`, independently of `a`'s dtype, so for a
given seed the fast path selects *the same* clusters as the reference — asserted in
`tests/test_bootstrap_equivalence.py`.

The property the study depends on is *paired* resampling: raw and direction live in
one wide video table, so a single resample scores both modes, and the paired
difference (direction − raw) is computed on identical videos.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from study.cluster_bootstrap import cluster_bootstrap_metric

CLUSTER_COL = "video_key"


def _fast_cluster_bootstrap(df: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                            cluster_col: str, n_iter: int, rng_seed: int) -> dict:
    """Equivalent to `cluster_bootstrap_metric` when each cluster is a single row."""
    order = np.argsort(df[cluster_col].to_numpy().astype(str), kind="stable")
    ordered = df.iloc[order]
    n = len(ordered)
    rng = np.random.default_rng(rng_seed)
    point = metric(df)
    vals = []
    for _ in range(n_iter):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        v = metric(ordered.iloc[idx])
        if v == v:  # drop NaN draws (e.g. a resample with one class only)
            vals.append(v)
    vals = np.asarray(vals, float)
    return {"point": float(point),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_iter": int(vals.size)}


def bootstrap_metric(videos: pd.DataFrame, metric: Callable[[pd.DataFrame], float],
                     n_iter: int = 2000, rng_seed: int = 20260730,
                     fast: bool | None = None) -> dict:
    """Percentile CI of `metric` under resampling of whole videos.

    `fast=None` (default) picks the fast path when every cluster is one row, which
    is the case for the video tables this study builds, and the reference
    implementation otherwise.
    """
    if CLUSTER_COL not in videos.columns:
        raise ValueError(f"video table must carry a {CLUSTER_COL!r} cluster column")
    if fast is None:
        fast = bool(videos[CLUSTER_COL].is_unique)
    if fast:
        return _fast_cluster_bootstrap(videos, metric, CLUSTER_COL, n_iter, rng_seed)
    return cluster_bootstrap_metric(videos, metric, CLUSTER_COL,
                                    n_iter=n_iter, rng_seed=rng_seed)


def paired_bootstrap(videos: pd.DataFrame,
                     metric_raw: Callable[[pd.DataFrame], float],
                     metric_direction: Callable[[pd.DataFrame], float],
                     n_iter: int = 2000, rng_seed: int = 20260730) -> dict:
    """Point + CI for raw, direction and their paired difference, from one resample set."""
    raw = bootstrap_metric(videos, metric_raw, n_iter=n_iter, rng_seed=rng_seed)
    direction = bootstrap_metric(videos, metric_direction, n_iter=n_iter, rng_seed=rng_seed)
    diff = bootstrap_metric(videos, lambda df: metric_direction(df) - metric_raw(df),
                            n_iter=n_iter, rng_seed=rng_seed)
    return {
        "raw_point": raw["point"], "raw_ci_low": raw["ci_low"], "raw_ci_high": raw["ci_high"],
        "direction_point": direction["point"], "direction_ci_low": direction["ci_low"],
        "direction_ci_high": direction["ci_high"],
        "diff_point": diff["point"], "diff_ci_low": diff["ci_low"],
        "diff_ci_high": diff["ci_high"],
        "n_iter": int(diff["n_iter"]),
        "rng_seed": int(rng_seed),
    }
