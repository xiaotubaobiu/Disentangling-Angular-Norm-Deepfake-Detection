"""Bootstrap: fixed seed reproducibility, pairing, and equivalence with the reused
reference implementation `study.cluster_bootstrap.cluster_bootstrap_metric`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from study.cluster_bootstrap import cluster_bootstrap_metric
from study_threshold.bootstrap import CLUSTER_COL, bootstrap_metric, paired_bootstrap


def _videos(n=120, seed=0):
    rng = np.random.default_rng(seed)
    label = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    s_raw = np.where(label == 1, rng.normal(2, 1, n), rng.normal(-1, 1, n))
    return pd.DataFrame({
        "video_key": [f"v{i}" for i in range(n)],
        "label": label,
        "s_raw": s_raw,
        "s_dir": s_raw * 0.1 + rng.normal(0, 0.05, n),
    })


def _mean_fake(df):
    return float(df.loc[df.label == 1, "s_raw"].mean())


def test_fixed_seed_is_reproducible():
    df = _videos()
    a = bootstrap_metric(df, _mean_fake, n_iter=200, rng_seed=20260730)
    b = bootstrap_metric(df, _mean_fake, n_iter=200, rng_seed=20260730)
    assert a == b
    c = bootstrap_metric(df, _mean_fake, n_iter=200, rng_seed=1)
    assert (c["ci_low"], c["ci_high"]) != (a["ci_low"], a["ci_high"])


def test_fast_path_matches_the_reference_implementation():
    """One row per cluster: the fast path must draw the same clusters as the reference."""
    df = _videos(n=60, seed=3)
    fast = bootstrap_metric(df, _mean_fake, n_iter=100, rng_seed=20260730, fast=True)
    reference = cluster_bootstrap_metric(df, _mean_fake, CLUSTER_COL,
                                         n_iter=100, rng_seed=20260730)
    assert fast["point"] == reference["point"]
    assert fast["ci_low"] == reference["ci_low"]
    assert fast["ci_high"] == reference["ci_high"]
    assert fast["n_iter"] == reference["n_iter"]


def test_paired_bootstrap_uses_one_resample_set():
    df = _videos(n=80, seed=5)
    raw = lambda d: float(d.loc[d.label == 1, "s_raw"].mean())          # noqa: E731
    direction = lambda d: float(d.loc[d.label == 1, "s_dir"].mean())    # noqa: E731
    res = paired_bootstrap(df, raw, direction, n_iter=200, rng_seed=20260730)
    assert res["diff_point"] == direction(df) - raw(df)
    # a paired difference CI is tighter than naively subtracting the two marginals
    naive_width = ((res["direction_ci_high"] - res["direction_ci_low"])
                   + (res["raw_ci_high"] - res["raw_ci_low"]))
    assert (res["diff_ci_high"] - res["diff_ci_low"]) < naive_width
    assert res["rng_seed"] == 20260730 and res["n_iter"] <= 200


def test_cluster_column_is_required():
    df = _videos().drop(columns=["video_key"])
    try:
        bootstrap_metric(df, _mean_fake, n_iter=10)
    except ValueError as exc:
        assert CLUSTER_COL in str(exc)
    else:
        raise AssertionError("missing cluster column must raise")
