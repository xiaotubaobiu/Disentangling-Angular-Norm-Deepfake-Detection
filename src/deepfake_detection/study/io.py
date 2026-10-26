# src/deepfake_detection/study/io.py
"""DuckDB-backed parquet loading with DFD exclusion and real dedup."""
from __future__ import annotations
import pandas as pd

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

from deepfake_detection.study.canonical import add_canonical_keys

_COLS = [
    "checkpoint_id","backbone","seed","train_mode","infer_mode","domain",
    "method_id","label","video_id","image_path","feature_norm",
    "real_logit","fake_logit","signed_logit","prob_fake","prediction_correct",
]

def load_frame_subset(parquet_path, train_mode=None, infer_mode=None,
                      backbones=None, seeds=None, domains=None, methods=None):
    cols = ", ".join(_COLS)
    where = ["domain != 'dfd'"]
    if train_mode: where.append(f"train_mode = '{train_mode}'")
    if infer_mode: where.append(f"infer_mode = '{infer_mode}'")
    if backbones: where.append("backbone IN ('" + "','".join(backbones) + "')")
    if seeds: where.append("seed IN (" + ",".join(map(str, seeds)) + ")")
    if domains: where.append("domain IN ('" + "','".join(domains) + "')")
    if methods: where.append("method_id IN ('" + "','".join(methods) + "')")
    sql = f"SELECT {cols} FROM read_parquet('{parquet_path}') WHERE " + " AND ".join(where)
    if _HAS_DUCKDB:
        return duckdb.query(sql).df()
    df = pd.read_parquet(parquet_path, columns=_COLS)
    return df.query(" and ".join(where))

def dedup_reals(df):
    """Drop duplicated real frames (parquet replicates real rows per method).

    Dedup key includes `infer_mode` when present: Q1 loads multiple infer_modes
    at once (raw + direction) and needs to keep one real row PER infer_mode so
    the per-frame raw/direction merge stays intact, while still collapsing the
    cross-method real copies within each infer_mode. For single-infer_mode
    loaders (Q2/Q3) the extra key is a no-op.
    """
    if df.empty:
        return df
    df = add_canonical_keys(df)
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]
    key = ["domain", "source_video_id", "canonical_frame_id"]
    if "infer_mode" in real.columns:
        key.append("infer_mode")
    real = real.drop_duplicates(subset=key).copy()
    out = pd.concat([real, fake], ignore_index=True)
    return out
