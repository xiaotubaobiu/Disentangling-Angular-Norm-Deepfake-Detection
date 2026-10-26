"""Loading and video aggregation for the threshold-transfer study.

Two inputs, both read-only and owned by earlier phases:

* source domain (FF++ **validation**): per-frame ``s_raw`` / ``s_dir`` caches
  produced by phase 2026-07-27 (`results/val_cache/<checkpoint_id>__val.parquet`).
  FF++ val is *not* in the phase-2 frame parquet (its `ffpp` rows are the test
  split), which is why the val caches exist at all.
* target domain (CDF): the phase-2 frame parquet
  (`per_frame_scores.parquet`), filtered to `train_mode=raw` and
  `infer_mode in {raw, direction}`.

Both are reshaped to the same *wide* frame table

    image_path | video_id | method_id | label | s_raw | s_dir

so raw and direction are guaranteed to describe the identical frames (they come
from one row each, merged on `image_path`), and therefore the identical videos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from study.io import dedup_reals
from study_pooling.aggregators import aggregate

WIDE_COLS = ["image_path", "video_id", "method_id", "label", "s_raw", "s_dir"]
MODES = ("raw", "direction")
_SCORE_COL = {"raw": "s_raw", "direction": "s_dir"}


def score_column(mode: str) -> str:
    if mode not in _SCORE_COL:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(_SCORE_COL)}")
    return _SCORE_COL[mode]


# --------------------------------------------------------------------------- #
# source domain: FF++ validation
# --------------------------------------------------------------------------- #

def val_cache_path(val_cache_dir, checkpoint_id: str) -> Path:
    return Path(val_cache_dir) / f"{checkpoint_id}__val.parquet"


def dedup_source_reals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the FF++ val cache's per-method copies of every real frame.

    The cache was written by the pooling phase's val loader, which pairs each
    forgery method with the pristine video it was built from. Consequently one
    real frame appears once per method (16x) and its `video_id` carries the
    method suffix (`815_pirender`), so a naive video aggregation would count the
    same real video 16 times and inflate the real class by 16x.

    Fix: real `video_id` is taken from the frame path (`.../frames/<vid>/<f>.png`
    -> `<vid>`) and duplicate real `image_path`s are dropped. Fake rows are
    untouched (their `video_id` is already method-specific).
    """
    real = df[df["label"] == 0].copy()
    fake = df[df["label"] == 1].copy()
    if not real.empty:
        real["video_id"] = [Path(p).parent.name for p in real["image_path"]]
        real["method_id"] = "real"
        real = real.drop_duplicates(subset=["image_path"])
    return pd.concat([real, fake], ignore_index=True)


def load_source_frames(val_cache_dir, checkpoint_id: str,
                       methods: list[str] | None = None) -> pd.DataFrame:
    """Per-frame FF++ validation scores for one checkpoint (wide, read-only)."""
    path = val_cache_path(val_cache_dir, checkpoint_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing FF++ validation cache for {checkpoint_id}: {path}. "
            "Run 01_prepare_ffpp_validation.py --extract-missing (needs a GPU)."
        )
    df = pd.read_parquet(path)
    missing = [c for c in WIDE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    df = dedup_source_reals(df[WIDE_COLS].copy())
    if methods:
        keep = set(methods) | {"real"}
        df = df[df["method_id"].isin(keep)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# target domain: CDF (from the phase-2 frame parquet)
# --------------------------------------------------------------------------- #

def load_target_frames(frame_parquet, backbone: str, seed: int, domain: str = "cdf",
                       methods: list[str] | None = None,
                       exclude_domains: tuple[str, ...] = ("dfd",)) -> pd.DataFrame:
    """Per-frame target-domain scores for one checkpoint, reshaped wide.

    `raw` and `direction` rows of the same frame share `image_path`, so the inner
    merge below both pairs them and guarantees an identical frame (hence video)
    set for the two modes.
    """
    import duckdb

    if domain in exclude_domains:
        raise ValueError(f"domain {domain!r} is excluded by configuration")
    sql = f"""
        SELECT infer_mode, image_path, video_id, method_id, label, signed_logit
        FROM read_parquet('{Path(frame_parquet)}')
        WHERE domain = '{domain}'
          AND train_mode = 'raw'
          AND infer_mode IN ('raw', 'direction')
          AND backbone = '{backbone}'
          AND seed = {int(seed)}
    """
    if methods:
        wanted = "','".join(list(methods) + ["real"])
        sql += f"      AND method_id IN ('{wanted}')\n"
    long = duckdb.query(sql).df()
    if long.empty:
        raise ValueError(f"no {domain} rows for backbone={backbone} seed={seed}")

    long["domain"] = domain
    long = dedup_reals(long)  # parquet replicates real frames once per method

    raw = (long[long.infer_mode == "raw"]
           [["image_path", "video_id", "method_id", "label", "signed_logit"]]
           .rename(columns={"signed_logit": "s_raw"}))
    direction = (long[long.infer_mode == "direction"]
                 [["image_path", "signed_logit"]]
                 .rename(columns={"signed_logit": "s_dir"}))
    wide = raw.merge(direction, on="image_path", how="inner")
    return wide[WIDE_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# video aggregation
# --------------------------------------------------------------------------- #

def aggregate_videos(frames: pd.DataFrame, aggregator: str = "mean_logit") -> pd.DataFrame:
    """Frames -> videos, keyed by ``(method_id, video_id)``.

    The key includes `method_id` on purpose: in FF++/CDF the same `video_id`
    (pair id) is reused by several forgery methods, and grouping by `video_id`
    alone silently averages different methods' fakes together (the bug behind the
    2026-05-05 `direction > raw` claim).

    Returns one row per video with both modes scored the same way:
        method_id | video_id | label | n_frames | s_raw | s_dir | video_prob_raw | video_prob_dir
    where `s_*` use `aggregator` (default `mean_logit`, the study's primary
    score) and `video_prob_*` are the deliberately mis-specified
    `mean_t(sigmoid(s))` baseline.
    """
    rows = []
    for (method_id, video_id), g in frames.groupby(["method_id", "video_id"], sort=False):
        s_raw = g["s_raw"].to_numpy(float)
        s_dir = g["s_dir"].to_numpy(float)
        rows.append({
            "method_id": method_id,
            "video_id": video_id,
            "label": int(g["label"].iloc[0]),
            "n_frames": int(len(g)),
            "s_raw": aggregate(aggregator, s_raw),
            "s_dir": aggregate(aggregator, s_dir),
            "video_prob_raw": aggregate("mean_prob", s_raw),
            "video_prob_dir": aggregate("mean_prob", s_dir),
        })
    out = pd.DataFrame(rows)
    out["video_key"] = out["method_id"].astype(str) + "|" + out["video_id"].astype(str)
    return out


def assert_modes_share_videos(videos: pd.DataFrame) -> None:
    """Both modes must be defined on exactly the same rows (no NaN, no partial merge)."""
    for col in ("s_raw", "s_dir"):
        if col not in videos:
            raise ValueError(f"video table lacks {col}")
        if not np.isfinite(videos[col].to_numpy(float)).all():
            raise ValueError(f"non-finite scores in {col}: raw/direction video sets diverge")
    if videos["video_key"].duplicated().any():
        raise ValueError("duplicated (method_id, video_id) keys in the video table")


def method_slice(videos: pd.DataFrame, method_id: str) -> pd.DataFrame:
    """That method's fake videos plus all real videos (native counts)."""
    return videos[(videos["method_id"] == method_id) | (videos["label"] == 0)].copy()
