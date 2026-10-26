"""Video-table construction: real de-duplication and raw/direction parity.

The FF++ val cache stores each real frame once per forgery method, with the method
name glued onto `video_id` (`815_pirender`). Without de-duplication the real class
is inflated 16x and every threshold moves. The target parquet has the analogous
replication, handled by `study.io.dedup_reals`.

Raw and direction must always describe the same videos: they are two columns of one
wide table, and `assert_modes_share_videos` is the tripwire.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from study_threshold.io import (
    aggregate_videos,
    assert_modes_share_videos,
    dedup_source_reals,
    load_source_frames,
    method_slice,
)

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
CFG = yaml.safe_load((PHASE / "configs/threshold_study.yaml").read_text())
VAL_DIR = ROOT / CFG["inputs"]["source_val_cache"]


def _synthetic_source(n_methods=4, n_frames=3):
    """One real video replicated per method (as the cache does) + one fake per method."""
    rows = []
    for m in range(n_methods):
        method = f"m{m}"
        for f in range(n_frames):
            rows.append(dict(image_path=f"/data/original/frames/900/{f}.png",
                             video_id=f"900_{method}", method_id="real", label=0,
                             s_raw=-2.0 + f, s_dir=-0.2 + 0.1 * f, feature_norm=10.0))
            rows.append(dict(image_path=f"/data/{method}/frames/900_1/{f}.png",
                             video_id=f"900_1_{method}", method_id=method, label=1,
                             s_raw=2.0 + f, s_dir=0.2 + 0.1 * f, feature_norm=12.0))
    return pd.DataFrame(rows)


def test_dedup_source_reals_collapses_per_method_copies():
    df = _synthetic_source(n_methods=4, n_frames=3)
    assert (df.label == 0).sum() == 12          # 4 methods x 3 frames of the SAME frames
    out = dedup_source_reals(df)
    real = out[out.label == 0]
    assert len(real) == 3                       # one row per distinct frame
    assert set(real.video_id) == {"900"}        # method suffix stripped
    assert set(real.method_id) == {"real"}
    assert (out.label == 1).sum() == 12         # fakes untouched


def test_video_aggregation_counts_each_real_video_once():
    videos = aggregate_videos(dedup_source_reals(_synthetic_source()))
    assert (videos.label == 0).sum() == 1
    assert (videos.label == 1).sum() == 4
    assert videos.video_key.is_unique


def test_video_key_includes_method_id():
    """Same video_id under two methods must stay two videos."""
    frames = pd.DataFrame([
        dict(image_path="/a/1.png", video_id="v1", method_id="m1", label=1, s_raw=1.0, s_dir=0.1),
        dict(image_path="/b/1.png", video_id="v1", method_id="m2", label=1, s_raw=2.0, s_dir=0.2),
    ])
    videos = aggregate_videos(frames)
    assert len(videos) == 2
    assert set(videos.video_key) == {"m1|v1", "m2|v1"}


@pytest.mark.skipif(not VAL_DIR.is_dir(), reason="FF++ val cache is a local artifact")
def test_real_video_count_on_the_actual_cache():
    frames = load_source_frames(VAL_DIR, "clip_raw_s42")
    real = frames[frames.label == 0]
    assert real.image_path.is_unique
    assert real.video_id.nunique() == 100          # FF++ val subsample: 100 real videos
    videos = aggregate_videos(frames)
    assert (videos.label == 0).sum() == 100
    assert (videos.label == 1).sum() == 1600       # 16 methods x 100 videos


@pytest.mark.skipif(not VAL_DIR.is_dir(), reason="FF++ val cache is a local artifact")
def test_raw_and_direction_share_the_same_videos():
    frames = load_source_frames(VAL_DIR, "clip_raw_s42")
    videos = aggregate_videos(frames)
    assert_modes_share_videos(videos)
    # identical support: both modes are defined on exactly the same keys, and the
    # per-video frame counts they average over are the same by construction
    assert videos[["s_raw", "s_dir"]].notna().all().all()
    assert videos.video_key.is_unique
    sub = method_slice(videos, "fsgan")
    assert set(sub.loc[sub.label == 1, "method_id"]) == {"fsgan"}
    assert (sub.label == 0).sum() == 100
    # a mode-specific NaN must be caught
    broken = videos.copy()
    broken.loc[broken.index[0], "s_dir"] = np.nan
    with pytest.raises(ValueError):
        assert_modes_share_videos(broken)
