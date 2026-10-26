"""Unit tests for the strict video aggregation protocol.

Locks down spec section II:

    P_video(V) = mean_t( softmax(frame_logits_t)[fake] )

We verify:
  - video_prob_fake equals mean of per-frame softmax fake probabilities.
  - video_prob_fake does NOT equal softmax(mean(per-frame logits))[fake]
    (i.e. the forbidden alternative is structurally not what we compute).
  - aggregation handles multiple videos, mixed labels, and single-frame edge.
  - method_id from the method_map is joined correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diagnosis.aggregation import (
    aggregate_frames_to_videos,
    rows_to_frame_dataframe,
)
from diagnosis.feature_extractor import FrameFeatures
from diagnosis.infer_modes import ModeScores


def _softmax_fake(real_logit: float, fake_logit: float) -> float:
    """Stable softmax([real, fake])[1]."""
    m = max(real_logit, fake_logit)
    e_real = np.exp(real_logit - m)
    e_fake = np.exp(fake_logit - m)
    return float(e_fake / (e_real + e_fake))


def _make_frame_df() -> pd.DataFrame:
    """Two videos: v1 (4 frames, fake) and v2 (3 frames, real).

    Frame logits are chosen so per-frame prob_fake varies, which makes the
    distinction between mean(prob) and softmax(mean(logit)) obvious.
    """
    rows = [
        # v1 (fake=1), 4 frames
        ("v1", 1, "raw", 0.0, 2.0),
        ("v1", 1, "raw", 0.0, 1.0),
        ("v1", 1, "raw", 0.0, 0.5),
        ("v1", 1, "raw", 0.0, 3.0),
        # v2 (real=0), 3 frames
        ("v2", 0, "raw", 2.0, 0.0),
        ("v2", 0, "raw", 1.0, 0.0),
        ("v2", 0, "raw", 0.0, 1.0),
    ]
    records = []
    for vid, label, mode, rl, fl in rows:
        prob = _softmax_fake(rl, fl)
        records.append({
            "checkpoint_id": "test_ckpt",
            "backbone": "test",
            "seed": 42,
            "train_mode": "raw",
            "infer_mode": mode,
            "domain": "ffpp",
            "method_id": "simswap" if label == 1 else "real",
            "label": label,
            "video_id": vid,
            "image_path": f"/tmp/{vid}_{rl}_{fl}.png",
            "feature_norm": float(abs(rl) + abs(fl) + 0.1),
            "real_logit": float(rl),
            "fake_logit": float(fl),
            "signed_logit": float(fl - rl),
            "prob_fake": prob,
            "prediction_correct": int((prob >= 0.5) == (label == 1)),
        })
    return pd.DataFrame(records)


def test_video_prob_equals_mean_of_frame_prob_fake():
    """Core assertion: P_video = mean_t(softmax(frame_logits)[fake])."""
    df = _make_frame_df()
    video_df = aggregate_frames_to_videos(df)

    v1_frames = df[df["video_id"] == "v1"]
    v2_frames = df[df["video_id"] == "v2"]
    expected_v1 = float(v1_frames["prob_fake"].mean())
    expected_v2 = float(v2_frames["prob_fake"].mean())

    got_v1 = float(video_df[video_df["video_id"] == "v1"]["video_prob_fake"].iloc[0])
    got_v2 = float(video_df[video_df["video_id"] == "v2"]["video_prob_fake"].iloc[0])

    assert np.isclose(got_v1, expected_v1, atol=1e-7), (
        f"v1: got {got_v1}, expected mean(prob_fake)={expected_v1}"
    )
    assert np.isclose(got_v2, expected_v2, atol=1e-7), (
        f"v2: got {got_v2}, expected mean(prob_fake)={expected_v2}"
    )


def test_video_prob_does_not_equal_softmax_of_mean_logits():
    """Verify the forbidden alternative (softmax after logit-mean) gives a
    different answer — confirming our code does not implement that path."""
    df = _make_frame_df()
    video_df = aggregate_frames_to_videos(df)

    v1_frames = df[df["video_id"] == "v1"]
    mean_real = float(v1_frames["real_logit"].mean())
    mean_fake = float(v1_frames["fake_logit"].mean())
    forbidden = _softmax_fake(mean_real, mean_fake)

    got_v1 = float(video_df[video_df["video_id"] == "v1"]["video_prob_fake"].iloc[0])
    assert not np.isclose(got_v1, forbidden, atol=1e-3), (
        f"v1: got {got_v1}, but forbidden softmax(mean(logits))={forbidden} "
        "should NOT match — aggregation may have been implemented wrong."
    )


def test_method_id_preserved_through_aggregation():
    """method_id grouping: v1=simswap, v2=real."""
    df = _make_frame_df()
    video_df = aggregate_frames_to_videos(df)
    by_vid = video_df.set_index("video_id")
    assert by_vid.loc["v1", "method_id"] == "simswap"
    assert by_vid.loc["v2", "method_id"] == "real"


def test_single_frame_video_std_is_zero():
    """A video with only one sampled frame should have std_feature_norm == 0
    (filled from NaN), not NaN."""
    df = _make_frame_df()
    # Add a one-frame video.
    df = pd.concat([df, pd.DataFrame([{
        "checkpoint_id": "test_ckpt", "backbone": "test", "seed": 42,
        "train_mode": "raw", "infer_mode": "raw", "domain": "ffpp",
        "method_id": "real", "label": 0, "video_id": "v3",
        "image_path": "/tmp/v3.png", "feature_norm": 1.0,
        "real_logit": 1.0, "fake_logit": 0.0, "signed_logit": -1.0,
        "prob_fake": _softmax_fake(1.0, 0.0),
        "prediction_correct": 1,
    }])], ignore_index=True)
    video_df = aggregate_frames_to_videos(df)
    v3 = video_df[video_df["video_id"] == "v3"].iloc[0]
    assert v3["std_feature_norm"] == 0.0
    assert v3["std_signed_logit"] == 0.0
    assert v3["num_frames"] == 1


def test_rows_to_frame_dataframe_method_join():
    """End-to-end: rows_to_frame_dataframe should carry method_id from
    FrameFeatures.method_ids and produce one row per (frame, mode)."""
    features = FrameFeatures(
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        feature_norms=np.array([2.236, 5.0], dtype=np.float32),
        video_ids=["vidA", "vidB"],
        image_paths=["/tmp/a.png", "/tmp/b.png"],
        labels=np.array([1, 0], dtype=np.int64),
        method_ids=["simswap", "real"],
    )
    scores = ModeScores(
        mode="raw",
        real_logit=np.array([0.0, 1.0], dtype=np.float32),
        fake_logit=np.array([2.0, 0.0], dtype=np.float32),
        signed_logit=np.array([2.0, -1.0], dtype=np.float32),
        prob_fake=np.array([_softmax_fake(0.0, 2.0), _softmax_fake(1.0, 0.0)], dtype=np.float32),
    )
    df = rows_to_frame_dataframe(
        frame_features=features,
        mode_scores_per_mode={"raw": scores},
        checkpoint_id="effnet_raw_s42",
        backbone="efficientnet",
        seed=42,
        train_mode="raw",
        domain="ffpp",
    )
    assert len(df) == 2
    assert set(df["method_id"]) == {"simswap", "real"}
    assert (df["prediction_correct"] == 1).all()


def test_cross_method_same_pair_id_not_merged():
    """Critical regression test: two methods' fakes with the SAME pair_id must
    NOT be averaged together at the video level. The video group key includes
    method_id, so the two (method, pair_id) combinations stay separate.
    """
    rows = []
    # simswap fake video '000_003', 2 frames
    for rl, fl in [(0.0, 2.0), (0.0, 1.5)]:
        rows.append({
            "checkpoint_id": "c", "backbone": "b", "seed": 1,
            "train_mode": "raw", "infer_mode": "raw", "domain": "ffpp",
            "method_id": "simswap", "label": 1, "video_id": "000_003",
            "image_path": f"/tmp/simswap_{rl}_{fl}.png",
            "feature_norm": 1.0, "real_logit": rl, "fake_logit": fl,
            "signed_logit": fl - rl, "prob_fake": _softmax_fake(rl, fl),
            "prediction_correct": 1,
        })
    # blendface fake video with the SAME pair_id '000_003', 2 frames
    for rl, fl in [(0.0, 0.2), (0.0, 0.4)]:
        rows.append({
            "checkpoint_id": "c", "backbone": "b", "seed": 1,
            "train_mode": "raw", "infer_mode": "raw", "domain": "ffpp",
            "method_id": "blendface", "label": 1, "video_id": "000_003",
            "image_path": f"/tmp/blendface_{rl}_{fl}.png",
            "feature_norm": 1.0, "real_logit": rl, "fake_logit": fl,
            "signed_logit": fl - rl, "prob_fake": _softmax_fake(rl, fl),
            "prediction_correct": 1,
        })
    df = pd.DataFrame(rows)
    video_df = aggregate_frames_to_videos(df)

    # Expect exactly 2 video rows for pair_id '000_003': one per method.
    pair_rows = video_df[video_df["video_id"] == "000_003"]
    assert len(pair_rows) == 2, (
        f"Expected 2 rows for same pair_id across methods, got {len(pair_rows)} — "
        "video aggregation may be merging cross-method frames."
    )
    methods = set(pair_rows["method_id"])
    assert methods == {"simswap", "blendface"}, methods

    # And each video_prob_fake must equal the mean of that method's frames only.
    simswap_video = pair_rows[pair_rows["method_id"] == "simswap"].iloc[0]
    expected_simswap = (
        _softmax_fake(0.0, 2.0) + _softmax_fake(0.0, 1.5)
    ) / 2
    assert np.isclose(
        float(simswap_video["video_prob_fake"]), expected_simswap, atol=1e-6
    )
