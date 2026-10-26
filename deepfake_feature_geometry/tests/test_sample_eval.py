from __future__ import annotations

import math

import torch

from evaluation.sample_eval import (
    build_sample_rows,
    frame_metrics_from_rows,
    sample_rows_to_video_predictions,
    video_metrics_from_rows,
)


def _row(method_id: str, video_id: str, frame: int, label: int, probability: float) -> dict:
    return {
        "sample_id": f"{method_id}-{video_id}-{frame}",
        "image_path": f"/{method_id}/{video_id}/{frame}.png",
        "method_id": method_id,
        "video_id": video_id,
        "video_key": f"{method_id}::{video_id}",
        "label": label,
        "logit_real": 0.0,
        "logit_fake": 0.0,
        "prob_fake": probability,
        "feature_norm": 1.0,
    }


def test_build_sample_rows_preserves_semantic_video_key():
    batch = {
        "sample_id": ["a", "b"],
        "image_path": ["/a.png", "/b.png"],
        "method_id": ["simswap", "real"],
        "video_id": ["001", "001"],
        "video_key": ["simswap::001", "real::001"],
        "label": torch.tensor([1, 0]),
    }
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    probabilities = torch.softmax(logits, dim=1)[:, 1]

    rows = build_sample_rows(batch, logits, probabilities, torch.ones(2))

    assert [row["method_id"] for row in rows] == ["simswap", "real"]
    assert [row["video_key"] for row in rows] == ["simswap::001", "real::001"]


def test_same_video_id_across_methods_is_not_merged():
    rows = [
        _row("simswap", "001", 0, 1, 0.9),
        _row("simswap", "001", 1, 1, 0.7),
        _row("blendface", "001", 0, 1, 0.3),
        _row("blendface", "001", 1, 1, 0.1),
        _row("real", "001", 0, 0, 0.05),
    ]

    labels, scores = sample_rows_to_video_predictions(rows)

    assert labels == [1, 1, 0]
    assert scores == [0.8, 0.2, 0.05]


def test_frame_and_video_metrics_include_ap_and_brier():
    rows = [
        _row("fake_a", "v1", 0, 1, 0.9),
        _row("fake_a", "v1", 1, 1, 0.7),
        _row("real", "v2", 0, 0, 0.2),
        _row("real", "v2", 1, 0, 0.0),
    ]

    frame = frame_metrics_from_rows(rows)
    video = video_metrics_from_rows(rows)

    assert frame["auc"] == 1.0
    assert frame["ap"] == 1.0
    assert math.isclose(frame["brier"], (0.01 + 0.09 + 0.04 + 0.0) / 4)
    assert video["auc"] == 1.0
    assert video["ap"] == 1.0
    assert math.isclose(video["brier"], ((1.0 - 0.8) ** 2 + 0.1**2) / 2)
