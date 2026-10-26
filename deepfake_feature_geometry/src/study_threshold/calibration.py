"""Calibrators fitted on the source domain only.

Two calibrators, both wrapping `study_pooling.temperature` (no re-implementation):

1. **Temperature scaling** on per-frame signed logits: ``p = sigmoid(s / T)``.
   Fitted with weights that make every FF++ val video contribute equally *and*
   the two classes contribute equally, so long videos and the fake-heavy method
   mix do not dominate the NLL.
2. **Video-level Platt scaling** on the aggregated video score:
   ``p = sigmoid(A * s + B)`` with ``A > 0`` enforced (monotone, so it cannot
   flip the ranking and cannot change AUC).

Every function here takes source-domain data. The only way calibrated
probabilities reach the target domain is `apply_temperature` / `apply_platt` with
parameters produced here — the target domain never enters a fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from study_pooling.temperature import (
    apply_T,
    brier,
    ece,
    fit_platt,
    fit_temperature,
    nll,
)

SOURCE_ONLY_NOTE = "fitted on FF++ validation only"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def frame_weights(frames: pd.DataFrame) -> np.ndarray:
    """Per-frame weights: every video equal, then the two classes equal."""
    counts = frames.groupby("video_id")["image_path"].transform("count").to_numpy(float)
    w_video = 1.0 / np.maximum(counts, 1.0)
    label = frames["label"].to_numpy(int)
    n_pos, n_neg = int((label == 1).sum()), int((label == 0).sum())
    w_class = np.where(label == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    w = w_video * w_class
    total = w.sum()
    return w / total if total > 0 else w


def video_weights(videos: pd.DataFrame) -> np.ndarray:
    """Per-video weights making the two classes equal."""
    label = videos["label"].to_numpy(int)
    n_pos, n_neg = int((label == 1).sum()), int((label == 0).sum())
    w = np.where(label == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    total = w.sum()
    return w / total if total > 0 else w


@dataclass(frozen=True)
class Calibrator:
    """Frozen calibration parameters for one (checkpoint, mode)."""

    mode: str
    temperature: float
    platt_a: float
    platt_b: float
    # source-domain diagnostics (frame level for T, video level for Platt)
    frame_nll_before: float
    frame_nll_after: float
    frame_ece_before: float
    frame_ece_after: float
    frame_brier_before: float
    frame_brier_after: float
    video_nll_before: float
    video_nll_after_temperature: float
    video_nll_after_platt: float
    video_ece_before: float
    video_ece_after_temperature: float
    video_ece_after_platt: float
    n_val_frames: int
    n_val_videos: int
    fitted_on: str = SOURCE_ONLY_NOTE

    def as_row(self) -> dict:
        return asdict(self)


def apply_temperature(s, temperature: float) -> np.ndarray:
    return apply_T(s, temperature)


def apply_platt(s, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * np.asarray(s, float) + b)


def platt_decision_logit(a: float, b: float) -> float:
    """Score at which the Platt probability crosses 0.5 (``A*s + B = 0``)."""
    return float(-b / a) if a != 0 else float("nan")


def fit_calibrators(source_frames: pd.DataFrame, source_videos: pd.DataFrame,
                    mode: str) -> Calibrator:
    """Fit temperature (frames) and video-level Platt (videos) on the source domain."""
    from study_threshold.io import score_column

    col = score_column(mode)
    s_frame = source_frames[col].to_numpy(float)
    y_frame = source_frames["label"].to_numpy(int)
    w_frame = frame_weights(source_frames)

    temperature = fit_temperature(s_frame, y_frame, sample_weight=w_frame)

    s_video = source_videos[col].to_numpy(float)
    y_video = source_videos["label"].to_numpy(int)
    w_video = video_weights(source_videos)
    platt_a, platt_b = fit_platt(s_video, y_video, sample_weight=w_video)
    if platt_a <= 0:
        raise ValueError("Platt slope must be positive (monotone calibration)")

    p_frame_before = np.clip(_sigmoid(s_frame), 1e-12, 1 - 1e-12)
    p_frame_after = apply_temperature(s_frame, temperature)
    p_video_before = np.clip(_sigmoid(s_video), 1e-12, 1 - 1e-12)
    p_video_temp = apply_temperature(s_video, temperature)
    p_video_platt = apply_platt(s_video, platt_a, platt_b)

    def _nll_p(p, y, w):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.average(loss, weights=w))

    return Calibrator(
        mode=mode,
        temperature=float(temperature),
        platt_a=float(platt_a),
        platt_b=float(platt_b),
        frame_nll_before=nll(s_frame, y_frame, 1.0, w_frame),
        frame_nll_after=nll(s_frame, y_frame, temperature, w_frame),
        frame_ece_before=ece(p_frame_before, y_frame),
        frame_ece_after=ece(p_frame_after, y_frame),
        frame_brier_before=brier(p_frame_before, y_frame),
        frame_brier_after=brier(p_frame_after, y_frame),
        video_nll_before=_nll_p(p_video_before, y_video, w_video),
        video_nll_after_temperature=_nll_p(p_video_temp, y_video, w_video),
        video_nll_after_platt=_nll_p(p_video_platt, y_video, w_video),
        video_ece_before=ece(p_video_before, y_video),
        video_ece_after_temperature=ece(p_video_temp, y_video),
        video_ece_after_platt=ece(p_video_platt, y_video),
        n_val_frames=int(len(source_frames)),
        n_val_videos=int(len(source_videos)),
    )


def probability_metrics(s, y, calibrator: Calibrator) -> dict:
    """ECE / NLL / Brier of the three probability views on any domain."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    views = {
        "uncalibrated": np.clip(_sigmoid(s), 1e-12, 1 - 1e-12),
        "temperature": apply_temperature(s, calibrator.temperature),
        "platt": apply_platt(s, calibrator.platt_a, calibrator.platt_b),
    }
    out = {}
    for name, p in views.items():
        p = np.clip(p, 1e-12, 1 - 1e-12)
        out[f"ece_{name}"] = ece(p, y)
        out[f"nll_{name}"] = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
        out[f"brier_{name}"] = brier(p, y)
    return out
