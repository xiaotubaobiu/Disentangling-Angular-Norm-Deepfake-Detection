from __future__ import annotations

from math import floor


def sample_uniform_frame_indices(num_frames: int, num_samples: int = 8) -> list[int]:
    if num_frames < 1:
        return []
    if num_samples <= 1:
        return [0]
    num_samples = min(num_samples, num_frames)
    return [floor(i * (num_frames - 1) / (num_samples - 1)) for i in range(num_samples)]
