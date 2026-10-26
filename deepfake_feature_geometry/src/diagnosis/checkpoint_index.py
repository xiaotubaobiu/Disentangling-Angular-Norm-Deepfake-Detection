"""Enumerate the 30 retained checkpoints under
experiments/2026-05-05_paper_feature_geometry/results/checkpoints.

Layout (verified on disk):
    <ckpt_root>/<backbone>/<mode>/seed_<seed>__<ts>/output/best_model.pth
    <ckpt_root>/<backbone>/<mode>/seed_<seed>__<ts>/config/config.yaml

backbone ∈ {clip, dinov2, efficientnet}
mode     ∈ {raw, normtrain}
seed     ∈ {7, 42, 123, 999, 2048}

The per-checkpoint config/config.yaml is the canonical config source: it is
present for all 30 checkpoints (output/config.yaml exists only for CLIP).
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path


CKPT_GLOB_PATTERN = "*/*/seed_*__*/output/best_model.pth"
CKPT_PATH_RE = re.compile(
    r"/(?P<backbone>clip|dinov2|efficientnet)/"
    r"(?P<mode>raw|normtrain)/"
    r"seed_(?P<seed>\d+)__(?P<ts>\d{8}_\d{6})/"
)


@dataclass(frozen=True)
class CheckpointInfo:
    """A single retained checkpoint with parsed metadata."""

    ckpt_path: str
    config_path: str
    backbone: str        # lowercase: clip | dinov2 | efficientnet
    mode: str            # raw | normtrain
    seed: int
    timestamp: str       # YYYYMMDD_HHMMSS

    @property
    def checkpoint_id(self) -> str:
        """Stable identifier used as a key in Parquet rows."""
        return f"{self.backbone}_{self.mode}_s{self.seed}"

    def matches(self, backbones=None, modes=None, seeds=None) -> bool:
        if backbones and self.backbone not in backbones:
            return False
        if modes and self.mode not in modes:
            return False
        if seeds and self.seed not in seeds:
            return False
        return True


def discover_checkpoints(
    ckpt_root: str = "experiments/2026-05-05_paper_feature_geometry/results/checkpoints",
    backbones: list[str] | None = None,
    modes: list[str] | None = None,
    seeds: list[int] | None = None,
    limit: int | None = None,
) -> list[CheckpointInfo]:
    """Discover and filter checkpoints.

    Args:
        ckpt_root: root directory containing <backbone>/<mode>/seed_*/.
        backbones: optional filter (lowercase names). None = all.
        modes: optional filter ({raw, normtrain}). None = all.
        seeds: optional filter (ints). None = all.
        limit: if set, return only the first N matches (after filtering).

    Returns:
        Sorted list of CheckpointInfo. Sorted by (backbone, mode, seed) so the
        same N checkpoints are returned deterministically for smoke tests.
    """
    root = Path(ckpt_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    pattern = str(root / CKPT_GLOB_PATTERN)
    paths = sorted(glob.glob(pattern))

    infos: list[CheckpointInfo] = []
    for path in paths:
        m = CKPT_PATH_RE.search(path)
        if m is None:
            continue
        info = CheckpointInfo(
            ckpt_path=path,
            config_path=str(
                Path(path).parents[1] / "config" / "config.yaml"
            ),
            backbone=m.group("backbone"),
            mode=m.group("mode"),
            seed=int(m.group("seed")),
            timestamp=m.group("ts"),
        )
        if not info.matches(backbones=backbones, modes=modes, seeds=seeds):
            continue
        infos.append(info)

    infos.sort(key=lambda c: (c.backbone, c.mode, c.seed))
    if limit is not None:
        infos = infos[:limit]
    return infos
