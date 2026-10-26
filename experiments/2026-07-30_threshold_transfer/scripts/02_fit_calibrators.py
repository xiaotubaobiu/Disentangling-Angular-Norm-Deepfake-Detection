#!/usr/bin/env python3
"""Stage 02 — fit temperature and video-level Platt scaling on FF++ validation.

Inputs : phase-4 / local FF++ val per-frame caches (temperature) and
         results/val_cache/source_video_scores.parquet from stage 01 (Platt).
Outputs: results/tables/02_calibrators.csv
GPU    : no.
Target-domain labels: **no — by construction**. This stage never reads
         `inputs.frame_scores`; the CDF parquet path is not even opened, which is
         what `tests/test_no_leakage.py` asserts.

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/02_fit_calibrators.py
    python .../02_fit_calibrators.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from study_threshold.calibration import fit_calibrators  # noqa: E402
from study_threshold.io import MODES, load_source_frames  # noqa: E402
from study_threshold.reporting import cell_id, phase_paths, write_table  # noqa: E402

sys.path.insert(0, str(PHASE / "scripts"))
from importlib import import_module  # noqa: E402

_stage01 = import_module("01_prepare_ffpp_validation")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = _stage01.load_cfg(Path(args.config) if args.config else None)
    paths = phase_paths(PHASE, cfg)
    backbones, seeds, methods = _stage01.resolve_dims(cfg, args.dry_run)

    source_videos = pd.read_parquet(paths["val_cache"] / _stage01.OUT_NAME)

    rows = []
    for backbone in backbones:
        for seed in seeds:
            ckpt = cell_id(backbone, seed)
            frames = load_source_frames(
                _stage01.resolve_cache_dir(cfg, paths, ckpt), ckpt, methods=methods)
            videos = source_videos[source_videos.checkpoint_id == ckpt]
            if videos.empty:
                raise SystemExit(f"stage 01 produced no source videos for {ckpt}")
            for mode in MODES:
                calib = fit_calibrators(frames, videos, mode)
                row = {"checkpoint_id": ckpt, "backbone": backbone, "seed": seed}
                row.update(calib.as_row())
                rows.append(row)
                print(f"[calib] {ckpt} {mode:9s} T={calib.temperature:7.3f} "
                      f"A={calib.platt_a:7.3f} B={calib.platt_b:+7.3f} "
                      f"| val frame ECE {calib.frame_ece_before:.3f}->{calib.frame_ece_after:.3f} "
                      f"| val video ECE {calib.video_ece_before:.3f}->"
                      f"{calib.video_ece_after_platt:.3f} (Platt)")

    write_table(pd.DataFrame(rows), paths["tables"] / "02_calibrators.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
