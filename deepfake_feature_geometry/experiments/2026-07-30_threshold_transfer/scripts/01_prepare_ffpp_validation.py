#!/usr/bin/env python3
"""Stage 01 — build the source-domain (FF++ validation) video table.

Inputs : phase-4 per-frame FF++ val caches (`inputs.source_val_cache`), or this
         phase's own `results/val_cache/` if a checkpoint was re-inferred here.
Outputs: results/val_cache/source_video_scores.parquet   (video level, both modes)
         results/tables/01_source_videos.csv              (per-cell summary)
GPU    : only with `--extract-missing`, which re-infers FF++ val for checkpoints
         that have no cache. Default path is CPU-only (reads existing caches).
Target-domain labels: no. This stage never opens the CDF parquet.

FF++ val is not part of the phase-2 frame parquet (its `ffpp` rows are the *test*
split), so the source domain has to come from the val caches. They are read
in place — never copied — and any newly extracted cache is written here, leaving
historical phases untouched.

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/01_prepare_ffpp_validation.py
    python .../01_prepare_ffpp_validation.py --dry-run
    python .../01_prepare_ffpp_validation.py --extract-missing --gpu 0   # GPU 0-5 only
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

from study_threshold.io import (  # noqa: E402
    aggregate_videos,
    assert_modes_share_videos,
    load_source_frames,
    val_cache_path,
)
from study_threshold.reporting import cell_id, phase_paths, write_table  # noqa: E402

OUT_NAME = "source_video_scores.parquet"


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PHASE / "configs/threshold_study.yaml").read_text())


def resolve_dims(cfg: dict, dry_run: bool) -> tuple[list[str], list[int], list[str]]:
    if dry_run:
        d = cfg["dry_run"]
        return d["backbones"], d["seeds"], d["methods"]
    d = cfg["dims"]
    return d["backbones"], d["seeds"], d["methods"]


def resolve_cache_dir(cfg: dict, paths: dict, checkpoint_id: str) -> Path:
    """Prefer a locally re-extracted cache, fall back to the phase-4 cache."""
    local = paths["val_cache"]
    if val_cache_path(local, checkpoint_id).is_file():
        return local
    return ROOT / cfg["inputs"]["source_val_cache"]


def extract_missing(cfg: dict, paths: dict, checkpoint_id: str, gpu: int) -> Path:
    """Re-infer FF++ val for one checkpoint into THIS phase's val_cache (GPU)."""
    if not 0 <= gpu <= 5:
        raise SystemExit("only GPUs 0-5 may be used in this repository")
    import numpy as np
    import torch

    from data.builders import build_external_eval_loader, build_val_loader
    from diagnosis.checkpoint_index import discover_checkpoints
    from diagnosis.feature_extractor import extract_raw_features
    from study.q4_extract import build_model_from_checkpoint

    infos = [i for i in discover_checkpoints(str(ROOT / cfg["inputs"]["checkpoint_root"]))
             if i.checkpoint_id == checkpoint_id]
    if not infos:
        raise SystemExit(f"checkpoint {checkpoint_id} not found")
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model, model_cfg = build_model_from_checkpoint(infos[0], device=device)
    w = model.classifier.weight.detach().cpu().numpy()
    b = model.classifier.bias.detach().cpu().numpy()
    w_signed, b_signed = (w[1] - w[0]).astype("float64"), float(b[1] - b[0])

    model_cfg = dict(model_cfg)
    model_cfg.setdefault("train", {})
    model_cfg["train"].setdefault("per_gpu_batch", 64)
    model_cfg["train"].setdefault("num_workers", 8)
    records = list(build_val_loader(model_cfg, distributed=False).dataset.records)
    loader = build_external_eval_loader(model_cfg, records=records, distributed=False,
                                        batch_size=64, num_workers=8)
    feats = extract_raw_features(model, loader, device)
    z, r = feats.features, feats.feature_norms
    df = pd.DataFrame({
        "image_path": feats.image_paths, "video_id": feats.video_ids,
        "method_id": feats.method_ids, "label": feats.labels.astype(int),
        "s_raw": z @ w_signed + b_signed,
        "s_dir": (z / r[:, None]) @ w_signed + b_signed,
        "feature_norm": r,
    })
    out = val_cache_path(paths["val_cache"], checkpoint_id)
    df.to_parquet(out)
    print(f"[extract] {out}  ({len(df)} frames)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extract-missing", action="store_true",
                    help="re-infer FF++ val for checkpoints without a cache (GPU 0-5)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config) if args.config else None)
    paths = phase_paths(PHASE, cfg)
    backbones, seeds, methods = resolve_dims(cfg, args.dry_run)
    aggregator = cfg["protocol"]["video_aggregator"]

    videos, summary = [], []
    for backbone in backbones:
        for seed in seeds:
            ckpt = cell_id(backbone, seed)
            cache_dir = resolve_cache_dir(cfg, paths, ckpt)
            if not val_cache_path(cache_dir, ckpt).is_file():
                if not args.extract_missing:
                    raise SystemExit(
                        f"no FF++ val cache for {ckpt}; rerun with --extract-missing (GPU 0-5)"
                    )
                extract_missing(cfg, paths, ckpt, args.gpu)
                cache_dir = paths["val_cache"]

            frames = load_source_frames(cache_dir, ckpt, methods=methods)
            table = aggregate_videos(frames, aggregator=aggregator)
            assert_modes_share_videos(table)
            table.insert(0, "backbone", backbone)
            table.insert(1, "seed", seed)
            table.insert(2, "checkpoint_id", ckpt)
            videos.append(table)
            summary.append({
                "checkpoint_id": ckpt, "backbone": backbone, "seed": seed,
                "cache_dir": str(Path(cache_dir).relative_to(ROOT)),
                "n_frames": int(len(frames)),
                "n_videos": int(len(table)),
                "n_real_videos": int((table.label == 0).sum()),
                "n_fake_videos": int((table.label == 1).sum()),
                "n_methods": int(table.loc[table.label == 1, "method_id"].nunique()),
                "median_frames_per_video": float(table.n_frames.median()),
                "aggregator": aggregator,
            })
            print(f"[source] {ckpt}: {len(table)} videos "
                  f"({int((table.label == 0).sum())} real / {int((table.label == 1).sum())} fake)")

    all_videos = pd.concat(videos, ignore_index=True)
    out = paths["val_cache"] / OUT_NAME
    all_videos.to_parquet(out, index=False)
    print(f"[write] {out}  ({len(all_videos)} rows)")
    write_table(pd.DataFrame(summary), paths["tables"] / "01_source_videos.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
