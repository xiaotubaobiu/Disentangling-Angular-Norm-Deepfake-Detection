"""Encode each E2 evaluation frame once with the locked frozen H1 vision tower."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config_io import load_config
from data.builders import build_eval_dataset
from evaluation.e2_feature_cache import build_feature_cache
from models.factory import build_model


def _repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--domains", nargs="+", default=("ffpp", "cdf", "dfd"),
                        choices=("ffpp", "cdf", "dfd"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if len(args.domains) != len(set(args.domains)):
        parser.error("--domains must not contain duplicates")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("cache loader size must be positive and workers nonnegative")

    cfg = load_config(args.config)
    cfg["train"]["seed"] = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    model.load_h1_visual_checkpoint(_repo_path(cfg["train"]["init_visual_checkpoint"]))
    model.eval()
    for domain in args.domains:
        dataset = build_eval_dataset(cfg, domain)
        cache = build_feature_cache(model, dataset, args.output_root, domain=domain,
                                    cfg=cfg, device=device, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        print(json.dumps({"domain": domain, "frames": len(dataset),
                          "feature_shape": cache.manifest["shape"],
                          "visual_sha256": cache.manifest["visual_sha256"],
                          "cache_root": str(args.output_root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
