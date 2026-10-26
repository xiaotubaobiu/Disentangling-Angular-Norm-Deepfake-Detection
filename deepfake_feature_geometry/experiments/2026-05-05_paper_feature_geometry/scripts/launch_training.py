#!/usr/bin/env python3
"""Generate and optionally launch training configs for the multi-backbone study.

Inputs : repo-root `configs/<backbone>_base.yaml` templates (referenced by the
         generated configs through `_base_`).
Outputs: one YAML per (backbone, variant, seed) under `--config-dir`
         (default: this phase's `configs/multibb/`), optionally followed by
         `torchrun ... train.py` runs.
GPU    : only with `--run` (training). Config generation is CPU-only.
Target-domain labels: not used.

Run standalone:
    python experiments/2026-05-05_paper_feature_geometry/scripts/launch_training.py \
        --efficientnet-lr 3e-4 --dinov2-lr 5e-5

Supported training backbones: efficientnet, dinov2 (and CLIP, which is trained
straight from `configs/clip_base.yaml`, not through this generator).

Xception is a *legacy* backbone: `src/models/xception.py` and its factory entry
are kept so old checkpoints stay loadable, but the training configs were removed
on 2026-07-29 and are only present inside
`.deleted_configs_backup_20260729.tar.gz`. Asking this script for Xception fails
with an explicit message instead of generating a config that points at a
non-existent `_base_`.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
_TRAIN = _PHASE / "scripts" / "train.py"

XCEPTION_ARCHIVED_MSG = (
    "Xception training configs are archived in "
    ".deleted_configs_backup_20260729.tar.gz and are not maintained in the "
    "current experiment."
)

# Backbones this generator still maintains configs for.
BACKBONES = {
    "efficientnet": {
        "base_config": "configs/efficientnet_base.yaml",
        "model_name": "efficientnet_b0",
        "lr": None,
    },
    "dinov2": {
        "base_config": "configs/dinov2_base.yaml",
        "model_name": "dinov2_vitb14",
        "lr": None,
    },
}

# Backbones whose model code still exists but whose training configs are archived.
ARCHIVED_BACKBONES = {"xception": XCEPTION_ARCHIVED_MSG}

SEEDS = [42, 7, 123, 999, 2048]
VARIANTS = [
    ("raw", False),
    ("normtrain", True),
]


def resolve_backbones(requested: list[str] | None) -> list[str]:
    """Validate a backbone selection, rejecting archived ones with a clear error."""
    names = list(BACKBONES) if not requested else list(requested)
    for name in names:
        if name in ARCHIVED_BACKBONES:
            raise SystemExit(ARCHIVED_BACKBONES[name])
        if name not in BACKBONES:
            raise SystemExit(
                f"Unknown backbone: {name}. Supported: {', '.join(BACKBONES)}"
            )
    return names


def generate_config(backbone, variant, seed, model_name, normalize, lr, base_config, output_dir):
    experiment_name = f"multibb_{variant}_s{seed}_{backbone}"
    config_text = f"""_base_: {base_config}
experiment_name: {experiment_name}
model:
  name: {model_name}
  normalize_features: {str(normalize).lower()}
train:
  lr: {lr}
  seed: {seed}
"""
    config_path = output_dir / f"{experiment_name}.yaml"
    config_path.write_text(config_text)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--efficientnet-lr", type=float, default=None)
    parser.add_argument("--dinov2-lr", type=float, default=None)
    parser.add_argument("--backbones", nargs="+", default=None,
                        help=f"Subset of {', '.join(BACKBONES)} (default: all supported)")
    parser.add_argument("--run", action="store_true", help="Launch training (otherwise just generate configs)")
    parser.add_argument("--config-dir", default=str(_PHASE / "configs" / "multibb"))
    args = parser.parse_args()

    backbones = resolve_backbones(args.backbones)

    lrs = {"efficientnet": args.efficientnet_lr, "dinov2": args.dinov2_lr}
    missing = [b for b in backbones if lrs.get(b) is None]
    if missing:
        raise SystemExit(
            "Missing learning rate for: " + ", ".join(missing)
            + " (pass --" + "-lr, --".join(missing) + "-lr)"
        )
    for name in backbones:
        BACKBONES[name]["lr"] = lrs[name]

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for bb_name in backbones:
        bb_cfg = BACKBONES[bb_name]
        for variant_name, normalize in VARIANTS:
            for seed in SEEDS:
                path = generate_config(
                    backbone=bb_name, variant=variant_name, seed=seed,
                    model_name=bb_cfg["model_name"], normalize=normalize,
                    lr=bb_cfg["lr"], base_config=bb_cfg["base_config"],
                    output_dir=config_dir,
                )
                configs.append(path)

    print(f"Generated {len(configs)} configs in {config_dir}/")
    for c in sorted(configs):
        print(f"  {c.name}")

    if args.run:
        for i, config_path in enumerate(sorted(configs)):
            print(f"\n[{i+1}/{len(configs)}] Training {config_path.name}")
            subprocess.run(
                ["torchrun", "--nproc_per_node=8", str(_TRAIN), "--config", str(config_path)],
                check=True,
            )
        print("All training complete.")


if __name__ == "__main__":
    main()
