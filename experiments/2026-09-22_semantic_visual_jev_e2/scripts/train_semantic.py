"""Leakage-safe DDP training for the E2 semantic Visual-Jev model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config_io import load_config
from data.builders import (
    build_semantic_train_loader,
    build_semantic_val_loader,
)
from data.semantic_language import load_language_bank
from engine.ddp import barrier, init_ddp, is_main_process
from engine.semantic_trainer import (
    TrainableEMA,
    run_semantic_train_epoch,
    run_semantic_val_epoch,
    trainable_parameters,
)
from models.clip_semantic_jev import EncodedTextLookup
from models.factory import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the locked E2 semantic Visual-Jev model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 7, 2048))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args(argv)
    max_batches = (args.max_train_batches, args.max_val_batches)
    if not args.smoke and any(value is not None for value in max_batches):
        parser.error("max-batch flags are allowed only with --smoke")
    if any(value is not None and value <= 0 for value in max_batches):
        parser.error("max-batch values must be positive")
    return args


def checkpoint_path(cfg: dict, seed: int, smoke: bool) -> Path:
    root = Path(cfg["train"]["output_dir"])
    if smoke:
        return root / "smoke" / f"seed_{seed}" / "best_model.pth"
    return root / f"seed_{seed}" / "best_model.pth"


def ensure_checkpoint_writable(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {path}; pass --allow-overwrite to replace it"
        )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_training_epoch(loader, epoch: int) -> None:
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    loader.dataset.set_epoch(epoch)


def build_training_text_lookup(model, language_bank, device):
    return EncodedTextLookup.build(
        model,
        language_bank,
        allowed_splits=("train", "fixed"),
        device=device,
    )


def build_training_loaders(cfg, language_bank, distributed):
    train_loader = build_semantic_train_loader(
        cfg, language_bank, distributed=distributed
    )
    seen_val_loader = build_semantic_val_loader(
        cfg,
        language_bank,
        distributed=distributed,
        split="train",
    )
    return train_loader, seen_val_loader


def is_better_seen_ce(val_seen_ce: float, best_seen_ce: float) -> bool:
    return math.isfinite(val_seen_ce) and val_seen_ce < best_seen_ce


def effective_epochs(cfg: dict, smoke: bool) -> int:
    return 1 if smoke else int(cfg["train"]["epochs"])


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def _clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in _raw_model(model).state_dict().items()
    }


def build_checkpoint_payload(
    model,
    ema,
    *,
    epoch: int,
    val_seen_ce: float,
    seed: int,
    language_manifest_sha256: str,
    cfg: dict,
) -> dict:
    if ema is not None:
        ema.apply(model)
    try:
        model_state = _clone_state_dict(model)
    finally:
        if ema is not None:
            ema.restore(model)
    return {
        "epoch": int(epoch),
        "model_state_dict": model_state,
        "ema_state_dict": None if ema is None else ema.state_dict(),
        "val_seen_ce": float(val_seen_ce),
        "seed": int(seed),
        "language_manifest_sha256": language_manifest_sha256,
        "config": copy.deepcopy(cfg),
    }


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _limited(loader, maximum: int | None):
    return loader if maximum is None else itertools.islice(loader, maximum)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_locked_training_contract(cfg: dict, seed: int) -> None:
    if cfg.get("loss", {}).get("name") != "semantic_cross_entropy":
        raise ValueError("E2 training permits semantic_cross_entropy only")
    if cfg.get("model", {}).get("name") != "clip_semantic_jev":
        raise ValueError("E2 training requires clip_semantic_jev")
    if seed not in cfg.get("train", {}).get("seeds", ()):
        raise ValueError(f"seed {seed} is not registered in the locked config")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    _validate_locked_training_contract(cfg, args.seed)
    cfg = copy.deepcopy(cfg)
    cfg["train"]["seed"] = args.seed

    save_path = checkpoint_path(cfg, args.seed, args.smoke)
    ensure_checkpoint_writable(save_path, args.allow_overwrite)
    configure_determinism(args.seed)

    local_rank = init_ddp()
    distributed = dist.is_available() and dist.is_initialized()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    try:
        language_manifest = _repo_path(cfg["language"]["manifest"])
        language_bank = load_language_bank(language_manifest)
        manifest_hash = file_sha256(language_manifest)

        raw_model = build_model(cfg["model"]).to(device)
        visual_checkpoint = _repo_path(
            cfg["train"]["init_visual_checkpoint"]
        )
        raw_model.load_h1_visual_checkpoint(visual_checkpoint)
        text_lookup = build_training_text_lookup(
            raw_model, language_bank, device
        )

        model = raw_model
        if distributed:
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )

        decision_parameters = trainable_parameters(model)
        optimizer = AdamW(
            decision_parameters,
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        epochs = effective_epochs(cfg, args.smoke)
        scheduler = build_scheduler(
            optimizer,
            epochs,
            cfg["train"]["warmup_epochs"],
        )
        use_amp = (
            device.type == "cuda"
            and cfg["train"].get("precision") == "amp"
        )
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        ema = TrainableEMA(model, decay=cfg["train"]["ema_decay"])

        train_loader, seen_val_loader = build_training_loaders(
            cfg, language_bank, distributed
        )
        max_train_batches = args.max_train_batches
        max_val_batches = args.max_val_batches
        if args.smoke:
            max_train_batches = max_train_batches or 1
            max_val_batches = max_val_batches or 1

        best_seen_ce = float("inf")
        patience_counter = 0
        best_epoch = None
        for epoch in range(epochs):
            set_training_epoch(train_loader, epoch)
            train_loss = run_semantic_train_epoch(
                model,
                _limited(train_loader, max_train_batches),
                optimizer,
                scaler,
                device,
                text_lookup,
                cfg,
                ema=ema,
            )

            ema.apply(model)
            try:
                val_metrics = run_semantic_val_epoch(
                    model,
                    _limited(seen_val_loader, max_val_batches),
                    text_lookup,
                    device,
                )
            finally:
                ema.restore(model)
            val_seen_ce = val_metrics["loss"]

            if is_main_process():
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "train_ce": train_loss,
                            "val_seen_ce": val_seen_ce,
                            "val_seen_accuracy": val_metrics["accuracy"],
                            "val_seen_count": val_metrics["count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if is_better_seen_ce(val_seen_ce, best_seen_ce):
                best_seen_ce = val_seen_ce
                best_epoch = epoch
                patience_counter = 0
                if is_main_process():
                    payload = build_checkpoint_payload(
                        model,
                        ema,
                        epoch=epoch,
                        val_seen_ce=val_seen_ce,
                        seed=args.seed,
                        language_manifest_sha256=manifest_hash,
                        cfg=cfg,
                    )
                    _atomic_save(payload, save_path)
            else:
                patience_counter += 1

            scheduler.step()
            barrier()
            if patience_counter >= cfg["train"]["patience"]:
                break

        if best_epoch is None:
            raise RuntimeError("training produced no finite seen-language validation CE")
        if is_main_process():
            print(
                json.dumps(
                    {
                        "best_epoch": best_epoch + 1,
                        "best_val_seen_ce": best_seen_ce,
                        "checkpoint": str(save_path),
                        "seed": args.seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        barrier()
        return 0
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
