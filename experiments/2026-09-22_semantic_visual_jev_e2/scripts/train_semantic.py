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
from torch.utils.data import DataLoader


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


EFFECTIVE_H1_AUGMENT_POLICY = {
    "pipeline": "build_rgb_augment",
    "transforms": [
        {"name": "HorizontalFlip", "p": 0.5},
        {"name": "Rotate", "p": 0.5, "limit": [-10, 10]},
        {"name": "GaussianBlur", "p": 0.5, "blur_limit": [3, 7]},
        {
            "name": "OneOf",
            "p": 0.5,
            "transforms": [
                {
                    "name": "RandomBrightnessContrast",
                    "brightness_limit": 0.1,
                    "contrast_limit": 0.1,
                },
                {"name": "FancyPCA"},
                {"name": "HueSaturationValue"},
            ],
        },
        {
            "name": "ImageCompression",
            "p": 0.5,
            "quality_range": [40, 100],
        },
    ],
    "normalization": {
        "name": "CLIP",
        "selected_by": "clip_model",
        "mean": [0.48145466, 0.45762756, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}
TRAIN_TEXT_SPLITS = ("train", "fixed")


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
        allowed_splits=TRAIN_TEXT_SPLITS,
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


def build_grad_scaler(device: torch.device, precision: str):
    use_amp = device.type == "cuda" and precision == "amp"
    return torch.cuda.amp.GradScaler(enabled=use_amp)


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
    smoke: bool,
    config_sha256: str,
    visual_checkpoint_sha256: str,
    rank_diagnostics: list[dict],
    runtime_validated: bool = False,
) -> dict:
    if not smoke and not runtime_validated:
        raise ValueError("reportable checkpoint requires validated four-GPU runtime")
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
        "config_sha256": config_sha256,
        "visual_checkpoint_sha256": visual_checkpoint_sha256,
        "encoded_text_splits": list(TRAIN_TEXT_SPLITS),
        "smoke": bool(smoke),
        "reportable": not smoke,
        "runtime_validated": bool(runtime_validated),
        "world_size": len(rank_diagnostics),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "rank_diagnostics": copy.deepcopy(rank_diagnostics),
        "attention_backend": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
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
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


class _BoundedBatchSampler:
    def __init__(self, batch_sampler, maximum: int):
        self.batch_sampler = batch_sampler
        self.maximum = maximum

    def __iter__(self):
        return itertools.islice(iter(self.batch_sampler), self.maximum)

    def __len__(self):
        return min(len(self.batch_sampler), self.maximum)


def bounded_loader(loader: DataLoader, maximum: int | None) -> DataLoader:
    """Construct an exhaustible smoke loader before spawning any workers."""
    if maximum is None:
        return loader
    if maximum < 1:
        raise ValueError("maximum smoke batches must be positive")
    options = dict(
        dataset=loader.dataset,
        batch_sampler=_BoundedBatchSampler(loader.batch_sampler, maximum),
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
        persistent_workers=False,
    )
    if loader.num_workers:
        options["prefetch_factor"] = loader.prefetch_factor
        options["multiprocessing_context"] = loader.multiprocessing_context
    return DataLoader(**options)


def collect_rank_diagnostics(local: dict) -> list[dict]:
    if not (dist.is_available() and dist.is_initialized()):
        return [local]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return sorted(gathered, key=lambda item: item["rank"])


def validate_smoke_diagnostics(ranks: list[dict]) -> None:
    if len(ranks) != 4 or {item["rank"] for item in ranks} != {0, 1, 2, 3}:
        raise ValueError("smoke requires evidence from exactly ranks 0 through 3")
    for item in ranks:
        if item["train_batches"] < 1 or item["val_batches"] < 1:
            raise ValueError("each smoke rank must process train and validation batches")
        peak = item["peak_cuda_allocated_bytes"]
        if peak <= 0 or peak >= 30 * 1024**3:
            raise ValueError("each smoke rank must stay below 30 GiB peak CUDA allocation")
    for width in ("2", "3"):
        if sum(item["candidate_counts"].get(width, 0) for item in ranks) < 1:
            raise ValueError(f"smoke requires {width}-candidate examples")


def validate_reportable_runtime(
    device: torch.device, world_size: int, visible_devices: str, cuda_devices: int
) -> None:
    """Forbid CPU, partial-GPU, or wrong-GPU runs from becoming reportable."""
    if (device.type != "cuda" or world_size != 4 or visible_devices != "0,1,2,3"
            or cuda_devices != 4):
        raise ValueError(
            "reportable training requires four DDP CUDA ranks on exactly "
            "CUDA_VISIBLE_DEVICES=0,1,2,3"
        )


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def synchronized_main_process(operation) -> None:
    """Propagate checkpoint serialization failures to every DDP rank."""
    error = None
    if is_main_process():
        try:
            operation()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        outcome = [error]
        dist.broadcast_object_list(outcome, src=0)
        error = outcome[0]
    if error is not None:
        raise RuntimeError(error)


def _validate_locked_training_contract(cfg: dict, seed: int) -> None:
    validate_h1_augment_policy(cfg)
    if cfg.get("loss", {}).get("name") != "semantic_cross_entropy":
        raise ValueError("E2 training permits semantic_cross_entropy only")
    if cfg.get("model", {}).get("name") != "clip_semantic_jev":
        raise ValueError("E2 training requires clip_semantic_jev")
    if seed not in cfg.get("train", {}).get("seeds", ()):
        raise ValueError(f"seed {seed} is not registered in the locked config")


def validate_h1_augment_policy(cfg: dict) -> None:
    if cfg.get("augment") != EFFECTIVE_H1_AUGMENT_POLICY:
        raise ValueError(
            "augment must exactly match the effective H1 augmentation policy"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    _validate_locked_training_contract(cfg, args.seed)
    cfg = copy.deepcopy(cfg)
    cfg["train"]["seed"] = args.seed
    config_hash = file_sha256(_repo_path(args.config))

    save_path = checkpoint_path(cfg, args.seed, args.smoke)
    ensure_checkpoint_writable(save_path, args.allow_overwrite)
    configure_determinism(args.seed)

    local_rank = init_ddp()
    distributed = dist.is_available() and dist.is_initialized()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    runtime_validated = False
    if not args.smoke:
        validate_reportable_runtime(
            device, dist.get_world_size() if distributed else 1,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""), torch.cuda.device_count(),
        )
        runtime_validated = True

    try:
        language_manifest = _repo_path(cfg["language"]["manifest"])
        language_bank = load_language_bank(language_manifest)
        manifest_hash = file_sha256(language_manifest)

        raw_model = build_model(cfg["model"]).to(device)
        visual_checkpoint = _repo_path(
            cfg["train"]["init_visual_checkpoint"]
        )
        visual_checkpoint_hash = file_sha256(visual_checkpoint)
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
        scaler = build_grad_scaler(
            device, precision=cfg["train"].get("precision", "amp")
        )
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
            diagnostics = {"rank": dist.get_rank() if distributed else 0,
                           "train_batches": 0, "val_batches": 0,
                           "candidate_counts": {"2": 0, "3": 0}}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            set_training_epoch(train_loader, epoch)
            train_loss = run_semantic_train_epoch(
                model,
                bounded_loader(train_loader, max_train_batches),
                optimizer,
                scaler,
                device,
                text_lookup,
                cfg,
                ema=ema,
                diagnostics=diagnostics,
            )

            ema.apply(model)
            try:
                val_metrics = run_semantic_val_epoch(
                    model,
                    bounded_loader(seen_val_loader, max_val_batches),
                    text_lookup,
                    device,
                    diagnostics=diagnostics,
                )
            finally:
                ema.restore(model)
            val_seen_ce = val_metrics["loss"]
            diagnostics["peak_cuda_allocated_bytes"] = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            )
            rank_diagnostics = collect_rank_diagnostics(diagnostics)
            validate_smoke_diagnostics(rank_diagnostics)
            if not math.isfinite(train_loss) or not math.isfinite(val_seen_ce):
                raise RuntimeError("training or seen-language validation loss is nonfinite")

            if is_main_process():
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "train_ce": train_loss,
                            "val_seen_ce": val_seen_ce,
                            "val_seen_accuracy": val_metrics["accuracy"],
                            "val_seen_count": val_metrics["count"],
                            "rank_diagnostics": rank_diagnostics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if is_better_seen_ce(val_seen_ce, best_seen_ce):
                best_seen_ce = val_seen_ce
                best_epoch = epoch
                patience_counter = 0
                def save_best_checkpoint():
                    payload = build_checkpoint_payload(
                        model,
                        ema,
                        epoch=epoch,
                        val_seen_ce=val_seen_ce,
                        seed=args.seed,
                        language_manifest_sha256=manifest_hash,
                        cfg=cfg,
                        smoke=args.smoke,
                        config_sha256=config_hash,
                        visual_checkpoint_sha256=visual_checkpoint_hash,
                        rank_diagnostics=rank_diagnostics,
                        runtime_validated=runtime_validated,
                    )
                    _atomic_save(payload, save_path)
                synchronized_main_process(save_best_checkpoint)
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
