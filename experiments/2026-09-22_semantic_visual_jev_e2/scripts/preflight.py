"""Fail-closed E2 source, checkpoint, language, model, and GPU preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config_io import load_config
from data.constants import ALL_METHODS
from data.index_ffpp import load_json_index
from data.semantic_dataset import TASKS, method_family
from data.semantic_language import load_language_bank
from models.factory import build_model


H1_CHECKPOINTS = {
    42: "experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth",
    7: "experiments/2026-09-21_visual_jev_h1/results/training/seed_7/20260921_203731/output/best_model.pth",
    2048: "experiments/2026-09-21_visual_jev_h1/results/training/seed_2048/20260921_203738/output/best_model.pth",
}
H1_BASE_CONFIG = REPO_ROOT / "experiments/2026-09-21_visual_jev_h1/configs/visual_jev_base.yaml"
SPLITS = ("train", "val", "ffpp", "cdf")
TRAINABLE_PREFIXES = ("image_projection.", "question_projection.",
                      "candidate_projection.", "decoder.", "scorer.")


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_inventory(inventory: dict) -> None:
    if set(inventory) != set(ALL_METHODS):
        raise ValueError("dataset inventory must cover exactly the 16 H1 methods")
    for method, splits in inventory.items():
        for split in SPLITS:
            if int(splits.get(split, 0)) < 1:
                raise ValueError(f"{method}/{split} has no indexed fake frames")


def validate_devices(devices: list[dict], visible: str | None) -> None:
    if visible != "0,1,2,3" or len(devices) != 4 or {d["index"] for d in devices} != set(range(4)):
        raise ValueError("E2 execution requires exactly CUDA_VISIBLE_DEVICES=0,1,2,3")
    if any(d["free_bytes"] < 28 * 1024**3 for d in devices):
        raise ValueError("each of GPUs 0,1,2,3 needs at least 28 GiB free")


def index_split_records(cfg: dict, method: str, name: str, label: int):
    root = cfg["dataset"]["root"]
    first = int(cfg["dataset"]["train_videos_per_method"])
    if name == "train":
        return load_json_index(root, method, "ff", "train", label,
                               max_videos=first)
    if name == "val":
        count = int(cfg["dataset"]["val_videos_per_method"])
        return load_json_index(root, method, "ff", "train", label,
                               video_range=(first, first + count))
    if name in ("ffpp", "cdf"):
        return load_json_index(root, method, "ff" if name == "ffpp" else "cdf",
                               "test", label)
    raise ValueError(f"unknown E2 split: {name}")


def _dataset_inventory(cfg: dict) -> tuple[dict, dict]:
    inventory, real_inventory = {}, {}
    for method in ALL_METHODS:
        inventory[method], real_inventory[method] = {}, {}
        for name in SPLITS:
            for label, target in ((1, inventory), (0, real_inventory)):
                records = index_split_records(cfg, method, name, label)
                target[method][name] = len(records)
                expected_family = method_family(method if label else "real", label)
                if expected_family == "unknown" or any(
                    record.label != label or method_family(record.method, record.label) != expected_family
                    for record in records
                ):
                    raise ValueError(f"{method}/{name}: indexed label or method family contradicts H1 truth")
                if records and not all(Path(record.frame_path).is_file() for record in (records[0], records[-1])):
                    raise ValueError(f"{method}/{name}: sampled frame path is missing")
    validate_dataset_inventory(inventory)
    if any(count < 1 for splits in real_inventory.values() for count in splits.values()):
        raise ValueError("each H1 method/split must also expose real records")
    return inventory, real_inventory


def _language_inventory(path: Path) -> dict:
    bank = load_language_bank(path)
    if set(bank.tasks) != set(TASKS):
        raise ValueError("language manifest must contain exactly T1–T5")
    train_questions = sum(len(bank.task(task, "train").questions) for task in TASKS)
    if train_questions != 9:
        raise ValueError(f"expected nine training questions, found {train_questions}")
    train_text = {entry.text for entry in bank.entries("train")}
    heldout_text = {entry.text for entry in bank.entries("heldout")}
    if not train_text or not heldout_text or train_text & heldout_text:
        raise ValueError("train/heldout language is empty or overlapping")
    return {"sha256": _sha256(path), "train_questions": train_questions,
            "tasks": list(TASKS), "train_texts": len(train_text),
            "heldout_texts": len(heldout_text)}


def _model_inventory(cfg: dict) -> dict:
    model = build_model(cfg["model"])
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith(TRAINABLE_PREFIXES) for name in trainable):
        raise ValueError("E2 trainable names exceed projections, decoder, and scorer")
    if any(not any(name.startswith(prefix) for name in trainable) for prefix in TRAINABLE_PREFIXES):
        raise ValueError("E2 trainable decision module is missing")
    if any(parameter.requires_grad for parameter in model.clip.parameters()):
        raise ValueError("the CLIP towers must remain frozen")
    return {"trainable_names": trainable,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters())}


def _checkpoint_inventory(cfg: dict) -> dict:
    base_cfg = load_config(H1_BASE_CONFIG)
    baseline_a = _repo_path(base_cfg["train"]["init_checkpoint"])
    paths = {"h1_base": baseline_a,
             **{f"h1_jev_seed_{seed}": _repo_path(path) for seed, path in H1_CHECKPOINTS.items()}}
    expected_visual = _repo_path(cfg["train"]["init_visual_checkpoint"])
    if expected_visual != paths["h1_jev_seed_42"]:
        raise ValueError("E2 visual initialization differs from locked H1-Jev seed 42")
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing {name} checkpoint: {path}")
    return {name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()}


def _device_inventory() -> list[dict]:
    return [{"index": index, "name": torch.cuda.get_device_name(index),
             "free_bytes": int(torch.cuda.mem_get_info(index)[0]),
             "total_bytes": int(torch.cuda.get_device_properties(index).total_memory)}
            for index in range(torch.cuda.device_count())]


def build_preflight(cfg: dict, config_path: Path) -> dict:
    from importlib.util import module_from_spec, spec_from_file_location

    train_path = Path(__file__).with_name("train_semantic.py")
    spec = spec_from_file_location("e2_train_preflight", train_path)
    train_module = module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_module._validate_locked_training_contract(cfg, 42)
    methods, real = _dataset_inventory(cfg)
    language_path = _repo_path(cfg["language"]["manifest"])
    language = _language_inventory(language_path)
    checkpoints = _checkpoint_inventory(cfg)
    model = _model_inventory(cfg)
    devices = _device_inventory()
    validate_devices(devices, os.environ.get("CUDA_VISIBLE_DEVICES"))
    return {"valid": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.resolve()), "config_sha256": _sha256(config_path),
            "dataset_root": cfg["dataset"]["root"], "fake_frames": methods,
            "real_frames": real, "language": language, "checkpoints": checkpoints,
            "model": model, "devices": devices}


def _git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=v1", "-uall"],
                           cwd=REPO_ROOT, check=True, capture_output=True,
                           text=True).stdout.splitlines()
    return {"git_commit": head, "dirty_files": dirty}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock-output", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        result = build_preflight(cfg, args.config)
        if args.lock_output is not None:
            result = {**result, **_git_state()}
            _write_json(args.lock_output, result)
        _write_json(args.output, result)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"E2 preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"valid": True, "output": str(args.output),
                      "visible_gpus": len(result["devices"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
