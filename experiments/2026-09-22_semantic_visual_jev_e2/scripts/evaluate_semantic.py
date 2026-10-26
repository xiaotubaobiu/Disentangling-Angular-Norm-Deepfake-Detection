"""Distributed, leakage-safe evaluation for E2 Semantic Visual-Jev."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config_io import load_config
from data.builders import build_eval_dataset, build_semantic_eval_loader
from data.semantic_dataset import TASKS, method_family
from data.semantic_language import SEMANTIC_TO_ID, load_language_bank
from engine.ddp import barrier, init_ddp, is_main_process
from evaluation.sample_eval import gather_rows_across_ranks
from evaluation.semantic_eval import (
    aggregate_semantic_videos,
    compute_task_metrics,
    deduplicate_distributed_rows,
    frame_semantic_rows,
)
from models.clip_semantic_jev import EncodedTextLookup
from models.factory import build_model


ConditionSpec = namedtuple(
    "ConditionSpec", ("text_splits", "language_split", "fixed_h1"),
    defaults=(False,),
)


# Task 6 extends this registry with controls.  Keeping construction and
# inference behind these hooks avoids embedding control behavior in Task 5.
CONDITION_SPECS = {
    "fixed_h1": ConditionSpec(("fixed",), "fixed", fixed_h1=True),
    "seen": ConditionSpec(("train",), "train"),
    "heldout": ConditionSpec(("heldout",), "heldout"),
}

_MANIFEST_GATED_CONDITIONS = frozenset(
    {
        "heldout",
        "permutation",
        "counterfactual",
        "meaningless_candidates",
        "meaningless_questions",
    }
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the locked E2 semantic Visual-Jev checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 7, 2048))
    parser.add_argument(
        "--domains", nargs="+", default=("ffpp", "cdf", "dfd"),
        choices=("ffpp", "cdf", "dfd"),
    )
    parser.add_argument(
        "--conditions", nargs="+", default=tuple(CONDITION_SPECS),
        choices=tuple(CONDITION_SPECS),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if len(args.domains) != len(set(args.domains)):
        parser.error("--domains must not contain duplicates")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("--conditions must not contain duplicates")
    return args


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_text_splits(condition: str) -> tuple[str, ...]:
    try:
        return CONDITION_SPECS[condition].text_splits
    except KeyError as error:
        raise ValueError(f"unsupported evaluation condition: {condition}") from error


def validate_checkpoint_manifest(
    checkpoint: dict, expected_sha256: str, conditions
) -> None:
    """Gate held-out/control evaluation on the exact training manifest."""
    if not (_MANIFEST_GATED_CONDITIONS & set(conditions)):
        return
    actual = checkpoint.get("language_manifest_sha256")
    if actual != expected_sha256:
        raise ValueError(
            "checkpoint language manifest SHA256 is missing or does not match "
            "the requested held-out/control evaluation manifest"
        )


class FixedH1Dataset(Dataset):
    """Render the exact fixed H1 text while preserving semantic row metadata."""

    def __init__(self, base_dataset, language_bank):
        self.base_dataset = base_dataset
        self.question_text_id = language_bank.text_id("fixed_h1_q")
        self.real_text_id = language_bank.text_id("fixed_h1_real")
        self.fake_text_id = language_bank.text_id("fixed_h1_fake")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = dict(self.base_dataset[index])
        label = int(sample["label"])
        sample.update(
            {
                "task_id": "T1",
                "task_index": torch.tensor(0, dtype=torch.long),
                "question_text_id": torch.tensor(
                    self.question_text_id, dtype=torch.long
                ),
                "question_id": "fixed_h1_q",
                "question_yes_means": torch.tensor(-1, dtype=torch.long),
                "question_yes_means_name": "",
                "question_pair_id": "",
                "candidate_text_ids": torch.tensor(
                    [self.real_text_id, self.fake_text_id, -1], dtype=torch.long
                ),
                "candidate_semantic_ids": torch.tensor(
                    [SEMANTIC_TO_ID["real"], SEMANTIC_TO_ID["fake"], -1],
                    dtype=torch.long,
                ),
                "candidate_mask": torch.tensor([True, True, False]),
                "candidate_ids": ("fixed_h1_real", "fixed_h1_fake", ""),
                "candidate_texts": (
                    "an authentic facial image.",
                    "a manipulated deepfake facial image.",
                    "",
                ),
                "candidate_semantics": ("real", "fake", ""),
                "target_index": torch.tensor(label, dtype=torch.long),
                "target_semantic": "fake" if label else "real",
                "method_family": method_family(sample["method_id"], label),
                "variant_id": "fixed_h1_q::fixed_h1_candidates",
                "render_id": f"fixed_h1::{sample['sample_id']}",
            }
        )
        return sample


class PairedT4EvaluationDataset(Dataset):
    """Expand a semantic suite so every T4 frame has one exact family pair."""

    def __init__(self, semantic_dataset):
        if getattr(semantic_dataset, "mode", None) != "suite":
            raise ValueError("paired T4 evaluation requires a suite dataset")
        self.base_dataset = semantic_dataset.base_dataset
        self.sampler = semantic_dataset.sampler
        self.tasks = semantic_dataset.tasks
        self.split = semantic_dataset.split
        self.epoch = semantic_dataset.epoch

        t4_questions = {}
        if "T4" in self.tasks:
            language = self.sampler.language_bank.task("T4", self.split)
            for polarity in ("swap", "reenact"):
                matching = tuple(
                    question
                    for question in language.questions
                    if question.yes_means == polarity
                )
                if len(matching) != 1:
                    raise ValueError(
                        f"T4 {self.split} suite requires exactly one {polarity} question"
                    )
                t4_questions[polarity] = matching[0].name

        slots = []
        for task_id in self.tasks:
            if task_id == "T4":
                slots.extend(
                    (
                        ("T4", t4_questions["swap"]),
                        ("T4", t4_questions["reenact"]),
                    )
                )
            else:
                slots.append((task_id, None))
        self.slots = tuple(slots)

    def __len__(self):
        return len(self.base_dataset) * len(self.slots)

    def __getitem__(self, index):
        base_index, slot_index = divmod(index, len(self.slots))
        task_id, question_id = self.slots[slot_index]
        rendered = self.sampler.render(
            self.base_dataset[base_index],
            task_id,
            self.split,
            self.epoch,
            force_question=question_id,
        )
        if task_id == "T4":
            _, separator, candidate_set_id = rendered["variant_id"].partition("::")
            if not separator or not candidate_set_id:
                raise ValueError("T4 variant_id does not identify its candidate set")
            rendered["question_pair_id"] = (
                f"T4::{self.split}::{candidate_set_id}"
            )
        else:
            rendered["question_pair_id"] = ""
        return rendered

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _plain_loader(cfg: dict, dataset, distributed: bool) -> DataLoader:
    workers = int(cfg["train"].get("num_workers", 4))
    kwargs = {
        "batch_size": int(cfg["train"].get("per_gpu_batch", 32)),
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=False,
            prefetch_factor=int(cfg["train"].get("prefetch_factor", 4)),
            multiprocessing_context="spawn",
        )
    if distributed:
        kwargs["sampler"] = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, **kwargs)


def condition_tasks(domain: str, condition: str) -> tuple[str, ...]:
    if condition == "fixed_h1":
        return ("T1",)
    if domain == "dfd":
        return ("T1", "T2", "T3")
    return TASKS


def build_condition_loader(
    cfg: dict,
    language_bank,
    *,
    domain: str,
    condition: str,
    distributed: bool,
):
    """Condition hook intentionally limited to Task 5's three suites."""
    spec = CONDITION_SPECS[condition]
    if spec.fixed_h1:
        dataset = FixedH1Dataset(build_eval_dataset(cfg, domain), language_bank)
        return _plain_loader(cfg, dataset, distributed)
    ordinary_loader = build_semantic_eval_loader(
        cfg,
        language_bank,
        domain=domain,
        distributed=False,
        split=spec.language_split,
        tasks=condition_tasks(domain, condition),
    )
    return _plain_loader(
        cfg,
        PairedT4EvaluationDataset(ordinary_loader.dataset),
        distributed,
    )


def _load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("semantic checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("semantic checkpoint lacks model_state_dict")
    return checkpoint


def _move_batch(batch: dict, device: torch.device):
    images = batch["image"].to(device, non_blocking=True)
    question_ids = batch["question_text_id"].to(device, non_blocking=True)
    candidate_ids = batch["candidate_text_ids"].to(device, non_blocking=True)
    candidate_mask = batch["candidate_mask"].to(device, non_blocking=True)
    return images, question_ids, candidate_ids, candidate_mask


@torch.no_grad()
def infer_condition_rows(
    model,
    loader,
    text_lookup,
    device: torch.device,
    *,
    seed: int,
    domain: str,
    condition: str,
) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    for batch in loader:
        images, question_ids, candidate_ids, candidate_mask = _move_batch(
            batch, device
        )
        questions, candidates = text_lookup.lookup(
            question_ids, candidate_ids, candidate_mask
        )
        logits = model(images, questions, candidates, candidate_mask)
        probabilities = torch.softmax(logits.float(), dim=1)
        rows.extend(
            frame_semantic_rows(
                batch,
                probabilities.cpu(),
                {"seed": seed, "domain": domain, "condition": condition},
            )
        )
    return rows


def _npz_array(values: list[object]) -> np.ndarray:
    if all(isinstance(value, (bool, np.bool_)) for value in values):
        return np.asarray(values, dtype=bool)
    if all(isinstance(value, (int, np.integer)) for value in values):
        return np.asarray(values, dtype=np.int64)
    if all(isinstance(value, (int, float, np.number)) for value in values):
        return np.asarray(values, dtype=np.float64)
    return np.asarray(["" if value is None else str(value) for value in values])


def write_frame_npz(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    arrays = {
        column: _npz_array([row.get(column) for row in rows])
        for column in columns
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(
    rows: list[dict], *, output_dir: str | Path, condition: str, domain: str
) -> None:
    root = Path(output_dir) / condition
    root.mkdir(parents=True, exist_ok=True)
    videos = aggregate_semantic_videos(rows)
    metrics = compute_task_metrics(videos)

    write_frame_npz(rows, root / f"{domain}_frames.npz")
    video_path = root / f"{domain}_videos.csv"
    video_tmp = video_path.with_suffix(video_path.suffix + ".tmp")
    videos.to_csv(video_tmp, index=False)
    os.replace(video_tmp, video_path)

    metric_path = root / f"{domain}_metrics.json"
    metric_tmp = metric_path.with_suffix(metric_path.suffix + ".tmp")
    records = [
        {key: _json_value(value) for key, value in row.items()}
        for row in metrics.to_dict("records")
    ]
    metric_tmp.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    os.replace(metric_tmp, metric_path)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["seed"] = args.seed
    if args.batch_size is not None:
        cfg["train"]["per_gpu_batch"] = args.batch_size

    local_rank = init_ddp()
    distributed = dist.is_available() and dist.is_initialized()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    try:
        manifest_path = _repo_path(cfg["language"]["manifest"])
        language_bank = load_language_bank(manifest_path)
        checkpoint = _load_checkpoint(_repo_path(args.checkpoint))
        validate_checkpoint_manifest(
            checkpoint, file_sha256(manifest_path), args.conditions
        )
        checkpoint_seed = checkpoint.get("seed")
        if checkpoint_seed is not None and int(checkpoint_seed) != args.seed:
            raise ValueError(
                f"checkpoint seed {checkpoint_seed} does not match --seed {args.seed}"
            )

        model = build_model(cfg["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        # A lookup is built lazily per condition and contains only the split
        # that condition is authorized to encode.
        text_lookups = {}
        for condition in args.conditions:
            text_lookups[condition] = EncodedTextLookup.build(
                model,
                language_bank,
                allowed_splits=required_text_splits(condition),
                device=device,
            )
            for domain in args.domains:
                loader = build_condition_loader(
                    cfg,
                    language_bank,
                    domain=domain,
                    condition=condition,
                    distributed=distributed,
                )
                local_rows = infer_condition_rows(
                    model,
                    loader,
                    text_lookups[condition],
                    device,
                    seed=args.seed,
                    domain=domain,
                    condition=condition,
                )
                gathered = gather_rows_across_ranks(local_rows)
                if is_main_process():
                    rows = deduplicate_distributed_rows(gathered)
                    write_outputs(
                        rows,
                        output_dir=args.output_dir,
                        condition=condition,
                        domain=domain,
                    )
                barrier()
        return 0
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
