"""Distributed, leakage-safe evaluation for E2 Semantic Visual-Jev."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config_io import load_config
from data.builders import build_eval_dataset, build_semantic_eval_loader
from data.semantic_dataset import (
    TASKS,
    SemanticDataset,
    SemanticTemplateSampler,
    method_family,
)
from data.semantic_language import SEMANTIC_TO_ID, load_language_bank
from engine.ddp import barrier, init_ddp, is_main_process
from evaluation.sample_eval import gather_rows_across_ranks
from evaluation.semantic_controls import (
    build_counterfactual_requests,
    build_meaningless_candidate_request,
    build_meaningless_question_request,
    control_metric_rows,
    paired_control_rows,
    permutation_audit,
    permutation_metric_rows,
    select_global_audit_indices,
)
from evaluation.semantic_eval import (
    GROUP_KEYS,
    PROBABILITY_COLUMNS,
    aggregate_semantic_videos,
    compute_task_metrics,
    deduplicate_distributed_rows,
    frame_semantic_rows,
    map_candidate_probabilities,
)
from models.clip_semantic_jev import EncodedTextLookup
from models.factory import build_model


ConditionSpec = namedtuple(
    "ConditionSpec", ("text_splits", "language_split", "mode", "fixed_h1"),
    defaults=("ordinary", False),
)


# Task 6 extends this registry with controls.  Keeping construction and
# inference behind these hooks avoids embedding control behavior in Task 5.
CONDITION_SPECS = {
    "fixed_h1": ConditionSpec(("fixed",), "fixed", fixed_h1=True),
    "seen": ConditionSpec(("train",), "train"),
    "heldout": ConditionSpec(("heldout",), "heldout"),
    "permutation": ConditionSpec(("heldout",), "heldout", "permutation"),
    "counterfactual": ConditionSpec(("heldout",), "heldout", "counterfactual"),
    "meaningless_candidates": ConditionSpec(
        ("heldout", "control"), "heldout", "meaningless_candidates"
    ),
    "meaningless_questions": ConditionSpec(
        ("heldout", "control"), "heldout", "meaningless_questions"
    ),
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


def build_condition_text_lookup(model, language_bank, condition, device):
    """Encode only text entries reachable by one explicit condition."""
    splits = frozenset(required_text_splits(condition))
    control_kind = {
        "meaningless_candidates": "candidate",
        "meaningless_questions": "question",
    }.get(condition)
    entries = tuple(
        entry
        for entry in language_bank.text_entries
        if entry.split in splits
        and (
            entry.split != "control"
            or control_kind is None
            or entry.kind == control_kind
        )
    )
    scoped_bank = SimpleNamespace(text_entries=entries)
    return EncodedTextLookup.build(
        model,
        scoped_bank,
        allowed_splits=tuple(splits),
        device=device,
    )


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


class CounterfactualEvaluationDataset(Dataset):
    """Expand each physical sample into exact T2/T4 paired requests."""

    def __init__(self, base_dataset, language_bank, *, tasks, split, seed):
        self.base_dataset = base_dataset
        self.language_bank = language_bank
        self.split = split
        self.seed = int(seed)
        self.tasks = tuple(task for task in tasks if task in {"T2", "T4"})
        if not self.tasks:
            raise ValueError("counterfactual evaluation selected no paired tasks")
        self.slots = tuple(
            (task_id, pair_index)
            for task_id in self.tasks
            for pair_index in range(2)
        )

    def __len__(self):
        return len(self.base_dataset) * len(self.slots)

    def __getitem__(self, index):
        base_index, slot_index = divmod(index, len(self.slots))
        task_id, pair_index = self.slots[slot_index]
        requests = build_counterfactual_requests(
            task_id,
            self.split,
            self.base_dataset[base_index],
            self.language_bank,
            seed=self.seed,
        )
        return requests[pair_index]


class PairedMeaninglessEvaluationDataset(Dataset):
    """Emit a control request plus its exact hidden meaningful counterpart."""

    _PAIRED_FIELDS = (
        "question_text_id",
        "candidate_text_ids",
        "candidate_semantic_ids",
        "candidate_mask",
        "target_index",
        "question_id",
        "variant_id",
        "render_id",
    )

    def __init__(self, meaningful_dataset, language_bank, *, control):
        if control not in {"meaningless_candidates", "meaningless_questions"}:
            raise ValueError(f"unknown meaningless control: {control}")
        self.meaningful_dataset = meaningful_dataset
        self.language_bank = language_bank
        self.control = control
        permitted = (
            {"T2", "T4"}
            if control == "meaningless_questions"
            else set(TASKS)
        )
        self.slot_indices = tuple(
            index
            for index, (task_id, _question_id) in enumerate(
                meaningful_dataset.slots
            )
            if task_id in permitted
        )
        if not self.slot_indices:
            raise ValueError(f"{control} selected no eligible semantic tasks")

    def __len__(self):
        return len(self.meaningful_dataset.base_dataset) * len(self.slot_indices)

    def __getitem__(self, index):
        base_index, selected_slot = divmod(index, len(self.slot_indices))
        meaningful_index = (
            base_index * len(self.meaningful_dataset.slots)
            + self.slot_indices[selected_slot]
        )
        meaningful = self.meaningful_dataset[meaningful_index]
        sample_id = str(meaningful["sample_id"])
        if self.control == "meaningless_candidates":
            control = build_meaningless_candidate_request(
                meaningful, sample_id, self.language_bank
            )
        else:
            control = build_meaningless_question_request(
                meaningful, sample_id, self.language_bank
            )
        for field in self._PAIRED_FIELDS:
            value = meaningful[field]
            control[f"meaningful_{field}"] = (
                value.clone() if isinstance(value, torch.Tensor) else value
            )
        return control


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


def _base_sample_ids(dataset) -> list[str]:
    records = getattr(dataset, "records", None)
    if records is not None:
        return [str(record.sample_id) for record in records]
    return [str(dataset[index]["sample_id"]) for index in range(len(dataset))]


def build_condition_loader(
    cfg: dict,
    language_bank,
    *,
    domain: str,
    condition: str,
    distributed: bool,
):
    """Build an explicit condition without changing Task 5 primitives."""
    spec = CONDITION_SPECS[condition]
    if spec.fixed_h1:
        dataset = FixedH1Dataset(build_eval_dataset(cfg, domain), language_bank)
        return _plain_loader(cfg, dataset, distributed)
    if spec.mode != "ordinary":
        base_dataset = build_eval_dataset(cfg, domain)
        tasks = condition_tasks(domain, condition)
        seed = int(cfg["train"].get("seed", 42))
        if spec.mode == "permutation":
            selected = select_global_audit_indices(
                _base_sample_ids(base_dataset), limit=4096
            )
            audited_base = Subset(base_dataset, selected)
            dataset = SemanticDataset(
                audited_base,
                SemanticTemplateSampler(language_bank, seed=seed),
                mode="suite",
                tasks=tasks,
                split=spec.language_split,
            )
            # A global subset is selected before rank sharding. Unlike padded
            # DistributedSampler, this visits each audited render exactly once.
            if distributed:
                dataset = Subset(dataset, range(dist.get_rank(), len(dataset), dist.get_world_size()))
            return _plain_loader(cfg, dataset, distributed=False)
        elif spec.mode == "counterfactual":
            dataset = CounterfactualEvaluationDataset(
                base_dataset,
                language_bank,
                tasks=tasks,
                split=spec.language_split,
                seed=seed,
            )
        else:
            meaningful = SemanticDataset(
                base_dataset,
                SemanticTemplateSampler(language_bank, seed=seed),
                mode="suite",
                tasks=tasks,
                split=spec.language_split,
            )
            paired_meaningful = PairedT4EvaluationDataset(meaningful)
            dataset = PairedMeaninglessEvaluationDataset(
                paired_meaningful,
                language_bank,
                control=spec.mode,
            )
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


def _batch_list(value, size):
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().tolist()
    elif isinstance(value, (str, bytes)):
        result = [value] * size
    else:
        result = list(value)
    if len(result) != size:
        raise ValueError(f"batch field has {len(result)} values; expected {size}")
    return result


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
        if "control_pair_id" in batch or "hidden_visual_truth" in batch:
            current = rows[-probabilities.shape[0] :]
            for field in (
                "control_pair_id",
                "hidden_visual_truth",
                "hidden_method_family",
                "hidden_question_polarity",
            ):
                if field in batch:
                    for row, value in zip(
                        current, _batch_list(batch[field], len(current))
                    ):
                        row[field] = str(value)
    return rows


@torch.no_grad()
def infer_paired_control_rows(
    model,
    loader,
    text_lookup,
    device: torch.device,
    *,
    seed: int,
    domain: str,
    condition: str,
) -> list[dict]:
    """Score paired meaningful/control text using one visual feature cache."""
    model.eval()
    rows = []
    for batch in loader:
        images, question_ids, candidate_ids, candidate_mask = _move_batch(
            batch, device
        )
        image_features = model.extract_features(images)
        control_questions, control_candidates = text_lookup.lookup(
            question_ids, candidate_ids, candidate_mask
        )
        meaningful_question_ids = batch["meaningful_question_text_id"].to(device)
        meaningful_candidate_ids = batch["meaningful_candidate_text_ids"].to(device)
        meaningful_mask = batch["meaningful_candidate_mask"].to(device)
        meaningful_questions, meaningful_candidates = text_lookup.lookup(
            meaningful_question_ids, meaningful_candidate_ids, meaningful_mask
        )
        control_logits = model.classify_features(
            image_features,
            control_questions,
            control_candidates,
            candidate_mask,
        )
        meaningful_logits = model.classify_features(
            image_features,
            meaningful_questions,
            meaningful_candidates,
            meaningful_mask,
        )
        control_position_probabilities = torch.softmax(
            control_logits.float(), dim=1
        )
        meaningful_position_probabilities = torch.softmax(
            meaningful_logits.float(), dim=1
        )
        control_canonical = map_candidate_probabilities(
            control_position_probabilities,
            batch["candidate_semantic_ids"].to(device),
            candidate_mask,
        ).cpu()
        meaningful_canonical = map_candidate_probabilities(
            meaningful_position_probabilities,
            batch["meaningful_candidate_semantic_ids"].to(device),
            meaningful_mask,
        ).cpu()
        evaluation_batch = dict(batch)
        # Original question/variant identity remains evaluation metadata. The
        # actual model text IDs above still point only to the control text.
        for field in ("question_id", "variant_id", "render_id"):
            evaluation_batch[field] = batch[f"meaningful_{field}"]
        frame_rows = frame_semantic_rows(
            evaluation_batch,
            control_position_probabilities.cpu(),
            {"seed": seed, "domain": domain, "condition": condition},
        )
        pair_ids = _batch_list(batch["control_pair_id"], len(frame_rows))
        metadata = []
        for row, pair_id in zip(frame_rows, pair_ids):
            row["control_pair_id"] = str(pair_id)
            metadata.append(
                {
                    **row,
                    "control_pair_id": str(pair_id),
                }
            )
        paired = paired_control_rows(
            metadata, meaningful_canonical, control_canonical
        )
        meaningful_batch = dict(evaluation_batch)
        for field in PairedMeaninglessEvaluationDataset._PAIRED_FIELDS:
            meaningful_batch[field] = batch[f"meaningful_{field}"]
        meaningful_rows = frame_semantic_rows(
            meaningful_batch, meaningful_position_probabilities.cpu(),
            {"seed": seed, "domain": domain, "condition": condition},
        )
        for row, meaningful_row in zip(paired, meaningful_rows):
            for column in (*PROBABILITY_COLUMNS, "prob_fake"):
                row[f"meaningful_{column}"] = meaningful_row[column]
        for row, question_id in zip(paired, _batch_list(batch["question_id"], len(paired))):
            row["control_question_id"] = str(question_id)
        rows.extend(paired)
    return rows


@torch.no_grad()
def infer_permutation_metrics(
    model,
    loader,
    text_lookup,
    device: torch.device,
    *,
    seed: int,
    domain: str,
) -> list[dict]:
    """Run exhaustive orders with one visual extraction per audited batch."""
    maximum_error = 0.0
    consistent_count = 0
    comparison_count = 0
    model.eval()
    for batch in loader:
        images, question_ids, candidate_ids, candidate_mask = _move_batch(
            batch, device
        )
        image_features = model.extract_features(images)
        questions, candidates = text_lookup.lookup(
            question_ids, candidate_ids, candidate_mask
        )
        result = permutation_audit(
            model,
            image_features,
            questions,
            candidates,
            candidate_mask,
            reduce_across_ranks=False,
        )
        maximum_error = max(maximum_error, float(result["max_probability_error"]))
        consistent_count += int(result["consistent_count"])
        comparison_count += int(result["comparison_count"])
    # Only one collective per aggregate, after potentially uneven rank loops.
    if dist.is_available() and dist.is_initialized():
        error = torch.tensor(maximum_error, dtype=torch.float64, device=device)
        counts = torch.tensor([consistent_count, comparison_count], dtype=torch.long, device=device)
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        maximum_error = error.item()
        consistent_count, comparison_count = counts.tolist()
    aggregate = {
        "max_probability_error": maximum_error,
        "prediction_consistency": (
            consistent_count / comparison_count
            if comparison_count
            else float("nan")
        ),
        "consistent_count": consistent_count,
        "comparison_count": comparison_count,
    }
    return permutation_metric_rows(aggregate, seed=seed, domain=domain)


def counterfactual_control_rows(rows: list[dict]) -> list[dict]:
    """Compute paired consistency/ordering after averaging each video polarity."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    physical_keys = ["seed", "domain", "condition", "task_id", "sample_id", "video_key", "question_pair_id"]
    for identity, pair in frame.groupby(physical_keys, sort=False, dropna=False):
        expected = {"real", "fake"} if identity[3] == "T2" else {"swap", "reenact"}
        if len(pair) != 2 or set(pair.question_yes_means) != expected:
            raise ValueError("counterfactual physical question pair is incomplete or ambiguous")
    # Physical pair IDs remain in frame output. Video pairing pools all frames
    # of each question polarity within the same method-qualified video.
    video_keys = ["seed", "domain", "condition", "task_id", "video_key", "label", "method_family", "question_yes_means"]
    frame = frame.groupby(video_keys, sort=False, dropna=False)[["p_yes", "prob_fake"]].mean().reset_index()
    keys = [
        "seed",
        "domain",
        "condition",
        "task_id",
        "video_key",
    ]
    output = []
    for identity, pair in frame.groupby(keys, sort=False, dropna=False):
        if len(pair) != 2 or pair.question_yes_means.nunique() != 2:
            raise ValueError("counterfactual pair ID must identify exactly two polarities")
        seed, domain, condition, task_id, video_key = identity
        base = {
            "seed": int(seed),
            "domain": str(domain),
            "condition": str(condition),
            "task_id": str(task_id),
            "video_key": str(video_key),
            "pairing_key": "::".join(str(value) for value in identity),
            "count": 1,
        }
        if task_id == "T2":
            literal_answers = pair.p_yes.to_numpy(dtype=float) >= 0.5
            implied_fake = pair.prob_fake.to_numpy(dtype=float) >= 0.5
            value = float(
                literal_answers[0] != literal_answers[1]
                and implied_fake[0] == implied_fake[1]
            )
            output.append(
                {
                    **base,
                    "metric": "polarity_consistency",
                    "value": value,
                }
            )
        elif task_id == "T4" and int(pair.label.iloc[0]) == 1:
            family = str(pair.method_family.iloc[0])
            matching = pair.loc[pair.question_yes_means == family]
            counterfactual = pair.loc[pair.question_yes_means != family]
            if len(matching) != 1 or len(counterfactual) != 1:
                raise ValueError("T4 pair does not contain one true-family question")
            meaningful_score = float(matching.p_yes.iloc[0])
            control_score = float(counterfactual.p_yes.iloc[0])
            output.append(
                {
                    **base,
                    "metric": "true_family_ordering",
                    "value": float(meaningful_score > control_score),
                    "meaningful_score": meaningful_score,
                    "control_score": control_score,
                    "drop": meaningful_score - control_score,
                }
            )
    return output


def evaluate_condition(model, loader, text_lookup, device, *, seed, domain, condition):
    """Dispatch all public conditions through their actual evaluation path."""
    context = dict(seed=seed, domain=domain)
    if condition == "permutation":
        return infer_permutation_metrics(model, loader, text_lookup, device, **context)
    infer = infer_paired_control_rows if condition.startswith("meaningless_") else infer_condition_rows
    return infer(model, loader, text_lookup, device, condition=condition, **context)


def paired_control_videos(rows):
    """Keep canonical video aggregation and join counterparts by exact keys."""
    control = aggregate_semantic_videos(rows)
    meaningful_rows = []
    for source in rows:
        row = dict(source)
        for column in (*PROBABILITY_COLUMNS, "prob_fake"):
            row[column] = source[f"meaningful_{column}"]
        meaningful_rows.append(row)
    meaningful = aggregate_semantic_videos(meaningful_rows)
    values = [*PROBABILITY_COLUMNS, "prob_fake", "num_frames"]
    joined = control.merge(
        meaningful[[*GROUP_KEYS, *values]].rename(columns={column: f"meaningful_{column}" for column in values}),
        on=list(GROUP_KEYS), how="outer", validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("meaningful/control video keys do not match exactly")
    if not joined.num_frames.eq(joined.meaningful_num_frames).all():
        raise ValueError("meaningful/control video frame counts differ")
    joined = joined.drop(columns="_merge")
    joined["pairing_key"] = joined[list(GROUP_KEYS)].astype(str).agg("::".join, axis=1)
    return joined


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
    if condition == "permutation":
        _write_metrics(rows, root / f"{domain}_metrics.json")
        return
    aggregation_rows = rows
    if condition == "counterfactual":
        # Physical IDs must not split a video into single-frame groups.
        aggregation_rows = [dict(row, question_pair_id=f"{row['task_id']}::heldout::{row['variant_id'].split('::')[1]}") for row in rows]
    videos = aggregate_semantic_videos(aggregation_rows)
    metrics = compute_task_metrics(videos).to_dict("records")
    for row in metrics:
        row["pairing_key"] = "::".join(str(row[key]) for key in ("seed", "domain", "condition", "task_id", "method_id"))
    if condition.startswith("meaningless_"):
        paired = paired_control_videos(rows)
        _write_csv(paired, root / f"{domain}_paired_videos.csv")
        metrics.extend(control_metric_rows(paired.to_dict("records")))
    elif condition == "counterfactual":
        controls = counterfactual_control_rows(rows)
        _write_csv(pd.DataFrame(controls), root / f"{domain}_controls.csv")
        if controls:
            keys = ["seed", "domain", "condition", "task_id", "metric"]
            for identity, group in pd.DataFrame(controls).groupby(keys, sort=False):
                metrics.append({**dict(zip(keys, identity)), "value": float(group.value.mean()),
                                "count": len(group), "pairing_key": "::".join(map(str, identity))})

    write_frame_npz(rows, root / f"{domain}_frames.npz")
    _write_csv(videos, root / f"{domain}_videos.csv")
    _write_metrics(metrics, root / f"{domain}_metrics.json")


def _write_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_metrics(metrics, metric_path):
    metric_tmp = metric_path.with_suffix(metric_path.suffix + ".tmp")
    records = [
        {key: _json_value(value) for key, value in row.items()}
        for row in metrics
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
            text_lookups[condition] = build_condition_text_lookup(
                model,
                language_bank,
                condition=condition,
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
                local_rows = evaluate_condition(
                    model,
                    loader,
                    text_lookups[condition],
                    device,
                    seed=args.seed,
                    domain=domain,
                    condition=condition,
                )
                gathered = local_rows if condition == "permutation" else gather_rows_across_ranks(local_rows)
                if is_main_process():
                    rows = gathered if condition == "permutation" else deduplicate_distributed_rows(gathered)
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
