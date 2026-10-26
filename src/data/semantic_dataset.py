"""Deterministic semantic-task rendering over H1 frame datasets."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from data.constants import FACE_SWAP_METHODS, REENACTMENT_METHODS
from data.semantic_language import LanguageBank, SEMANTIC_TO_ID, TextEntry


TASKS = ("T1", "T2", "T3", "T4", "T5")
MAX_CANDIDATES = 3


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def method_family(method_id: str, label: int) -> str:
    """Map H1 method labels to their supervised coarse family."""
    if int(label) == 0:
        return "camera"
    if method_id in FACE_SWAP_METHODS:
        return "swap"
    if method_id in REENACTMENT_METHODS:
        return "reenact"
    return "unknown"


def _pack_candidates(
    options: Sequence[TextEntry], target_semantic: str, order: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ordered = [options[index] for index in order]
    target_index = next(
        index
        for index, option in enumerate(ordered)
        if option.semantic == target_semantic
    )
    text_ids = torch.full((MAX_CANDIDATES,), -1, dtype=torch.long)
    semantic_ids = torch.full((MAX_CANDIDATES,), -1, dtype=torch.long)
    mask = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
    for index, option in enumerate(ordered):
        text_ids[index] = option.text_id
        semantic_ids[index] = SEMANTIC_TO_ID[option.semantic]
        mask[index] = True
    return text_ids, semantic_ids, mask, target_index


class SemanticTemplateSampler:
    """Render language deterministically from a sample identity and epoch."""

    def __init__(self, language_bank: LanguageBank, seed: int = 42):
        self.language_bank = language_bank
        self.seed = int(seed)

    def render(
        self,
        sample: dict,
        task_id: str | None,
        split: str,
        epoch: int,
        permutation: tuple[int, ...] | None = None,
        *,
        force_question: str | None = None,
        force_candidate_set: str | None = None,
    ) -> dict:
        sample_id = sample["sample_id"]
        epoch = int(epoch)
        if task_id is None:
            task_id = TASKS[
                _stable_u64(self.seed, epoch, sample_id) % len(TASKS)
            ]
        if task_id not in TASKS:
            raise ValueError(f"unknown semantic task: {task_id}")

        language = self.language_bank.task(task_id, split)
        rng = random.Random(_stable_u64(self.seed, epoch, sample_id, task_id))
        question = self._choose_named(
            language.questions, force_question, rng, "question"
        )
        candidate_set = self._choose_named(
            language.candidate_sets, force_candidate_set, rng, "candidate set"
        )

        label = int(sample["label"])
        family = method_family(sample["method_id"], label)
        target_semantic = self._target_semantic(
            task_id, label, family, question.yes_means
        )

        candidate_count = len(candidate_set.options)
        if permutation is None:
            order = tuple(rng.sample(range(candidate_count), candidate_count))
        else:
            order = tuple(permutation)
            if sorted(order) != list(range(candidate_count)):
                raise ValueError(
                    f"invalid permutation for {candidate_count} candidates: {order}"
                )
        text_ids, semantic_ids, mask, target_index = _pack_candidates(
            candidate_set.options, target_semantic, order
        )
        ordered_options = [candidate_set.options[index] for index in order]
        padding = MAX_CANDIDATES - candidate_count

        rendered = dict(sample)
        rendered.update(
            {
                "task_id": task_id,
                "task_index": torch.tensor(TASKS.index(task_id), dtype=torch.long),
                "question_text_id": torch.tensor(
                    question.text_id, dtype=torch.long
                ),
                "question_id": question.name,
                "question_text": question.text,
                "question_yes_means": torch.tensor(
                    SEMANTIC_TO_ID.get(question.yes_means, -1), dtype=torch.long
                ),
                "question_yes_means_name": question.yes_means or "",
                "candidate_text_ids": text_ids,
                "candidate_semantic_ids": semantic_ids,
                "candidate_mask": mask,
                "candidate_ids": tuple(
                    option.name for option in ordered_options
                )
                + ("",) * padding,
                "candidate_texts": tuple(
                    option.text for option in ordered_options
                )
                + ("",) * padding,
                "candidate_semantics": tuple(
                    option.semantic for option in ordered_options
                )
                + ("",) * padding,
                "target_index": torch.tensor(target_index, dtype=torch.long),
                "target_semantic": target_semantic,
                "method_family": family,
                "variant_id": f"{question.name}::{candidate_set.name}",
                "render_id": format(
                    _stable_u64(
                        self.seed,
                        epoch,
                        sample_id,
                        task_id,
                        question.name,
                        candidate_set.name,
                        order,
                    ),
                    "016x",
                ),
            }
        )
        return rendered

    @staticmethod
    def _choose_named(items, forced_name, rng, kind):
        if not items:
            raise ValueError(f"semantic language has no {kind}s")
        if forced_name is None:
            return items[rng.randrange(len(items))]
        for item in items:
            if item.name == forced_name:
                return item
        raise ValueError(f"unknown {kind}: {forced_name}")

    @staticmethod
    def _target_semantic(
        task_id: str, label: int, family: str, yes_means: str | None
    ) -> str:
        visual_truth = "real" if label == 0 else "fake"
        if task_id in {"T1", "T3"}:
            return visual_truth
        if task_id == "T2":
            return "yes" if yes_means == visual_truth else "no"
        if task_id == "T4":
            if family == "unknown":
                raise ValueError("unknown method family is invalid for T4")
            return "yes" if yes_means == family else "no"
        if family == "unknown":
            raise ValueError("unknown method family is invalid for T5")
        return family


class SemanticDataset(Dataset):
    """Wrap an H1 dataset with dynamic train or deterministic suite language."""

    def __init__(
        self,
        base_dataset,
        sampler: SemanticTemplateSampler,
        mode: str = "train",
        tasks: tuple[str, ...] = TASKS,
        split: str = "train",
    ):
        if mode not in {"train", "suite"}:
            raise ValueError("mode must be 'train' or 'suite'")
        if not tasks or any(task not in TASKS for task in tasks):
            raise ValueError(f"invalid semantic tasks: {tasks}")
        self.base_dataset = base_dataset
        self.sampler = sampler
        self.mode = mode
        self.tasks = tuple(tasks)
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.base_dataset)
        return len(self.base_dataset) * len(self.tasks)

    def __getitem__(self, index: int) -> dict:
        if self.mode == "train":
            base_index, task_id = index, None
        else:
            base_index = index // len(self.tasks)
            task_id = self.tasks[index % len(self.tasks)]
        sample = self.base_dataset[base_index]
        return self.sampler.render(sample, task_id, self.split, self.epoch)
