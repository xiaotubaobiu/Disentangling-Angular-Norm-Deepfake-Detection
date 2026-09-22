from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from data.semantic_dataset import (
    TASKS,
    SemanticDataset,
    SemanticTemplateSampler,
    method_family,
)
from data.semantic_language import SEMANTIC_TO_ID, load_language_bank


MANIFEST = Path(
    "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
)


def _sample(label, method_id, sample_id="sample-1"):
    return {
        "image": torch.zeros(3, 8, 8),
        "label": torch.tensor(label),
        "method_id": method_id,
        "video_id": "001",
        "video_key": f"{method_id}::001",
        "image_path": "/tmp/frame.png",
        "sample_id": sample_id,
    }


class _BaseDataset(Dataset):
    def __init__(self, size=16, method_id="fomm"):
        self.size = size
        self.method_id = method_id

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return _sample(1, self.method_id, sample_id=f"sample-{index}")


@pytest.fixture
def bank():
    return load_language_bank(MANIFEST)


def test_method_family_never_guesses_unknown_fake():
    assert method_family("real", 0) == "camera"
    assert method_family("simswap", 1) == "swap"
    assert method_family("sadtalker", 1) == "reenact"
    assert method_family("DFD_fake", 1) == "unknown"


def test_t2_target_flips_with_question_polarity(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    real = _sample(0, "real")
    positive = sampler.render(
        real,
        "T2",
        "train",
        0,
        force_question="t2_real_q1_train",
        force_candidate_set="answer_train_short",
        permutation=(0, 1),
    )
    negative = sampler.render(
        real,
        "T2",
        "train",
        0,
        force_question="t2_fake_q1_train",
        force_candidate_set="answer_train_short",
        permutation=(0, 1),
    )
    assert positive["target_semantic"] == "yes"
    assert negative["target_semantic"] == "no"
    assert positive["question_yes_means"].item() == SEMANTIC_TO_ID["real"]
    assert negative["question_yes_means"].item() == SEMANTIC_TO_ID["fake"]


def test_t4_and_t5_truth_tables(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    swap = _sample(1, "simswap")
    assert sampler.render(
        swap, "T4", "train", 0, force_question="t4_swap_q_train"
    )["target_semantic"] == "yes"
    assert sampler.render(
        swap, "T4", "train", 0, force_question="t4_reenact_q_train"
    )["target_semantic"] == "no"
    t5 = sampler.render(swap, "T5", "train", 0)
    assert t5["target_semantic"] == "swap"
    assert t5["candidate_mask"].tolist() == [True, True, True]


def test_binary_candidates_are_padded_without_targeting_padding(bank):
    row = SemanticTemplateSampler(bank, seed=42).render(
        _sample(1, "fsgan"), "T1", "train", 0
    )
    assert row["candidate_mask"].tolist().count(True) == 2
    assert row["candidate_text_ids"].shape == (3,)
    assert row["candidate_semantic_ids"].shape == (3,)
    assert row["candidate_text_ids"][2].item() == -1
    assert row["target_index"].item() < 2


def test_candidate_permutation_moves_text_semantics_and_target_together(bank):
    row = SemanticTemplateSampler(bank, seed=42).render(
        _sample(1, "simswap"),
        "T5",
        "train",
        0,
        force_candidate_set="t5_c1_train",
        permutation=(2, 0, 1),
    )
    assert row["candidate_semantic_ids"].tolist() == [
        SEMANTIC_TO_ID["reenact"],
        SEMANTIC_TO_ID["camera"],
        SEMANTIC_TO_ID["swap"],
    ]
    assert row["target_index"].item() == 2


def test_hashing_is_repeatable_and_epoch_sensitive(bank):
    sampler_a = SemanticTemplateSampler(bank, seed=2048)
    sampler_b = SemanticTemplateSampler(bank, seed=2048)
    item = _sample(1, "fomm", sample_id="stable-id")
    first = sampler_a.render(item, None, "train", 3)
    repeated = sampler_b.render(item, None, "train", 3)
    changed = sampler_a.render(item, None, "train", 4)
    assert first["render_id"] == repeated["render_id"]
    torch.testing.assert_close(
        first["candidate_text_ids"], repeated["candidate_text_ids"]
    )
    assert first["render_id"] != changed["render_id"]


def test_language_render_is_identical_with_zero_or_two_workers(bank):
    zero_worker = SemanticDataset(
        _BaseDataset(), SemanticTemplateSampler(bank, seed=42), mode="train"
    )
    two_workers = SemanticDataset(
        _BaseDataset(), SemanticTemplateSampler(bank, seed=42), mode="train"
    )
    zero_worker.set_epoch(6)
    two_workers.set_epoch(6)
    ids_zero = [
        render_id
        for batch in DataLoader(zero_worker, batch_size=8, num_workers=0)
        for render_id in batch["render_id"]
    ]
    ids_two = [
        render_id
        for batch in DataLoader(two_workers, batch_size=8, num_workers=2)
        for render_id in batch["render_id"]
    ]
    assert ids_zero == ids_two


def test_suite_exposes_canonical_task_id_and_numeric_tensor(bank):
    dataset = SemanticDataset(
        _BaseDataset(size=1),
        SemanticTemplateSampler(bank, seed=42),
        mode="suite",
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(TASKS))))
    assert batch["task_id"] == list(TASKS)
    torch.testing.assert_close(batch["task_index"], torch.arange(len(TASKS)))
    assert batch["question_text_id"].shape == (5,)
    assert batch["target_index"].shape == (5,)


def test_unknown_fake_supports_t1_to_t3_but_rejects_t4_t5(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    dfd = _sample(1, "DFD_fake")
    for task in ("T1", "T2", "T3"):
        sampler.render(dfd, task, "heldout", 0)
    for task in ("T4", "T5"):
        with pytest.raises(ValueError, match="unknown method family"):
            sampler.render(dfd, task, "heldout", 0)


def test_semantic_eval_loader_excludes_dfd_family_tasks(monkeypatch, bank):
    from data import builders

    monkeypatch.setattr(
        builders, "build_eval_dataset", lambda cfg, domain="ffpp": _BaseDataset()
    )
    cfg = {"dataset": {"root": "/unused"}, "train": {"num_workers": 0}}
    loader = builders.build_semantic_eval_loader(
        cfg, bank, domain="dfd", distributed=False
    )
    assert loader.dataset.tasks == ("T1", "T2", "T3")


def test_invalid_permutation_is_rejected(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    with pytest.raises(ValueError, match="permutation"):
        sampler.render(
            _sample(1, "fsgan"), "T1", "train", 0, permutation=(0, 0)
        )
