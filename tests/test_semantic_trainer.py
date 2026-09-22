from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import yaml

from engine.semantic_trainer import (
    TrainableEMA,
    run_semantic_train_epoch,
    run_semantic_val_epoch,
    semantic_classification_step,
    trainable_parameters,
)


REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / (
    "experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py"
)
LOCKED_CONFIG = REPO / (
    "experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml"
)


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_semantic", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ToySemanticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip = nn.Linear(2, 2, bias=False)
        self.clip.requires_grad_(False)
        self.decision = nn.Linear(2, 1, bias=False)

    def forward(self, images, questions, candidates, candidate_mask):
        del questions
        image_scores = self.decision(images).squeeze(-1)
        candidate_scores = candidates[..., 0]
        logits = candidate_scores + image_scores[:, None]
        return logits.masked_fill(~candidate_mask, float("-inf"))


class _Lookup:
    def lookup(self, question_ids, candidate_ids, candidate_mask):
        questions = torch.stack((question_ids.float(), question_ids.float()), dim=-1)
        candidates = torch.stack(
            (candidate_ids.clamp_min(0).float(), torch.zeros_like(candidate_ids).float()),
            dim=-1,
        )
        return questions, candidates.masked_fill(~candidate_mask[..., None], 0.0)


class _Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _row(image, candidates, target):
    candidate_ids = torch.tensor(candidates + [-1] * (3 - len(candidates)))
    candidate_mask = candidate_ids >= 0
    return {
        "image": torch.tensor(image, dtype=torch.float32),
        "question_text_id": torch.tensor(1),
        "candidate_text_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "target_index": torch.tensor(target),
    }


def _assert_state_dict_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name])


@pytest.fixture
def toy_model():
    model = _ToySemanticModel()
    with torch.no_grad():
        model.decision.weight.copy_(torch.tensor([[0.25, -0.5]]))
    return model


@pytest.fixture
def batch():
    return next(
        iter(
            DataLoader(
                _Rows(
                    [
                        _row([1.0, 0.0], [2, 5], 1),
                        _row([0.0, 1.0], [4, 1, 3], 0),
                    ]
                ),
                batch_size=2,
            )
        )
    )


@pytest.fixture
def lookup():
    return _Lookup()


def test_semantic_step_is_exact_candidate_cross_entropy(toy_model, batch, lookup):
    loss, logits = semantic_classification_step(
        toy_model, batch, lookup, torch.device("cpu")
    )

    expected = F.cross_entropy(logits, batch["target_index"])

    torch.testing.assert_close(loss, expected)


def test_optimizer_scope_excludes_clip(toy_model):
    parameters = trainable_parameters(toy_model)

    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert not any(
        parameter is frozen
        for parameter in parameters
        for frozen in toy_model.clip.parameters()
    )


def test_validation_is_repeatable_and_sample_weighted(toy_model, lookup):
    loader = DataLoader(
        _Rows(
            [
                _row([1.0, 0.0], [2, 5], 1),
                _row([0.0, 1.0], [4, 1], 0),
                _row([1.0, 1.0], [3, 2, 1], 2),
            ]
        ),
        batch_size=2,
        shuffle=False,
    )
    batches = list(loader)
    expected_loss_sum = sum(
        F.cross_entropy(
            semantic_classification_step(
                toy_model, item, lookup, torch.device("cpu")
            )[1],
            item["target_index"],
            reduction="sum",
        ).item()
        for item in batches
    )

    first = run_semantic_val_epoch(
        toy_model, loader, lookup, torch.device("cpu")
    )
    second = run_semantic_val_epoch(
        toy_model, loader, lookup, torch.device("cpu")
    )

    assert first == second
    assert first["count"] == 3
    assert first["loss"] == pytest.approx(expected_loss_sum / 3)
    assert first["accuracy"] == pytest.approx(2 / 3)


def test_validation_all_reduces_loss_correct_count_and_accuracy(
    monkeypatch, toy_model, lookup
):
    import engine.semantic_trainer as trainer

    loader = DataLoader(
        _Rows([_row([1.0, 0.0], [2, 5], 1)]), batch_size=1
    )
    batches = list(loader)
    local_logits = semantic_classification_step(
        toy_model, batches[0], lookup, torch.device("cpu")
    )[1]
    local_loss = F.cross_entropy(local_logits, torch.tensor([1])).item()

    monkeypatch.setattr(trainer.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer.dist, "is_initialized", lambda: True)

    def add_remote_totals(totals):
        totals += torch.tensor([1.25, 2.0, 3.0], dtype=torch.float64)

    monkeypatch.setattr(trainer.dist, "all_reduce", add_remote_totals)

    metrics = run_semantic_val_epoch(
        toy_model, batches, lookup, torch.device("cpu")
    )

    assert metrics == {
        "loss": pytest.approx((local_loss + 1.25) / 4),
        "accuracy": pytest.approx(3 / 4),
        "count": 4,
    }


def test_train_epoch_updates_only_trainable_decision_parameters(toy_model, lookup):
    loader = DataLoader(
        _Rows(
            [
                _row([1.0, 0.0], [2, 5], 1),
                _row([0.0, 1.0], [4, 1], 0),
            ]
        ),
        batch_size=2,
    )
    frozen_before = copy.deepcopy(toy_model.clip.state_dict())
    decision_before = toy_model.decision.weight.detach().clone()
    optimizer = torch.optim.AdamW(trainable_parameters(toy_model), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    loss = run_semantic_train_epoch(
        toy_model,
        loader,
        optimizer,
        scaler,
        torch.device("cpu"),
        lookup,
        {"train": {"precision": "fp32"}},
    )

    assert loss > 0
    _assert_state_dict_equal(toy_model.clip.state_dict(), frozen_before)
    assert not torch.equal(toy_model.decision.weight, decision_before)


def test_ema_tracks_and_applies_only_trainable_parameters(toy_model):
    ema = TrainableEMA(toy_model, decay=0.5)
    original_clip = copy.deepcopy(toy_model.clip.state_dict())
    initial_decision = toy_model.decision.weight.detach().clone()
    with torch.no_grad():
        toy_model.decision.weight.add_(2.0)
    changed_decision = toy_model.decision.weight.detach().clone()

    ema.update(toy_model)
    ema.apply(toy_model)

    torch.testing.assert_close(
        toy_model.decision.weight, (initial_decision + changed_decision) / 2
    )
    _assert_state_dict_equal(toy_model.clip.state_dict(), original_clip)
    assert set(ema.state_dict()) == {"decision.weight"}

    ema.restore(toy_model)
    torch.testing.assert_close(toy_model.decision.weight, changed_decision)


def test_max_batch_flags_are_rejected_outside_smoke():
    script = _load_train_script()

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--config",
                "config.yaml",
                "--seed",
                "42",
                "--max-train-batches",
                "1",
            ]
        )

    args = script.parse_args(
        [
            "--config",
            "config.yaml",
            "--seed",
            "7",
            "--smoke",
            "--max-train-batches",
            "2",
            "--max-val-batches",
            "3",
        ]
    )
    assert (args.seed, args.max_train_batches, args.max_val_batches) == (7, 2, 3)


def test_reportable_checkpoint_path_is_deterministic_and_refuses_overwrite(
    tmp_path,
):
    script = _load_train_script()
    cfg = {"train": {"output_dir": str(tmp_path)}}

    path = script.checkpoint_path(cfg, seed=2048, smoke=False)

    assert path == tmp_path / "seed_2048" / "best_model.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing reportable checkpoint")
    with pytest.raises(FileExistsError, match="allow-overwrite"):
        script.ensure_checkpoint_writable(path, allow_overwrite=False)
    script.ensure_checkpoint_writable(path, allow_overwrite=True)


def test_epoch_is_set_on_distributed_sampler_and_semantic_dataset():
    script = _load_train_script()

    class EpochReceiver:
        def __init__(self):
            self.epoch = None

        def set_epoch(self, epoch):
            self.epoch = epoch

    loader = type(
        "Loader",
        (),
        {"sampler": EpochReceiver(), "dataset": EpochReceiver()},
    )()

    script.set_training_epoch(loader, 9)

    assert loader.sampler.epoch == 9
    assert loader.dataset.epoch == 9


def test_training_lookup_encodes_only_train_and_fixed(monkeypatch):
    script = _load_train_script()

    class FakeLookup:
        @classmethod
        def build(cls, model, bank, allowed_splits, device):
            del model, bank, device
            return tuple(allowed_splits)

    monkeypatch.setattr(script, "EncodedTextLookup", FakeLookup)

    allowed = script.build_training_text_lookup(
        object(), object(), torch.device("cpu")
    )

    assert allowed == ("train", "fixed")


def test_training_builds_only_train_and_seen_language_validation_loaders(
    monkeypatch,
):
    script = _load_train_script()
    calls = []

    def train_builder(cfg, bank, distributed):
        calls.append(("train", cfg, bank, distributed))
        return "train-loader"

    def val_builder(cfg, bank, distributed, split):
        calls.append(("val", cfg, bank, distributed, split))
        return "val-loader"

    monkeypatch.setattr(script, "build_semantic_train_loader", train_builder)
    monkeypatch.setattr(script, "build_semantic_val_loader", val_builder)

    loaders = script.build_training_loaders("cfg", "bank", distributed=True)

    assert loaders == ("train-loader", "val-loader")
    assert calls == [
        ("train", "cfg", "bank", True),
        ("val", "cfg", "bank", True, "train"),
    ]


def test_model_selection_accepts_only_lower_seen_language_ce():
    script = _load_train_script()

    assert script.is_better_seen_ce(0.49, 0.50)
    assert not script.is_better_seen_ce(0.50, 0.50)
    assert not script.is_better_seen_ce(0.51, 0.50)


def test_smoke_mode_runs_exactly_one_epoch():
    script = _load_train_script()

    assert script.effective_epochs({"train": {"epochs": 15}}, smoke=True) == 1
    assert script.effective_epochs({"train": {"epochs": 15}}, smoke=False) == 15


def test_checkpoint_contains_ema_model_and_required_provenance(toy_model):
    script = _load_train_script()
    ema = TrainableEMA(toy_model, decay=0.9)
    ema_decision = toy_model.decision.weight.detach().clone()
    with torch.no_grad():
        toy_model.decision.weight.add_(3.0)
    live_decision = toy_model.decision.weight.detach().clone()

    payload = script.build_checkpoint_payload(
        toy_model,
        ema,
        epoch=4,
        val_seen_ce=0.125,
        seed=42,
        language_manifest_sha256="abc123",
        cfg={"loss": {"name": "semantic_cross_entropy"}},
    )

    assert set(payload) == {
        "epoch",
        "model_state_dict",
        "ema_state_dict",
        "val_seen_ce",
        "seed",
        "language_manifest_sha256",
        "config",
    }
    torch.testing.assert_close(
        payload["model_state_dict"]["decision.weight"], ema_decision
    )
    torch.testing.assert_close(toy_model.decision.weight, live_decision)
    assert payload["ema_state_dict"].keys() == {"decision.weight"}
    assert payload["val_seen_ce"] == 0.125


def test_locked_configuration_matches_accepted_e2_protocol():
    expected = {
        "experiment_name": "e2_semantic_visual_jev",
        "dataset": {
            "root": "/Dataset/deepfake_detection/DF40_all",
            "methods": [],
            "train_videos_per_method": 586,
            "val_videos_per_method": 100,
            "frames_per_video": 8,
            "image_size": 224,
        },
        "augment": {
            "horizontal_flip_p": 0.5,
            "rotate_p": 0.5,
            "rotate_limit": 10,
            "brightness_contrast_p": 0.5,
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
            "compression_p": 0.2,
            "compression_quality_lower": 40,
            "compression_quality_upper": 100,
        },
        "language": {
            "manifest": "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml",
            "train_tasks": ["T1", "T2", "T3", "T4", "T5"],
            "task_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
        },
        "model": {
            "name": "clip_semantic_jev",
            "clip_model_name": "ViT-B/16",
            "d_model": 256,
            "nhead": 8,
            "ffn_dim": 1024,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "loss": {"name": "semantic_cross_entropy"},
        "train": {
            "init_visual_checkpoint": "experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth",
            "output_dir": "experiments/2026-09-22_semantic_visual_jev_e2/results/training",
            "epochs": 15,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-2,
            "patience": 5,
            "warmup_epochs": 1,
            "ema_decay": 0.999,
            "per_gpu_batch": 128,
            "num_workers": 8,
            "prefetch_factor": 4,
            "precision": "amp",
            "seeds": [42, 7, 2048],
        },
        "evaluation": {
            "primary_domains": ["ffpp", "cdf"],
            "exploratory_domains": ["dfd"],
            "permutation_audit_samples": 4096,
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260922,
        },
        "gates": {
            "noninferiority_margin": 0.01,
            "heldout_macro_drop_max": 0.02,
            "heldout_task_drop_max": 0.05,
            "permutation_error_max": 1.0e-6,
            "t2_balanced_accuracy_min": 0.80,
            "t2_polarity_consistency_min": 0.95,
            "t4_balanced_accuracy_min": 0.65,
            "t4_counterfactual_order_min": 0.80,
            "t5_macro_balanced_accuracy_min": 0.60,
            "meaningless_drop_min": 0.10,
        },
    }

    assert yaml.safe_load(LOCKED_CONFIG.read_text()) == expected


def test_language_manifest_hash_is_raw_file_sha256():
    script = _load_train_script()
    manifest = REPO / (
        "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
    )

    assert script.file_sha256(manifest) == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
