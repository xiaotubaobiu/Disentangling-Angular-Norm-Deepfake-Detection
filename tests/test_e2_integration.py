"""Cross-task E2 sampler, frozen text lookup, decision, and row interface."""
from pathlib import Path
import importlib.util

import pytest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.semantic_dataset import SemanticDataset, SemanticTemplateSampler
from data.semantic_language import load_language_bank
from engine.semantic_trainer import semantic_classification_step
from evaluation.semantic_eval import frame_semantic_rows


MANIFEST = Path(__file__).resolve().parents[1] / "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
PREFLIGHT = Path(__file__).resolve().parents[1] / "experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py"


def _preflight():
    spec = importlib.util.spec_from_file_location("e2_preflight", PREFLIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _OneImage(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": torch.tensor([1., 0., 0.]), "label": torch.tensor(1),
                "method_id": "simswap", "video_id": "001",
                "video_key": "simswap::001", "sample_id": f"sample-{index}",
                "image_path": "/tmp/e2-integration.png"}


class _Visual(nn.Module):
    output_dim = 4

    def forward(self, images):
        return torch.nn.functional.pad(images, (0, 1))


class _TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.anchor = nn.Parameter(torch.arange(512, dtype=torch.float32).reshape(128, 4))

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens):
        return self.anchor[tokens[:, 0]]


def test_sampler_model_step_and_evaluator_round_trip(monkeypatch):
    import models.clip_semantic_jev as module

    monkeypatch.setattr(module.clip, "load", lambda *_args, **_kwargs: (_TinyCLIP(), None))
    monkeypatch.setattr(module.clip, "tokenize", lambda texts, truncate=True: torch.arange(len(texts))[:, None])
    bank = load_language_bank(MANIFEST)
    dataset = SemanticDataset(_OneImage(), SemanticTemplateSampler(bank, seed=42),
                              mode="suite", tasks=("T1", "T2", "T3", "T4", "T5"),
                              split="train")
    batch = next(iter(DataLoader(dataset, batch_size=5)))
    model = module.CLIPSemanticVisualJev("tiny", d_model=8, nhead=2,
                                         ffn_dim=16, dropout=0.0)
    lookup = module.EncodedTextLookup.build(model, bank, allowed_splits=("train",),
                                             device=torch.device("cpu"))

    loss, logits = semantic_classification_step(model, batch, lookup, torch.device("cpu"))
    rows = frame_semantic_rows(batch, logits.softmax(dim=-1),
                               {"seed": 42, "domain": "ffpp", "condition": "seen"})

    assert torch.isfinite(loss)
    assert len(rows) == 5
    assert {row["task_id"] for row in rows} == {"T1", "T2", "T3", "T4", "T5"}
    assert {row["video_key"] for row in rows} == {"simswap::001"}
    assert all(row["condition"] == "seen" for row in rows)


def test_preflight_fails_closed_on_missing_method_or_unauthorized_gpu():
    from data.constants import ALL_METHODS

    module = _preflight()
    inventory = {method: {split: 2 for split in ("train", "val", "ffpp", "cdf")}
                 for method in ALL_METHODS}
    module.validate_dataset_inventory(inventory)
    inventory[ALL_METHODS[0]]["cdf"] = 0
    with pytest.raises(ValueError, match="cdf"):
        module.validate_dataset_inventory(inventory)

    devices = [{"index": index, "free_bytes": 29 * 1024**3,
                "total_bytes": 32 * 1024**3} for index in range(4)]
    module.validate_devices(devices, "0,1,2,3")
    with pytest.raises(ValueError, match="0,1,2,3"):
        module.validate_devices(devices, "4,5,6,7")
    devices[2]["free_bytes"] = 27 * 1024**3
    with pytest.raises(ValueError, match="28 GiB"):
        module.validate_devices(devices, "0,1,2,3")


def test_preflight_uses_the_actual_h1_validation_video_slice(monkeypatch):
    module = _preflight()
    calls = []

    def index(root, method, domain, split, label, **kwargs):
        calls.append((root, method, domain, split, label, kwargs))
        return [object()]

    monkeypatch.setattr(module, "load_json_index", index)
    cfg = {"dataset": {"root": "/dataset", "train_videos_per_method": 586,
                       "val_videos_per_method": 100}}

    assert len(module.index_split_records(cfg, "simswap", "val", 1)) == 1
    assert calls == [("/dataset", "simswap", "ff", "train", 1,
                      {"video_range": (586, 686)})]
