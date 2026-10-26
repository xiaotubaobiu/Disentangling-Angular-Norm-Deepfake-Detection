"""Frozen visual cache must preserve direct features and reject stale inputs."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from evaluation.e2_feature_cache import build_feature_cache, open_feature_cache


class _TinyImages(Dataset):
    augment = False
    img_size = 224

    def __init__(self, root, order=(0, 1, 2)):
        self.order = order
        for i in set(order):
            path = root / f"{i}.png"
            if not path.exists():
                path.write_bytes(f"image-{i}".encode())
        self.records = [SimpleNamespace(sample_id=f"frame-{i}", frame_path=str(root / f"{i}.png"),
                                        label=i % 2, method="simswap", video_id=str(i),
                                        landmark_path=None) for i in order]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        frame_id = int(self.records[index].sample_id.split("-")[1])
        return {"image": torch.tensor([float(frame_id), 1.0]),
                "sample_id": self.records[index].sample_id}


class _TinyVisual(nn.Module):
    output_dim = 3

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, images):
        return self.linear(images)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = nn.Module()
        self.clip.visual = _TinyVisual()

    def extract_features(self, images):
        return self.clip.visual(images).float()


def test_frozen_feature_cache_matches_direct_and_rejects_stale_provenance(tmp_path):
    torch.manual_seed(42)
    model = _TinyModel().eval()
    dataset = _TinyImages(tmp_path)
    cfg = {"dataset": {"image_size": 224}, "model": {"clip_model_name": "ViT-B/16"}}
    sample_ids = ["frame-2", "frame-0", "frame-2"]

    build_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg,
                        device=torch.device("cpu"), batch_size=2, num_workers=0)
    cache = open_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg)
    actual = cache.lookup(sample_ids, torch.device("cpu"))
    direct = model.extract_features(torch.stack((dataset[2]["image"],
                                                 dataset[0]["image"],
                                                 dataset[2]["image"])))

    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    with pytest.raises(ValueError, match="fingerprint"):
        open_feature_cache(model, _TinyImages(tmp_path, order=(2, 1, 0)), tmp_path,
                           domain="ffpp", cfg=cfg)
    with pytest.raises(ValueError, match="fingerprint"):
        open_feature_cache(model, dataset, tmp_path, domain="ffpp",
                           cfg={"dataset": {"image_size": 256}, "model": cfg["model"]})
    with torch.no_grad():
        model.clip.visual.linear.weight.add_(1.0)
    with pytest.raises(ValueError, match="fingerprint"):
        open_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg)


def test_identical_index_duplicates_share_one_feature_but_keep_full_provenance(tmp_path):
    model = _TinyModel().eval()
    dataset = _TinyImages(tmp_path, order=(0, 1, 0, 2))
    cfg = {"dataset": {"image_size": 224}, "model": {"clip_model_name": "ViT-B/16"}}

    cache = build_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg,
                                device=torch.device("cpu"), batch_size=2)

    assert cache.manifest["shape"] == [3, 3]
    torch.testing.assert_close(cache.lookup(["frame-0", "frame-0"], torch.device("cpu")),
                               model.extract_features(torch.stack([dataset[0]["image"], dataset[2]["image"]])),
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match="fingerprint"):
        open_feature_cache(model, _TinyImages(tmp_path, order=(0, 1, 2, 0)), tmp_path,
                           domain="ffpp", cfg=cfg)


def test_conflicting_index_duplicate_is_rejected(tmp_path):
    model = _TinyModel().eval()
    dataset = _TinyImages(tmp_path, order=(0, 0))
    dataset.records[1].frame_path = "/different/image.png"
    cfg = {"dataset": {"image_size": 224}, "model": {"clip_model_name": "ViT-B/16"}}

    with pytest.raises(ValueError, match="conflicting"):
        build_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg,
                            device=torch.device("cpu"))


def test_changed_image_at_same_record_path_invalidates_cache(tmp_path):
    model = _TinyModel().eval()
    dataset = _TinyImages(tmp_path)
    cfg = {"dataset": {"image_size": 224}, "model": {"clip_model_name": "ViT-B/16"}}
    build_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg,
                        device=torch.device("cpu"))
    (tmp_path / "1.png").write_bytes(b"changed-image-content")

    with pytest.raises(ValueError, match="fingerprint"):
        open_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg)


def test_changed_feature_extraction_implementation_invalidates_cache(tmp_path, monkeypatch):
    model = _TinyModel().eval()
    dataset = _TinyImages(tmp_path)
    cfg = {"dataset": {"image_size": 224}, "model": {"clip_model_name": "ViT-B/16"}}
    build_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg,
                        device=torch.device("cpu"))

    def changed_extract_features(self, images):
        return self.clip.visual(images).float() + 1.0

    monkeypatch.setattr(_TinyModel, "extract_features", changed_extract_features)
    with pytest.raises(ValueError, match="fingerprint"):
        open_feature_cache(model, dataset, tmp_path, domain="ffpp", cfg=cfg)
