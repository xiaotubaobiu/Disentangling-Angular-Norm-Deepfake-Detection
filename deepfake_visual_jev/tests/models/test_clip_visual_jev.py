from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FakeVisual(nn.Module):
    output_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, self.output_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


class _FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()
        self.text_anchor = nn.Parameter(torch.arange(8, dtype=torch.float32).view(2, 4))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_anchor[tokens[:, 0]]


@pytest.fixture
def patched_clip(monkeypatch):
    import models.clip_visual_jev as module

    fake_model = _FakeCLIP()
    monkeypatch.setattr(module.clip, "load", lambda *_args, **_kwargs: (fake_model, None))
    monkeypatch.setattr(
        module.clip,
        "tokenize",
        lambda prompts, truncate=True: torch.tensor([[0], [1]], dtype=torch.long),
    )
    return module


def test_visual_jev_freezes_clip_and_only_trains_one_shared_scorer(patched_clip):
    model = patched_clip.CLIPVisualJevClassifier(clip_model_name="fake")

    assert model.candidate_embeddings.shape == (2, 4)
    assert not model.candidate_embeddings.requires_grad
    assert all(not parameter.requires_grad for parameter in model.visual.parameters())
    assert all(
        name.startswith("scorer.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )

    scalar_outputs = [
        layer for layer in model.scorer.modules()
        if isinstance(layer, nn.Linear) and layer.out_features == 1
    ]
    assert len(scalar_outputs) == 1


def test_visual_jev_scores_two_candidates_with_expected_shape(patched_clip):
    model = patched_clip.CLIPVisualJevClassifier(clip_model_name="fake")
    images = torch.randn(5, 3)

    logits = model(images)

    assert logits.shape == (5, 2)
    assert torch.isfinite(logits).all()


def test_candidate_swap_permutes_logits_and_preserves_remapped_loss(patched_clip):
    model = patched_clip.CLIPVisualJevClassifier(clip_model_name="fake")
    features = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 1, 0])

    original = model.classify_features(features)
    swapped = model.classify_features(features, candidate_order=(1, 0))

    torch.testing.assert_close(swapped, original[:, [1, 0]])
    original_loss = F.cross_entropy(original, labels)
    swapped_loss = F.cross_entropy(swapped, 1 - labels)
    torch.testing.assert_close(swapped_loss, original_loss)


def test_train_mode_keeps_frozen_visual_in_eval_mode(patched_clip):
    model = patched_clip.CLIPVisualJevClassifier(clip_model_name="fake")

    model.train()

    assert model.training
    assert model.scorer.training
    assert not model.visual.training


def test_a_checkpoint_loads_visual_and_ignores_binary_head(patched_clip):
    model = patched_clip.CLIPVisualJevClassifier(clip_model_name="fake")
    replacement = torch.full_like(model.visual.projection.weight, 3.25)
    a_state = {
        "visual.projection.weight": replacement,
        "classifier.weight": torch.randn(2, 4),
        "classifier.bias": torch.randn(2),
    }

    incompatible = model.load_state_dict(a_state, strict=False)

    torch.testing.assert_close(model.visual.projection.weight, replacement)
    assert set(incompatible.unexpected_keys) == {"classifier.weight", "classifier.bias"}
    assert all(key.startswith("scorer.") or key == "candidate_embeddings" for key in incompatible.missing_keys)
