from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from data.semantic_language import load_language_bank


LANGUAGE_BANK = Path(
    "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
)


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
        self.text_anchor = nn.Parameter(
            torch.arange(512, dtype=torch.float32).view(128, 4)
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_anchor[tokens[:, 0]]


@pytest.fixture
def patched_clip(monkeypatch):
    import models.clip_semantic_jev as module

    monkeypatch.setattr(
        module.clip,
        "load",
        lambda *_args, **_kwargs: (_FakeCLIP(), None),
    )
    monkeypatch.setattr(
        module.clip,
        "tokenize",
        lambda texts, truncate=True: torch.arange(len(texts))[:, None],
    )
    return module


@pytest.fixture
def language_bank():
    return load_language_bank(LANGUAGE_BANK)


def test_only_decision_modules_are_trainable_and_clip_stays_in_eval(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(
            (
                "image_projection.",
                "question_projection.",
                "candidate_projection.",
                "decoder.",
                "scorer.",
            )
        )
        for name in trainable
    )
    assert all(not parameter.requires_grad for parameter in model.clip.parameters())
    assert not model.clip.training

    model.train()

    assert model.training
    assert model.decoder.training
    assert not model.clip.training


def test_default_architecture_matches_the_locked_e2_decision_model(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev("fake")

    for projection in (
        model.image_projection,
        model.question_projection,
        model.candidate_projection,
    ):
        assert isinstance(projection.normalization, nn.LayerNorm)
        assert projection.normalization.normalized_shape == (4,)
        assert isinstance(projection.linear, nn.Linear)
        assert (projection.linear.in_features, projection.linear.out_features) == (
            4,
            256,
        )
    assert len(model.decoder.layers) == 2
    layer = model.decoder.layers[0]
    assert layer.self_attn.num_heads == 8
    assert layer.linear1.out_features == 1024
    assert layer.dropout.p == pytest.approx(0.1)
    scalar_scorers = [
        child
        for child in model.scorer.modules()
        if isinstance(child, nn.Linear) and child.out_features == 1
    ]
    assert len(scalar_scorers) == 1


@pytest.mark.parametrize("candidate_count", [2, 3])
def test_two_and_three_candidate_shapes_and_padding(patched_clip, candidate_count):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    image = torch.randn(4, 4)
    question = torch.randn(4, 4)
    candidates = torch.randn(4, 3, 4)
    mask = torch.arange(3)[None, :] < candidate_count
    mask = mask.expand(4, -1)

    logits = model.classify_features(image, question, candidates, mask)

    assert logits.shape == (4, 3)
    assert torch.isneginf(logits[:, candidate_count:]).all()
    assert torch.isfinite(logits[:, :candidate_count]).all()


def test_all_t5_permutations_are_equivariant(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16, dropout=0.0
    ).eval()
    image = torch.randn(2, 4)
    question = torch.randn(2, 4)
    candidates = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    reference = model.classify_features(image, question, candidates, mask)

    for order in itertools.permutations(range(3)):
        order = list(order)
        actual = model.classify_features(
            image, question, candidates[:, order], mask[:, order]
        )
        torch.testing.assert_close(actual, reference[:, order], atol=1e-6, rtol=0)


def test_padded_candidate_has_no_gradient(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16, dropout=0.0
    )
    image = torch.randn(2, 4)
    question = torch.randn(2, 4)
    candidates = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])

    logits = model.classify_features(image, question, candidates, mask)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()

    assert candidates.grad is not None
    assert candidates.grad[:, 2].abs().max().item() == 0.0


def test_frozen_clip_feature_interfaces_do_not_build_gradients(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    images = torch.randn(3, 3, requires_grad=True)

    image_features = model.extract_features(images)
    text_features = model.encode_texts(["a", "b"])

    assert image_features.shape == (3, 4)
    assert text_features.shape == (2, 4)
    assert not image_features.requires_grad
    assert not text_features.requires_grad


def test_h1_checkpoint_loads_only_actual_visual_prefix(patched_clip, tmp_path):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    expected = torch.full_like(model.clip.visual.projection.weight, 2.5)
    original_scorer = model.scorer[1].weight.detach().clone()
    path = tmp_path / "h1.pth"
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": {
                "visual.projection.weight": expected,
                "candidate_embeddings": torch.randn(2, 4),
                "scorer.0.weight": torch.randn(2, 2),
            },
            "auc": 0.75,
        },
        path,
    )

    model.load_h1_visual_checkpoint(path)

    torch.testing.assert_close(model.clip.visual.projection.weight, expected)
    torch.testing.assert_close(model.scorer[1].weight, original_scorer)
    assert all(not parameter.requires_grad for parameter in model.clip.parameters())
    assert not model.clip.training


def test_h1_checkpoint_without_visual_weights_is_rejected(patched_clip, tmp_path):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    path = tmp_path / "not_h1.pth"
    torch.save({"model_state_dict": {"scorer.0.weight": torch.randn(2, 2)}}, path)

    with pytest.raises(ValueError, match="visual weights"):
        model.load_h1_visual_checkpoint(path)


def test_train_lookup_rejects_heldout_text_id(patched_clip, language_bank):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    lookup = patched_clip.EncodedTextLookup.build(
        model,
        language_bank,
        allowed_splits=("train", "fixed"),
        device=torch.device("cpu"),
    )
    heldout_id = language_bank.text_id("t1_q_heldout")

    with pytest.raises(KeyError, match="not encoded"):
        lookup.lookup(
            torch.tensor([heldout_id]),
            torch.tensor(
                [[language_bank.text_id("t1_real_train"), -1, -1]]
            ),
            torch.tensor([[True, False, False]]),
        )


def test_lookup_returns_zeroed_padding_and_encoded_rows(patched_clip, language_bank):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    lookup = patched_clip.EncodedTextLookup.build(
        model,
        language_bank,
        allowed_splits=("train",),
        device=torch.device("cpu"),
    )
    question_id = language_bank.text_id("t1_q_train")
    real_id = language_bank.text_id("t1_real_train")
    fake_id = language_bank.text_id("t1_fake_train")

    questions, candidates = lookup.lookup(
        torch.tensor([question_id]),
        torch.tensor([[real_id, fake_id, -1]]),
        torch.tensor([[True, True, False]]),
    )

    assert questions.shape == (1, 4)
    assert candidates.shape == (1, 3, 4)
    assert torch.count_nonzero(candidates[:, 2]) == 0


def test_lookup_rejects_unmasked_padding_id_even_when_all_splits_are_encoded(
    patched_clip, language_bank
):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16
    )
    lookup = patched_clip.EncodedTextLookup.build(
        model,
        language_bank,
        allowed_splits=("train", "fixed", "heldout"),
        device=torch.device("cpu"),
    )
    question_id = language_bank.text_id("t1_q_train")

    with pytest.raises(KeyError, match="not encoded"):
        lookup.lookup(
            torch.tensor([question_id]),
            torch.tensor([[-1, -1, -1]]),
            torch.tensor([[True, False, False]]),
        )


def test_factory_passes_semantic_architecture_configuration(monkeypatch):
    import models.factory as factory

    received = {}

    def fake_constructor(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "CLIPSemanticVisualJev", fake_constructor)
    result = factory.build_model(
        {
            "name": "clip_semantic_jev",
            "clip_model_name": "custom-clip",
            "d_model": 24,
            "nhead": 3,
            "ffn_dim": 48,
            "num_layers": 4,
            "dropout": 0.2,
        }
    )

    assert result is not None
    assert received == {
        "clip_model_name": "custom-clip",
        "d_model": 24,
        "nhead": 3,
        "ffn_dim": 48,
        "num_layers": 4,
        "dropout": 0.2,
    }
