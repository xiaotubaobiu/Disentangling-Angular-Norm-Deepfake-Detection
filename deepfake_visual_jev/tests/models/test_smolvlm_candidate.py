"""Candidate scoring must preserve context, gradients, and candidate semantics."""
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from models.smolvlm_candidate import SmolVLMCandidateScorer, CandidateTextLookup


@pytest.mark.parametrize('question,candidate', [
    ('Is this face real?', 'Yes'),
    ('Is this face not authentic?', 'No'),
    ('Which source describes this face?', 'A face swap rather than a camera original.'),
])
def test_cached_prompt_matches_official_unsplit_image_processor(question, candidate):
    from PIL import Image
    from transformers import AutoProcessor
    from models.smolvlm_candidate import CandidatePrompts

    path = Path(__file__).resolve().parents[2] / 'experiments/2026-09-23_e3_smolvlm_candidate/results/pretrained/smolvlm_256m'
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    text = f'Question: {question}\nCandidate answer: {candidate}\nEvaluate whether this candidate correctly answers the question about the image.'
    chat = processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}],
        tokenize=False, add_generation_prompt=True)
    expected = processor(text=chat, images=Image.new('RGB', (224, 224)),
                         do_image_splitting=False, size={'longest_edge': 512},
                         max_image_size={'longest_edge': 512})['input_ids'][0]
    actual = CandidatePrompts(processor).encode_pair(question, candidate)
    assert actual == expected
    assert actual.count(processor.image_token_id) == 64


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(3, 8)
        self.connector = nn.Linear(8, 8)
        self.text_model = nn.Embedding(256, 8)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
        self.image_seq_len = 2
        self.calls = []

    def get_image_features(self, pixel_values, **kwargs):
        image = pixel_values.mean(dim=(-1, -2)).squeeze(1)
        features = self.connector(self.vision_model(image))[:, None, :].expand(-1, 2, -1)
        return SimpleNamespace(pooler_output=features)

    def forward(self, input_ids, attention_mask, image_hidden_states, use_cache, **kwargs):
        assert use_cache is False
        self.calls.append(input_ids.detach().clone())
        text = self.text_model(input_ids) * attention_mask[..., None]
        hidden = text.cumsum(1) + image_hidden_states.mean(1)[:, None, :]
        return SimpleNamespace(last_hidden_state=hidden)


class TinyPrompts:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def encode_pair(self, question, candidate):
        self.calls.append((question, candidate))
        return [1, 1, *[2 + ord(x) % 250 for x in question + ':' + candidate]]


def make_model():
    torch.manual_seed(9)
    model = SmolVLMCandidateScorer(TinyBackbone(), TinyPrompts(), score_batch_size=32)
    entries = [SimpleNamespace(text_id=i, text=t, split=s) for i, t, s in
               [(0, 'Is it real?', 'train'), (1, 'yes', 'train'), (2, 'no', 'train'),
                (3, 'Which source?', 'heldout'), (4, 'camera', 'heldout'),
                (5, 'swap', 'heldout'), (6, 'reenact', 'heldout')]]
    bank = SimpleNamespace(text_entries=entries)
    lookup = CandidateTextLookup.build(model, bank, ('train', 'heldout'), torch.device('cpu'))
    return model, lookup, bank


def test_shared_scorer_is_equivariant_for_two_and_three_candidates():
    model, lookup, _ = make_model()
    model.eval()
    image = torch.randn(2, 3, 5, 5)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    questions, candidates = lookup.lookup(torch.tensor([0, 3]), torch.tensor([[1, 2, -1], [4, 5, 6]]), mask)
    scores = model(image, questions, candidates, mask)
    order = torch.tensor([1, 0, 2])
    permuted = model(image, questions, candidates[:, order], mask[:, order])
    torch.testing.assert_close(permuted, scores[:, order])
    assert scores.shape == (2, 3) and torch.isneginf(scores[0, 2])
    assert torch.isfinite(scores[mask]).all()
    assert not any('lm_head' in name for name, _ in model.named_modules())


def test_training_reaches_pretrained_language_and_shared_head_but_not_frozen_vision():
    model, lookup, _ = make_model()
    model.train()
    mask = torch.tensor([[True, True]])
    q, c = lookup.lookup(torch.tensor([0]), torch.tensor([[1, 2]]), mask)
    logits = model(torch.randn(1, 3, 5, 5), q, c, mask)
    nn.functional.cross_entropy(logits, torch.tensor([0])).backward()
    assert model.backbone.text_model.weight.grad.abs().sum() > 0
    assert model.scorer[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.vision_model.parameters())
    assert not model.backbone.vision_model.training
    assert not model.backbone.connector.training


def test_model_receives_question_candidate_context_without_targets():
    model, lookup, _ = make_model()
    mask = torch.tensor([[True, True]])
    q, c = lookup.lookup(torch.tensor([0]), torch.tensor([[1, 2]]), mask)
    model(torch.randn(1, 3, 5, 5), q, c, mask)
    assert model.prompts.calls == [('Is it real?', 'yes'), ('Is it real?', 'no')]


def test_heldout_text_cannot_be_looked_up_by_training_lookup():
    model, _, bank = make_model()
    train = CandidateTextLookup.build(model, bank, ('train',), torch.device('cpu'))
    with pytest.raises(KeyError, match='split'):
        train.lookup(torch.tensor([3]), torch.tensor([[1, 2]]), torch.tensor([[True, True]]))


def test_unknown_candidate_id_is_rejected_before_language_forward():
    model, lookup, _ = make_model()
    with pytest.raises(KeyError):
        lookup.lookup(torch.tensor([0]), torch.tensor([[1, 999]]), torch.tensor([[True, True]]))
    assert model.backbone.calls == []
