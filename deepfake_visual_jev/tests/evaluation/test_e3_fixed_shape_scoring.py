"""Evaluation batching must not change a candidate's numerical context."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation.semantic_controls import permutation_audit
from models.smolvlm_candidate import SmolVLMCandidateScorer

spec = importlib.util.spec_from_file_location('e3_candidate_fixtures',
    Path(__file__).parents[1] / 'models/test_smolvlm_candidate.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def make_case():
    from evaluation.smolvlm_fixed_shape import FixedShapeCandidateScorer
    baseline, lookup, _ = fixtures.make_model()
    # A long candidate crosses the 128-token boundary; 5 examples cross chunks.
    baseline.register_texts([SimpleNamespace(text_id=7, text='long ' * 30)])
    model = FixedShapeCandidateScorer(fixtures.TinyBackbone(), fixtures.TinyPrompts(), score_batch_size=4)
    model.load_state_dict(baseline.state_dict(), strict=True)
    model.register_texts([SimpleNamespace(text_id=k, text=v) for k, v in baseline._texts.items()])
    model.eval()
    baseline.eval()
    mask = torch.tensor([[True, True, False], [True, True, True], [True, True, False],
                         [True, True, True], [True, True, False]])
    q = torch.tensor([[0], [3], [0], [3], [0]])
    c = torch.tensor([[[1], [2], [-1]], [[4], [7], [6]], [[2], [1], [-1]],
                      [[7], [6], [5]], [[1], [2], [-1]]])
    features = torch.randn(5, 2, 8)
    return baseline, model, features, q, c, mask


def test_fixed_shape_scoring_preserves_context_scores_and_state_dict():
    baseline, model, features, q, c, mask = make_case()
    with torch.inference_mode():
        expected = baseline.classify_features(features, q, c, mask)
        actual = model.classify_features(features, q, c, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.isneginf(actual[~mask]).all()
    assert model.state_dict().keys() == baseline.state_dict().keys()
    assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in model.state_dict().items())


def test_subgroups_and_permutations_preserve_fixed_compute_shapes():
    _, model, features, q, c, mask = make_case()
    result = permutation_audit(model, features, q, c, mask, reduce_across_ranks=False)
    assert result['max_probability_error'] <= 1e-6
    assert result['prediction_consistency'] == 1.0
    assert {tuple(call.shape) for call in model.backbone.calls} == {(4, 128), (4, 256)}
    with torch.inference_mode():
        full = model.classify_features(features, q, c, mask)
        for i in range(len(mask)):
            single = model.classify_features(features[i:i+1], q[i:i+1], c[i:i+1], mask[i:i+1])
            torch.testing.assert_close(single, full[i:i+1], atol=1e-6, rtol=1e-6)


def test_evaluation_scorer_refuses_training_and_overlength_prompts():
    _, model, features, q, c, mask = make_case()
    model.train()
    with pytest.raises(ValueError, match='evaluation'):
        model.classify_features(features, q, c, mask)
    model.eval()
    model.register_texts([SimpleNamespace(text_id=8, text='x' * 1100)])
    c[0, 0, 0] = 8
    with pytest.raises(ValueError, match='1024'):
        model.classify_features(features, q, c, mask)
