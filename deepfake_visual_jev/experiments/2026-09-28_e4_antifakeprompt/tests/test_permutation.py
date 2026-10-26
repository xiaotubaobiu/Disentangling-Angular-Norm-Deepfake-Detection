import sys
from pathlib import Path
import torch

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]


def test_permutation_audit_changes_order_without_changing_batch_composition():
    from permutation_eval import fixed_batch_audit
    class BatchSensitive(torch.nn.Module):
        def classify_features(self, features, questions, candidates, mask):
            # A batching effect is separate from a candidate-position effect.
            return (candidates[..., 0].float() / len(features)).masked_fill(~mask, -torch.inf)
    model = BatchSensitive()
    candidates = torch.tensor([[[1], [2], [-1]], [[1], [2], [3]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    result = fixed_batch_audit(model, torch.zeros(2, 1), torch.zeros(2, 1), candidates, mask)
    assert result['max_probability_error'] < 1e-6
    assert result['consistent_count'] == result['comparison_count'] == 8


def test_permutation_audit_detects_actual_position_bias():
    from permutation_eval import fixed_batch_audit
    class PositionBiased(torch.nn.Module):
        def classify_features(self, features, questions, candidates, mask):
            return (candidates[..., 0].float() + torch.tensor([0., 20., 0.])).masked_fill(~mask, -torch.inf)
    model = PositionBiased()
    candidates = torch.tensor([[[1], [2], [-1]], [[1], [2], [3]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    result = fixed_batch_audit(model, torch.zeros(2, 1), torch.zeros(2, 1), candidates, mask)
    assert result['max_probability_error'] > .9
    assert result['consistent_count'] < result['comparison_count']
