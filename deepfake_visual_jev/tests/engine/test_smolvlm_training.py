import copy

import pytest
import torch
from torch import nn

from engine.smolvlm_trainer import train_epoch, validate_epoch


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, images, questions, candidates, mask):
        return self.linear(images)


class Lookup:
    def lookup(self, qids, cids, mask):
        return qids[:, None], cids[..., None]


def batches():
    return [{'image': torch.tensor([[float(i), 1.0], [1.0, float(i)]]),
             'question_text_id': torch.zeros(2, dtype=torch.long),
             'candidate_text_ids': torch.tensor([[1, 2], [1, 2]]),
             'candidate_mask': torch.ones(2, 2, dtype=torch.bool),
             'target_index': torch.tensor([0, 1])} for i in range(1, 4)]


def test_accumulation_matches_full_batches_including_final_partial_group():
    torch.manual_seed(4)
    model = SmallClassifier()
    expected = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    ref_optimizer = torch.optim.SGD(expected.parameters(), lr=.1)
    data = batches()
    for group in [data[:2], data[2:]]:
        ref_optimizer.zero_grad()
        images = torch.cat([b['image'] for b in group])
        target = torch.cat([b['target_index'] for b in group])
        nn.functional.cross_entropy(expected.linear(images), target).backward()
        nn.utils.clip_grad_norm_(expected.parameters(), 1.)
        ref_optimizer.step()
    result = train_epoch(model, data, Lookup(), optimizer, torch.amp.GradScaler('cuda', enabled=False),
                         torch.device('cpu'), accumulation=2, amp=False)
    for actual, reference in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, reference)
    assert result['optimizer_steps'] == 2 and result['skipped_steps'] == 0
    assert result['count'] == 6 and result['batches'] == 3


def test_validation_is_weighted_by_examples_and_does_not_update_parameters():
    model = SmallClassifier()
    data = batches()
    before = copy.deepcopy(model.state_dict())
    expected = nn.functional.cross_entropy(model.linear(torch.cat([b['image'] for b in data])),
                                          torch.cat([b['target_index'] for b in data])).item()
    result = validate_epoch(model, data, Lookup(), torch.device('cpu'))
    assert result['loss'] == pytest.approx(expected, rel=1e-6)
    assert result['count'] == 6
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
