import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from data.index_ffpp import FrameRecord
from data.semantic_language import load_language_bank


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('e3_evaluation_test', ROOT / 'experiments/2026-09-23_e3_smolvlm_candidate/scripts/evaluate_candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_evaluation_rejects_smoke_incomplete_and_wrong_seed_checkpoints():
    checkpoint = {'smoke': False, 'reportable': True, 'seed': 42, 'world_size': 4,
                  'visible_devices': '0,1,2,3', 'config_sha256': 'config',
                  'protocol_sha256': 'protocol', 'language_manifest_sha256': 'language',
                  'pretrained_manifest_sha256': 'weights', 'weights_kind': 'ema', 'epoch': 3}
    complete = {**checkpoint, 'completed': True, 'best_epoch': 3, 'best_checkpoint_sha256': 'checkpoint'}
    expected = {k: checkpoint[k] for k in ('config_sha256', 'protocol_sha256', 'language_manifest_sha256', 'pretrained_manifest_sha256')}
    module.validate_checkpoint(checkpoint, complete, expected, seed=42, checkpoint_sha='checkpoint')
    for field, value in [('smoke', True), ('reportable', False), ('seed', 7), ('protocol_sha256', 'changed')]:
        modified = {**checkpoint, field: value}
        with pytest.raises(ValueError):
            module.validate_checkpoint(modified, complete, expected, seed=42, checkpoint_sha='checkpoint')
    with pytest.raises(ValueError):
        module.validate_checkpoint(checkpoint, {**complete, 'completed': False}, expected, seed=42, checkpoint_sha='checkpoint')


@pytest.mark.parametrize('condition', ['fixed_h1', 'seen', 'heldout', 'permutation', 'counterfactual', 'meaningless_candidates', 'meaningless_questions'])
def test_e3_semantic_requests_match_frozen_e2_metadata(condition, monkeypatch):
    e2 = module.load_e2()
    records = [FrameRecord('real', 'r', 'a.png', '/a.png', None, 0),
               FrameRecord('simswap', 's', 'b.png', '/b.png', None, 1),
               FrameRecord('fomm', 'f', 'c.png', '/c.png', None, 1)]
    class Base:
        augment = False
        img_size = 224
        def __init__(self): self.records = records
        def __len__(self): return len(records)
    base = e2.CachedMetadataDataset(Base())
    bank = load_language_bank(ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml')
    monkeypatch.setattr(e2, 'build_eval_dataset', lambda *args, **kwargs: base)
    expected = e2.build_condition_loader({'train': {'seed': 42, 'num_workers': 0, 'per_gpu_batch': 4}},
        bank, domain='ffpp', condition=condition, distributed=False, metadata_only=True).dataset
    actual = module.condition_dataset(base, bank, seed=42, condition=condition, e2=e2)
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        a, b = actual[i], expected[i]
        assert a.keys() == b.keys()
        for field in a:
            if field == 'image':
                continue
            if isinstance(a[field], torch.Tensor):
                torch.testing.assert_close(a[field], b[field])
            else:
                assert a[field] == b[field], field
