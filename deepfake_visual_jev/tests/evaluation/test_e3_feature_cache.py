from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from data.index_ffpp import FrameRecord
from evaluation.smolvlm_feature_cache import build_cache, open_cache


class Images(Dataset):
    augment = False

    def __init__(self, root):
        self.records = []
        for i in range(2):
            path = root / f'{i}.png'
            path.write_bytes(b'physical image fixture')
            self.records.append(FrameRecord('real', f'v{i}', path.name, str(path), None, 0))
        # Duplicated source entries must reuse one physical visual feature.
        self.records.append(self.records[0])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {'image': torch.full((3, 2, 2), float(index + 1)), 'sample_id': self.records[index].sample_id}


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.vision_model = nn.Linear(3, 3, bias=False)
        self.backbone.connector = nn.Linear(3, 2, bias=False)
        self.backbone.image_seq_len = 2
        self.backbone.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
        self.backbone.requires_grad_(False)

    def extract_features(self, images):
        values = images.mean((2, 3))
        values = self.backbone.connector(self.backbone.vision_model(values))
        return values[:, None].expand(-1, 2, -1)


def test_cache_preserves_visual_tokens_sample_order_and_rejects_stale_weights(tmp_path):
    model = Vision()
    dataset = Images(tmp_path)
    cfg = {'model': {'revision': 'test'}}
    cache = build_cache(model, dataset, tmp_path / 'cache', domain='ffpp', cfg=cfg,
                        device=torch.device('cpu'), batch_size=2, num_workers=0)
    ids = [dataset.records[i].sample_id for i in (1, 0, 1)]
    expected = model.extract_features(torch.stack([dataset[i]['image'] for i in (1, 0, 1)]))
    torch.testing.assert_close(cache.lookup(ids, torch.device('cpu')), expected, rtol=0, atol=0)
    assert cache.features.shape == (2, 2, 2)
    with torch.no_grad():
        model.backbone.connector.weight.add_(1)
    with pytest.raises(ValueError, match='fingerprint'):
        open_cache(model, dataset, tmp_path / 'cache', domain='ffpp', cfg=cfg)


def test_cache_rejects_modified_physical_images_and_corrupt_features(tmp_path):
    model, dataset = Vision(), Images(tmp_path)
    cfg = {'model': {'revision': 'test'}}
    root = tmp_path / 'cache'
    build_cache(model, dataset, root, domain='cdf', cfg=cfg, device=torch.device('cpu'), batch_size=2, num_workers=0)
    feature_path = root / 'cdf/features.npy'
    values = np.load(feature_path, mmap_mode='r+')
    values[0, 0, 0] += 1
    values.flush()
    with pytest.raises(ValueError, match='content'):
        open_cache(model, dataset, root, domain='cdf', cfg=cfg)
    Path(dataset.records[0].frame_path).write_bytes(b'changed image')
    with pytest.raises(ValueError, match='fingerprint'):
        open_cache(model, dataset, root, domain='cdf', cfg=cfg)
