from pathlib import Path
import sys
import pytest
import torch
from PIL import Image

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_cache_preserves_native_values_and_rejects_missing_ids_or_corruption(tmp_path, dtype):
    from evaluation_cache import build_cache
    from experiment_data import ImageRecords
    from data.index_ffpp import FrameRecord
    records = []
    for i in range(3):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (3, 3), (i * 30, 90, 120)).save(path)
        records.append(FrameRecord('real', str(i), path.name, str(path), None, 0))
    dataset = ImageRecords(records, [0, 0, 0], [1, 1, 1])
    class Extractor(torch.nn.Module):
        def extract_features(self, images):
            return images.mean((2, 3)).unsqueeze(1).to(dtype)
    model = Extractor()
    cache = build_cache(model, dataset, tmp_path / 'cache', identity={'test': 1}, device='cpu', batch_size=2, workers=0)
    value = cache.lookup([records[2].sample_id, records[0].sample_id], 'cpu')
    assert value.dtype == dtype
    assert torch.equal(value, model.extract_features(torch.stack([dataset[2]['image'], dataset[0]['image']])))
    with pytest.raises(ValueError, match='absent'):
        cache.lookup(['missing'], 'cpu')
    del cache
    file = tmp_path / 'cache/features.npy'
    with file.open('r+b') as f:
        f.seek(-2, 2)
        f.write(b'xx')
    with pytest.raises(ValueError, match='hash'):
        build_cache(model, dataset, tmp_path / 'cache', identity={'test': 1}, device='cpu', batch_size=2, workers=0)
