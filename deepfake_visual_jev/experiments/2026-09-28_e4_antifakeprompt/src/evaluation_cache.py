"""Native-precision EVA tokens shared across prompt/scorer arms with provenance."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_data import atomic_json, sha256
from evaluation.e2_feature_cache import _image_stat_fingerprint


class FeatureCache:
    def __init__(self, array, ids, manifest):
        self.array, self.manifest = array, manifest
        self.index = {key: i for i, key in enumerate(ids)}

    def lookup(self, ids, device):
        try:
            indices = [self.index[str(key)] for key in ids]
        except KeyError as exc:
            raise ValueError(f'physical sample absent from cache: {exc}') from exc
        return torch.from_numpy(np.array(self.array[indices], copy=True)).to(device)


@torch.no_grad()
def build_cache(model, dataset, root, *, identity, device, batch_size=16, workers=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, path = root / 'manifest.json', root / 'features.npy'
    ids = [r.sample_id for r in dataset.records]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('cache needs unique nonempty physical records')
    expected = {'identity': identity, 'image_stat_sha256': _image_stat_fingerprint(dataset.records), 'ids': ids}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('complete') or any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('cache provenance mismatch')
        if sha256(path) != manifest['features_sha256']:
            raise ValueError('cache content hash mismatch')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if str(array.dtype) != manifest['dtype'] or list(array.shape) != manifest['shape']:
            raise ValueError('cache shape/dtype mismatch')
        return FeatureCache(array, ids, manifest)
    partial = root / 'features.partial.npy'
    if path.exists() or partial.exists():
        raise FileExistsError('partial cache retained; do not silently overwrite')
    options = {'batch_size': batch_size, 'num_workers': workers, 'pin_memory': str(device).startswith('cuda')}
    if workers:
        options.update(multiprocessing_context='spawn', prefetch_factor=2)
    loader = DataLoader(dataset, shuffle=False, **options)
    model.eval()
    array, written = None, 0
    for batch in loader:
        features = model.extract_features(batch['image'].to(device))
        if features.dtype not in (torch.float16, torch.float32) or not torch.isfinite(features).all():
            raise ValueError(f'cache needs finite native fp16/fp32 features, got {features.dtype}')
        if array is None:
            shape = (len(dataset), *features.shape[1:])
            native_dtype = features.cpu().numpy().dtype
            array = np.lib.format.open_memmap(partial, mode='w+', dtype=native_dtype, shape=shape)
            print('cache native dtype', str(native_dtype), 'shape', shape, flush=True)
        if features.cpu().numpy().dtype != array.dtype:
            raise ValueError('inconsistent native feature dtype')
        count = len(features)
        if list(batch['sample_id']) != ids[written:written + count]:
            raise ValueError('cache row order mismatch')
        array[written:written + count] = features.cpu().numpy()
        written += count
        if written % (batch_size * 100) == 0:
            print(json.dumps({'event': 'cache_progress', 'directory': str(root), 'written': written, 'total': len(ids)}), flush=True)
    if written != len(ids):
        raise ValueError('incomplete cache')
    array.flush()
    shape, dtype = list(array.shape), str(array.dtype)
    del array
    partial.replace(path)
    manifest = {**expected, 'shape': shape, 'dtype': dtype, 'features_sha256': sha256(path), 'complete': True}
    atomic_json(manifest_path, manifest)
    return FeatureCache(np.load(path, mmap_mode='r', allow_pickle=False), ids, manifest)
