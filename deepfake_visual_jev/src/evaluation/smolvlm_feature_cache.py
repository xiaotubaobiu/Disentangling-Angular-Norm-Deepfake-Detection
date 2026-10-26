"""FP32 native visual-token cache, shared across E3 seeds and candidates."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from data.smolvlm_data import ExactShardSampler, NativeImageDataset
from evaluation.e2_feature_cache import (
    FrozenFeatureCache, _record_identity, _image_stat_fingerprint,
    _digest_file, _digest_json,
)


def visual_fingerprint(model):
    digest = hashlib.sha256()
    for prefix in ('vision_model', 'connector'):
        module = getattr(model.backbone, prefix)
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError('feature cache requires frozen vision and connector')
        for name, value in sorted(module.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(f'{prefix}.{name}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def expected_manifest(model, dataset, *, domain, cfg):
    ids, indices, record_sha = _record_identity(dataset)
    expected = {
        'version': 1, 'model_family': 'e3_smolvlm_native_tokens', 'domain': domain,
        'dtype': 'float32', 'shape': [len(ids), model.backbone.image_seq_len,
                                     model.backbone.config.text_config.hidden_size],
        'index_rows': len(dataset), 'records_sha256': record_sha,
        'image_stat_sha256': _image_stat_fingerprint(dataset.records),
        'visual_sha256': visual_fingerprint(model),
        'preprocess_sha256': _digest_json({
            'native_dataset': inspect.getsource(NativeImageDataset),
            'extract_features': inspect.getsource(type(model).extract_features),
            'model_config': cfg['model'], 'torch_version': torch.__version__}),
    }
    return expected, ids, indices


def _rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def open_cache(model, dataset, root, *, domain, cfg):
    directory = Path(root) / domain
    expected, ids, _ = expected_manifest(model, dataset, domain=domain, cfg=cfg)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{domain} native cache fingerprint mismatch')
    path = directory / 'features.npy'
    rank, _ = _rank_world()
    error = [None]
    if rank == 0 and _digest_file(path) != manifest['features_sha256']:
        error[0] = f'{domain} native cache content fingerprint mismatch'
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])
    features = np.load(path, mmap_mode='r', allow_pickle=False)
    if features.dtype != np.float32 or list(features.shape) != expected['shape']:
        raise ValueError('native cache shape/dtype mismatch')
    return FrozenFeatureCache(features, ids, manifest)


@torch.no_grad()
def build_cache(model, dataset, root, *, domain, cfg, device, batch_size=32, num_workers=4):
    directory = Path(root) / domain
    if (directory / 'manifest.json').exists():
        return open_cache(model, dataset, root, domain=domain, cfg=cfg)
    expected, ids, unique_indices = expected_manifest(model, dataset, domain=domain, cfg=cfg)
    rank, world = _rank_world()
    temporary = directory / 'features.npy.tmp'
    final = directory / 'features.npy'
    if rank == 0:
        if temporary.exists() or final.exists():
            raise FileExistsError('partial cache preserved; inspect it before a recovery run')
        directory.mkdir(parents=True, exist_ok=True)
        mapping = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                           shape=tuple(expected['shape']))
        mapping.flush()
        del mapping
    _barrier()
    mapping = np.load(temporary, mmap_mode='r+', allow_pickle=False)
    dataset_unique = Subset(dataset, unique_indices)
    kwargs = {'num_workers': num_workers, 'pin_memory': device.type == 'cuda'}
    if num_workers:
        kwargs.update(multiprocessing_context='spawn', prefetch_factor=2)
    loader = DataLoader(dataset_unique, batch_size=batch_size,
                        sampler=ExactShardSampler(dataset_unique, rank=rank, world_size=world), **kwargs)
    row_for_id = {identity: row for row, identity in enumerate(ids)}
    model.eval()
    written = 0
    for batch_index, batch in enumerate(loader):
        features = model.extract_features(batch['image'].to(device)).float()
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite native visual features')
        rows = [row_for_id[str(identity)] for identity in batch['sample_id']]
        mapping[rows] = features.cpu().numpy()
        written += len(rows)
        if rank == 0 and (batch_index + 1) % 100 == 0:
            print(json.dumps({'event': 'native_cache_progress', 'domain': domain,
                              'rank0_rows': written, 'unique_rows': len(ids)}), flush=True)
    mapping.flush()
    del mapping
    count = torch.tensor(written, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(count)
    if int(count) != len(ids):
        raise ValueError('native cache did not cover each unique physical frame')
    _barrier()
    if rank == 0:
        temporary.replace(final)
        manifest = {**expected, 'features_sha256': _digest_file(final),
                    'complete': True, 'world_size': world}
        path = directory / 'manifest.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(manifest, indent=2) + '\n')
        temp.replace(path)
    _barrier()
    manifest = json.loads((directory / 'manifest.json').read_text())
    return FrozenFeatureCache(np.load(final, mmap_mode='r', allow_pickle=False), ids, manifest)
