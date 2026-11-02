"""Protocol D FP32 caches bound to new frozen formal training artifacts.

``extract(config_path, run_id, device='cuda')`` writes one cache per available
role to ``artifact_root/features/D/run_id/role``. ``verify_cache`` checks live
provenance and every committed chunk. ``load_features`` returns manifest,
raw FP32 features, ORIGINAL FP32 two-output weights, and ORIGINAL FP32 biases.
Promote both head rows to FP64 before subtracting them in the scoring module.

Only source-DEV selected EMA best_model.pth from a matching TRAINING_COMPLETE formal run is accepted. Source
development checkpoints and any earlier experiment's models are ineligible.
Chunk markers are committed after atomic .npy writes; incomplete unmarked files
are overwritten on resume, while corrupted committed chunks cause a hard stop.
"""
from __future__ import annotations

import hashlib
import contextlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import torch

from . import trainer

DIMENSIONS = {'clip_vit_b16': 512, 'dinov2_vit_b14': 768, 'efficientnet_b0': 1280}
ROLES = ('train', 'dev', 'source_cal', 'source_test', 'target_full', 'target_cal', 'target_test')
EPSILON = 1e-12
SCHEMA = 'journal_detector-best-ema-D-fp32-cache-1'


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _cache_code_hash():
    return trainer.digest(Path(__file__))


def _required_hash(path, description):
    if not Path(path).is_file():
        raise ValueError(f'missing {description}; provenance is incomplete: {path}')
    return trainer.digest(path)


def _context(config_path, run_id):
    """Recompute the trainer's complete identity without running any inference."""
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id) or run_id in ('.', '..'):
        raise ValueError('unsafe run identity')
    config_path = Path(config_path).resolve()
    config_hash = _required_hash(config_path, 'config')
    config = json.loads(config_path.read_text())
    root = Path(config['artifact_root']).resolve()
    task = config.get('runs', {}).get(run_id)
    if task is None:
        raise ValueError('unregistered run identity')
    if task.get('phase') != 'formal' or task.get('recipe_frozen') is not True:
        raise ValueError('D feature extraction requires a frozen formal recipe; development models cannot infer targets')
    backbone = task['backbone']
    if backbone not in DIMENSIONS:
        raise ValueError('unregistered backbone identity')
    if 'formal_seeds' in config and task['seed'] not in config['formal_seeds']:
        raise ValueError('run is not a registered formal seed')
    epochs = task['epochs']
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError('invalid frozen epoch budget')
    data = root / 'data/D'
    train_path = (root / task['train_manifest']).resolve()
    dev_path = (root / task['dev_manifest']).resolve()
    if train_path != data / 'train.parquet' or dev_path != data / 'dev.parquet':
        raise ValueError('protocol D requires new data/D train and dev manifests')
    init = Path(config['backbones'][backbone]['weight_path']).resolve()
    from .protocol import validate_config
    validate_config(config)
    identity=trainer.training_identity(config,task)
    run_root = root / 'runs' / run_id
    final_path = run_root / 'best_model.pth'
    if final_path.resolve().parent != run_root.resolve() or run_root.resolve().parent != (root / 'runs').resolve():
        raise ValueError('checkpoint must be a new local run best EMA, not an external model')
    receipt_path = run_root / 'complete.json'
    receipt_hash = _required_hash(receipt_path, 'training completion receipt')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'TRAINING_COMPLETE':
        raise ValueError('formal training is incomplete')
    if receipt.get('identity') != identity:
        raise ValueError('formal checkpoint identity mismatch: config/data/code/initialization/environment/qualification')
    if (not 1 <= receipt.get('epochs',0) <= epochs or receipt.get('max_epochs')!=epochs
        or receipt.get('selection') != 'source_dev_pooled_mean_probability_auc_ema'
        or not 1 <= receipt.get('best_epoch',0) <= receipt['epochs']):
        raise ValueError('formal checkpoint epoch/selection mismatch')
    if receipt.get('target_used') is not False:
        raise ValueError('formal completion provenance does not certify source-only selection')
    final_hash = _required_hash(final_path, 'formal best EMA checkpoint')
    if receipt.get('best_sha256') != final_hash:
        raise ValueError('formal checkpoint hash mismatch')
    base = {
        'schema': SCHEMA, 'config_path': str(config_path), 'config_sha256': config_hash,
        'artifact_root': str(root), 'run_id': run_id, 'backbone': backbone,
        'dimension': DIMENSIONS[backbone], 'epochs': receipt['epochs'], 'best_epoch':receipt['best_epoch'],
        'training_identity': identity, 'checkpoint_path': str(final_path),
        'checkpoint_sha256': final_hash, 'training_complete_sha256': receipt_hash,
        'cache_code_sha256': _cache_code_hash(),
    }
    return base, config, task


def _role_identity(base, role, manifest):
    manifest = Path(manifest).resolve()
    if role not in ROLES or manifest != Path(base['artifact_root']) / f'data/D/{role}.parquet':
        raise ValueError('unregistered D manifest role identity')
    return dict(base, role=role, source_manifest_path=str(manifest),
                source_manifest_sha256=_required_hash(manifest, 'role manifest'))


def _validate_live(identity):
    base, _, _ = _context(identity['config_path'], identity['run_id'])
    current = _role_identity(base, identity['role'], identity['source_manifest_path'])
    if identity != current:
        raise ValueError('cache live provenance identity/hash mismatch')


def _read_manifest(path):
    frame = pd.read_parquet(path)
    if frame.empty or not {'image_path', 'label'}.issubset(frame):
        raise ValueError('role manifest must contain nonempty image_path/label rows')
    if frame[['image_path', 'label']].isna().any().any() or not set(frame.label.unique()) <= {0, 1}:
        raise ValueError('invalid role manifest image/label identity')
    if 'sample_id' in frame and (frame.sample_id.isna().any() or frame.sample_id.duplicated().any()):
        raise ValueError('duplicate or missing role manifest sample identity')
    _validate_inputs(frame)
    return frame


def _validate_inputs(frame):
    """Check the frozen audit's stat fingerprint; rehash changed/missing stats.

    The D manifest builder freshly audits stale byte/RGB records and freezes
    size/mtime with each digest. Reusing its unchanged fingerprint avoids
    rereading the complete image corpus for every cache verification. Changed
    files must still reproduce the frozen byte hash before they can be used.
    """
    if 'content_sha256' not in frame:
        return
    columns = ['image_path', 'content_sha256']
    have_stat = {'bytes', 'mtime_ns'}.issubset(frame)
    if have_stat:
        columns += ['bytes', 'mtime_ns']
    unique = frame[columns].drop_duplicates()
    if unique.image_path.duplicated().any():
        raise ValueError('input path has conflicting frozen hash identity')
    for row in unique.itertuples(index=False):
        path = Path(row.image_path)
        try:
            stat = path.stat()
        except OSError as error:
            raise ValueError(f'input image identity is unavailable: {path}') from error
        unchanged = have_stat and stat.st_size == row.bytes and stat.st_mtime_ns == row.mtime_ns
        if not unchanged and trainer.digest(path) != row.content_sha256:
            raise ValueError(f'input image byte hash identity mismatch: {path}')


def _row_hash(frame, start, rows):
    # Column order and row order are part of the alignment record. The entire
    # exact Parquet file is separately protected by SHA256, including dtypes.
    text = frame.iloc[start:start + rows].to_json(orient='split', index=False, date_format='iso', double_precision=15)
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_features(x, rows, dimension):
    if x.ndim != 2 or x.shape != (rows, dimension) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('feature shape/dtype/dimension/finiteness mismatch')
    return int((np.linalg.norm(x.astype(np.float64), axis=1) < EPSILON).sum())


def _validate_head(weight, bias, dimension):
    w, b = np.asarray(weight), np.asarray(bias)
    if (w.shape != (2, dimension) or b.shape != (2,) or w.dtype != np.float32 or b.dtype != np.float32
            or not np.isfinite(w).all() or not np.isfinite(b).all()):
        raise ValueError('original two-output FP32 head shape/dtype/finiteness mismatch')
    return w, b


def _atomic_array(path, x):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.save(stream, x, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _read_chunks(root, frame, identity, chunk_rows):
    chunks = []
    position = 0
    markers = sorted(Path(root).glob('chunk_*.json'))
    for number, marker_path in enumerate(markers):
        name = f'chunk_{number:06d}'
        if marker_path.name != name + '.json':
            raise ValueError('committed chunks are not contiguous')
        item = json.loads(marker_path.read_text())
        expected_rows = min(chunk_rows, len(frame) - position)
        if (expected_rows <= 0 or item.get('file') != name + '.npy' or item.get('start') != position
                or item.get('rows') != expected_rows or item.get('dimension') != identity['dimension']
                or item.get('dtype') != 'float32' or item.get('identity_sha256') != _json_hash(identity)):
            raise ValueError('committed chunk identity/row/dimension mismatch')
        if item.get('row_sha256') != _row_hash(frame, position, expected_rows):
            raise ValueError('committed chunk manifest row hash mismatch')
        path = Path(root) / item['file']
        if _required_hash(path, 'committed feature chunk') != item.get('sha256'):
            raise ValueError('committed feature chunk hash mismatch')
        x = np.load(path, mmap_mode='r', allow_pickle=False)
        zeros = _validate_features(x, expected_rows, identity['dimension'])
        if item.get('zero_norm_count') != zeros:
            raise ValueError('committed feature chunk zero count mismatch')
        chunks.append(item)
        position += expected_rows
    return chunks, position


def _cache_role(root, manifest, identity, weight, bias, feature_batches, chunk_rows=4096):
    """Write/resume one role. feature_batches(start) must start at that row."""
    if not isinstance(chunk_rows, int) or chunk_rows < 4096:
        raise ValueError('cache chunks must contain at least 4096 rows, except the final chunk')
    root, manifest = Path(root), Path(manifest)
    dimension = identity['dimension']
    weight, bias = _validate_head(weight, bias, dimension)
    if _required_hash(manifest, 'role manifest') != identity['source_manifest_sha256']:
        raise ValueError('source role manifest hash mismatch')
    frame = _read_manifest(manifest)
    if 'role' in frame and not frame.role.eq(identity['role']).all():
        raise ValueError('role manifest identity mismatch')
    root.mkdir(parents=True, exist_ok=True)
    header = root / 'identity.json'
    if header.exists() and json.loads(header.read_text()) != identity:
        raise ValueError('resumed cache identity mismatch')
    trainer.save_json(header, identity)
    cached_manifest = root / 'manifest.parquet'
    if cached_manifest.exists():
        if trainer.digest(cached_manifest) != identity['source_manifest_sha256']:
            raise ValueError('cached manifest hash mismatch')
    else:
        temp = root / 'manifest.parquet.tmp'
        shutil.copyfile(manifest, temp)
        os.replace(temp, cached_manifest)
    head = {}
    for key, x in [('weight', weight), ('bias', bias)]:
        path = root / f'head_{key}.npy'
        if path.exists():
            saved = np.load(path, allow_pickle=False)
            if saved.dtype != np.float32 or not np.array_equal(saved, x):
                raise ValueError('resumed original head identity mismatch')
        else:
            _atomic_array(path, x)
        head[key + '_file'] = path.name
        head[key + '_sha256'] = trainer.digest(path)
    chunks, position = _read_chunks(root, frame, identity, chunk_rows)
    start = position
    pending = []
    pending_rows = 0

    def commit(x):
        nonlocal position
        number = len(chunks)
        name = f'chunk_{number:06d}.npy'
        zeros = _validate_features(x, len(x), dimension)
        item = dict(file=name, start=position, rows=len(x), dimension=dimension, dtype='float32',
                    identity_sha256=_json_hash(identity), row_sha256=_row_hash(frame, position, len(x)),
                    zero_norm_count=zeros)
        _atomic_array(root / name, x)
        item['sha256'] = trainer.digest(root / name)
        trainer.save_json(root / f'chunk_{number:06d}.json', item)
        chunks.append(item)
        position += len(x)

    if start < len(frame):
        for batch in feature_batches(start):
            x = np.asarray(batch)
            _validate_features(x, len(x), dimension)
            if len(x) == 0 or position + pending_rows + len(x) > len(frame):
                raise ValueError('feature batches contain empty or excessive rows')
            pending.append(x)
            pending_rows += len(x)
            while pending_rows >= chunk_rows:
                block = np.concatenate(pending, axis=0)
                commit(block[:chunk_rows])
                remaining = block[chunk_rows:]
                pending = [remaining] if len(remaining) else []
                pending_rows = len(remaining)
        if position + pending_rows != len(frame):
            raise ValueError('feature iterator ended before all manifest rows')
        if pending_rows:
            commit(np.concatenate(pending, axis=0))
    if _required_hash(manifest, 'role manifest') != identity['source_manifest_sha256']:
        raise ValueError('source manifest changed during feature extraction')
    meta = dict(schema=SCHEMA, status='CACHE_COMPLETE', identity=identity, rows=len(frame), dimension=dimension,
                precision='FP32', autocast=False, epsilon=EPSILON,
                zero_norm_count=sum(item['zero_norm_count'] for item in chunks), chunk_rows=chunk_rows,
                cached_manifest_sha256=trainer.digest(cached_manifest), head=head, chunks=chunks)
    trainer.save_json(root / 'complete.json', meta)
    return meta


def verify_cache(root, identity=None):
    """Reject changed live identities, incomplete/corrupt data, and row drift."""
    root = Path(root)
    if not (root / 'complete.json').is_file():
        raise ValueError('feature cache is incomplete: completion marker missing')
    meta = json.loads((root / 'complete.json').read_text())
    if meta.get('schema') != SCHEMA or meta.get('status') != 'CACHE_COMPLETE':
        raise ValueError('incomplete or unknown feature cache schema')
    saved_identity = meta['identity']
    if identity is not None and identity != saved_identity:
        raise ValueError('requested feature cache identity mismatch')
    _validate_live(saved_identity)
    header = root / 'identity.json'
    if not header.is_file() or json.loads(header.read_text()) != saved_identity:
        raise ValueError('cached identity header mismatch')
    manifest = root / 'manifest.parquet'
    if _required_hash(manifest, 'cached manifest') != meta.get('cached_manifest_sha256'):
        raise ValueError('cached manifest hash mismatch')
    if meta['cached_manifest_sha256'] != saved_identity['source_manifest_sha256']:
        raise ValueError('cached and live role manifest hash mismatch')
    frame = _read_manifest(manifest)
    if 'role' in frame and not frame.role.eq(saved_identity['role']).all():
        raise ValueError('cached manifest role identity mismatch')
    dimension = saved_identity['dimension']
    chunk_rows = meta.get('chunk_rows')
    if not isinstance(chunk_rows, int) or chunk_rows < 4096:
        raise ValueError('invalid chunk row budget')
    chunks, rows = _read_chunks(root, frame, saved_identity, chunk_rows)
    if chunks != meta.get('chunks') or rows != len(frame) or rows != meta.get('rows'):
        raise ValueError('feature cache chunk/row completion mismatch')
    if (meta.get('dimension') != dimension or meta.get('precision') != 'FP32' or meta.get('autocast') is not False
            or meta.get('epsilon') != EPSILON or meta.get('zero_norm_count') != sum(q['zero_norm_count'] for q in chunks)):
        raise ValueError('feature cache precision/dimension/zero-count mismatch')
    head = meta['head']
    arrays = []
    for key in ('weight', 'bias'):
        if head.get(key + '_file') != f'head_{key}.npy':
            raise ValueError('unsafe original head path')
        path = root / head[key + '_file']
        if _required_hash(path, 'original head') != head.get(key + '_sha256'):
            raise ValueError('original head hash mismatch')
        arrays.append(np.load(path, allow_pickle=False))
    _validate_head(*arrays, dimension)
    return meta, frame


def load_features(root, identity=None):
    """Return (manifest, raw_FP32_h, original_FP32_W[2,d], original_FP32_b[2])."""
    root = Path(root)
    meta, frame = verify_cache(root, identity)
    h = np.concatenate([np.load(root / item['file'], allow_pickle=False) for item in meta['chunks']], axis=0)
    w = np.load(root / meta['head']['weight_file'], allow_pickle=False)
    b = np.load(root / meta['head']['bias_file'], allow_pickle=False)
    return frame, h, w, b


def extract(config_path, run_id, device='cuda'):
    """Hold one run lock and, for CUDA, one approved GPU for the full export.

    Ownership is established before model construction or CUDA initialization.
    CPU is the only device that bypasses GPU reservation for unit checks.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id) or run_id in ('.', '..'):
        raise ValueError('unsafe run identity')
    config = json.loads(Path(config_path).read_text())
    root = Path(config['artifact_root']).resolve()
    with trainer.exclusive_file(root / 'features/D' / run_id / 'extract.lock'):
        ownership = contextlib.nullcontext() if torch.device(device).type == 'cpu' else trainer.owned_gpu(root, config)
        with ownership:
            return _extract(config_path, run_id, device)


def _extract(config_path, run_id, device='cuda'):
    """Export available D roles while the caller holds lifetime ownership."""
    from .models import build_model

    base, config, task = _context(config_path, run_id)
    root = Path(base['artifact_root'])
    checkpoint = torch.load(base['checkpoint_path'], map_location='cpu', weights_only=False)
    if checkpoint.get('identity') != base['training_identity']:
        raise ValueError('loaded best EMA checkpoint identity mismatch')
    if checkpoint.get('epoch') != base['best_epoch'] or checkpoint.get('weights')!='ema':
        raise ValueError('loaded best EMA checkpoint epoch mismatch')
    weight = checkpoint['model']['classifier.weight'].detach().cpu().numpy()
    bias = checkpoint['model']['classifier.bias'].detach().cpu().numpy()
    _validate_head(weight, bias, base['dimension'])
    chunk_rows = config.get('cache_chunk_rows', 4096)
    if not isinstance(chunk_rows, int) or chunk_rows < 4096:
        raise ValueError('cache chunk rows must be at least 4096')
    batch = config.get('cache_batch_size', task['microbatch'])
    workers = config.get('cache_workers', config.get('workers', 4))
    if not isinstance(batch, int) or batch < 1 or not isinstance(workers, int) or workers < 0:
        raise ValueError('invalid cache loader batch/workers')
    roles = [role for role in ROLES if (root / f'data/D/{role}.parquet').is_file()]
    if not {'train', 'dev'}.issubset(roles):
        raise ValueError('D source manifests are incomplete')
    identities = {role: _role_identity(base, role, root / f'data/D/{role}.parquet') for role in roles}
    model = None
    trainer.policy()
    out = root / 'features/D' / run_id
    outputs = {}
    for role in roles:
        role_root = out / role
        if (role_root / 'complete.json').is_file():
            meta, _ = verify_cache(role_root, identities[role])
        else:
            if model is None:
                model = build_model(task['backbone'], Path(config['backbones'][task['backbone']]['weight_path']))
                model.load_state_dict(checkpoint['model'], strict=True)
                model = model.to(device=device, dtype=torch.float32).eval()
            frame = _read_manifest(identities[role]['source_manifest_path'])

            def feature_batches(start, frame=frame):
                data = trainer.loader(frame.iloc[start:].reset_index(drop=True), task['backbone'], batch, workers=workers)
                with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
                    for x, _ in data:
                        h = model.encoder(x.to(device=device, dtype=torch.float32))
                        if h.dtype != torch.float32:
                            raise ValueError('encoder must export original FP32 features without autocast')
                        yield h.detach().cpu().numpy().copy()

            meta = _cache_role(role_root, identities[role]['source_manifest_path'], identities[role],
                               weight, bias, feature_batches, chunk_rows)
            verify_cache(role_root, identities[role])
        outputs[role] = dict(path=str(role_root), rows=meta['rows'],
                             complete_sha256=trainer.digest(role_root / 'complete.json'))
    # A concurrently changed config, model, code, or data invalidates completion.
    final_base, _, _ = _context(config_path, run_id)
    if final_base != base:
        raise ValueError('formal provenance changed during extraction')
    for role in roles:
        if _role_identity(base, role, identities[role]['source_manifest_path']) != identities[role]:
            raise ValueError('role manifest provenance changed during extraction')
    result = dict(status='CACHE_COMPLETE', run_id=run_id, identity=base, roles=outputs)
    trainer.save_json(out / 'complete.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(json.dumps(extract(args.config, args.run_id, args.device), ensure_ascii=False, allow_nan=False))
