"""FP32 H evaluation caches with live, fail-closed five-pair authorization.

Each of the seven accepted H evaluation tables has its own canonical cache at
artifact_root/features/H/<run_id>/<role>. Features and ORIGINAL two-output head
remain FP32; scoring promotes both head rows before subtraction. No source CAL
AUC is calculated here.

Every verification freshly hashes processed PNG bytes and decoded RGB, even if
stat metadata is absent or unchanged. Source-video bytes are covered by accepted
freeze evidence; the default freeze validator rechecks metadata, not video
bytes. This module does not claim a fresh source-video integrity scan.

Committed chunk markers follow atomic array writes. Unmarked partial arrays
are overwritten on resume; committed corruption is rejected. A completed cache
is verified and reused without constructing an encoder. CUDA export uses the
original trainer's lifetime GPU ownership and fresh H/D-priority boundaries.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import cache_d, data_h, queue_h, trainer, worker_h

DIMENSIONS = {'clip_vit_b16': 512, 'dinov2_vit_b14': 768, 'efficientnet_b0': 1280}
ROLES = ('B/SOURCE_CAL', 'B/SOURCE_TEST', 'B/TARGET_TEST',
         'common/B/SOURCE_TEST', 'common/B/TARGET_TEST',
         'common/AB/SOURCE_TEST', 'common/AB/TARGET_TEST')
SCHEMA = 'pipeline-reliability-v2-H-fp32-cache-1'
EPSILON = 1e-12
INPUT_VALIDATION = 'FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES'
# These helpers operate on immutable arrays/row hashes only. No D context,
# manifest reader, schema validation or extraction is used.
_json_hash = cache_d._json_hash
_row_hash = cache_d._row_hash
_atomic_array = cache_d._atomic_array
_validate_features = cache_d._validate_features
_validate_head = cache_d._validate_head
_read_chunks = cache_d._read_chunks


def _required_hash(path, description):
    if not Path(path).is_file():
        raise ValueError(f'missing {description}; provenance is incomplete: {path}')
    return trainer.digest(path)


def _cache_code_hash():
    return trainer.digest(__file__)


def _helper_code_hash():
    return trainer.digest(cache_d.__file__)


def _head_hash(weight, bias, dimension):
    weight, bias = _validate_head(weight, bias, dimension)
    return hashlib.sha256(weight.tobytes(order='C') + bias.tobytes(order='C')).hexdigest()


def _read_json(path, description):
    _required_hash(path, description)
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('invalid ' + description) from error


def _expected_runs(backbone, precision):
    return [queue_h.formal_id(backbone, history, seed, precision)
            for seed in queue_h.FORMAL_SEEDS for history in queue_h.HISTORIES]


def _target_authorization(config_path, config, task, run_id):
    """Recheck the immutable ledger AND all five current sealed paired streams."""
    root = Path(config['artifact_root']).resolve()
    backbone, precision = task['backbone'], task['precision']
    gate_path = root / f'qualification/H_target_export_{backbone}.json'
    gate = _read_json(gate_path, 'H target export gate')
    expected = _expected_runs(backbone, precision)
    checks = dict(status='PASS', backbone=backbone, precision=precision,
                  all_five_pairs_checked=True, effective_runs=expected,
                  H_data_freeze_sha256=config['H_data_freeze_sha256'],
                  h_code_bundle_sha256=config['h_code_bundle_sha256'],
                  target_performance_used=False)
    if any(gate.get(k) != v for k, v in checks.items()) or run_id not in expected:
        raise ValueError('H target export gate effective run/precision/five pairs identity mismatch')
    registry_path = root / 'H_run_registry.json'
    registry = _read_json(registry_path, 'H final run registry')
    formal = [queue_h.formal_id(b, h, s) for b in queue_h.BACKBONES
              for s in queue_h.FORMAL_SEEDS for h in queue_h.HISTORIES]
    if (registry.get('schema') != 'H_RUN_REGISTRY_V2'
            or registry.get('status') != 'H_FORMAL_COMPLETE_TARGET_CACHE_PENDING'
            or registry.get('target_export_allowed') is not True
            or registry.get('formal_model_count') != 30
            or set(registry.get('formal_run_ids', [])) != set(formal)
            or len(registry.get('formal_run_ids', [])) != 30
            or set(registry.get('effective_runs', {})) != set(queue_h.BACKBONES)):
        raise ValueError('H final registry is incomplete or target export is disallowed')
    for b, item in registry['effective_runs'].items():
        if item.get('precision') not in ('AMP', 'FP32') or item.get('run_ids') != _expected_runs(b, item['precision']):
            raise ValueError('H registry effective run identity mismatch')
    if registry['effective_runs'][backbone] != dict(precision=precision, run_ids=expected):
        raise ValueError('H registry effective precision supersedes this model')
    ledger_path = root / f'reports/H_{backbone}_{precision}_paired_ledger.json'
    if (Path(gate.get('paired_ledger_path', '')).resolve() != ledger_path
            or _required_hash(ledger_path, 'H paired ledger') != gate.get('paired_ledger_sha256')):
        raise ValueError('H paired ledger path/hash mismatch')
    saved = _read_json(ledger_path, 'H paired ledger')
    fresh = queue_h.audit_paired_ledgers(config_path, backbone, precision=precision)
    if ({k: v for k, v in saved.items() if k != 'utc'} != {k: v for k, v in fresh.items() if k != 'utc'}
            or saved.get('status') != 'MATCHED_SUCCESSFUL_UPDATE_STREAMS'
            or saved.get('formal_seeds') != list(queue_h.FORMAL_SEEDS)
            or len(saved.get('pairs', [])) != 5
            or saved.get('asymmetric_seed_count') != 0
            or saved.get('fallback_pair_count') != 0
            or saved.get('before_target_export') is not True
            or saved.get('target_performance_used') is not False):
        raise ValueError('H five pairs current numerical ledger identity mismatch')
    return dict(target_export_gate_sha256=trainer.digest(gate_path),
                paired_ledger_path=str(ledger_path), paired_ledger_sha256=trainer.digest(ledger_path),
                H_run_registry_sha256=trainer.digest(registry_path), effective_precision=precision,
                effective_run_ids=expected)


def _load_final(base, task):
    state = torch.load(base['checkpoint_path'], map_location='cpu', weights_only=False)
    if state.get('identity') != base['training_identity'] or state.get('epoch') != task['epochs']:
        raise ValueError('H final checkpoint identity/epoch mismatch')
    try:
        weight = state['model']['classifier.weight'].detach().cpu().numpy()
        bias = state['model']['classifier.bias'].detach().cpu().numpy()
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError('H final original head identity missing') from error
    _validate_head(weight, bias, base['dimension'])
    if _required_hash(base['checkpoint_path'], 'H final checkpoint') != base['checkpoint_sha256']:
        raise ValueError('H final checkpoint changed while loaded')
    return state, weight, bias


def _context(config_path, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'H_[A-Za-z0-9_.-]+', run_id):
        raise ValueError('unsafe H run identity')
    path = Path(config_path).resolve()
    config_hash = _required_hash(path, 'H configuration')
    config, task, training_identity = worker_h.validate_task(path, run_id)
    if task.get('phase') != 'formal' or task.get('recipe_frozen') is not True or task['backbone'] not in DIMENSIONS:
        raise ValueError('H cache requires a frozen formal model; development inference is ineligible')
    receipt = worker_h.verify_run_completion(path, run_id)
    authorization = _target_authorization(path, config, task, run_id)
    root = Path(config['artifact_root']).resolve()
    run_root = root / 'runs' / run_id
    final = run_root / 'final.pt'
    if run_root.resolve() != run_root or final.resolve() != final:
        raise ValueError('H checkpoint must be canonical local final.pt')
    final_hash = _required_hash(final, 'H formal final checkpoint')
    if final_hash != receipt.get('final_sha256'):
        raise ValueError('H formal final checkpoint hash mismatch')
    base = dict(schema=SCHEMA, config_path=str(path), config_sha256=config_hash, artifact_root=str(root),
        run_id=run_id, backbone=task['backbone'], history=task['history'], dimension=DIMENSIONS[task['backbone']],
        epochs=task['epochs'], training_precision=task['precision'], training_identity=training_identity,
        checkpoint_path=str(final), checkpoint_sha256=final_hash,
        training_complete_sha256=_required_hash(run_root / 'complete.json', 'H completion receipt'),
        completion_seal_sha256=_required_hash(run_root / 'H_completion_seal.json', 'H completion seal'),
        H_data_freeze_sha256=_required_hash(root / 'data/H/freeze.json', 'H freeze'),
        H_qualification_sha256=_required_hash(root / 'qualification/H_gate.json', 'H qualification gate'),
        H_recipe_sha256=_required_hash(root / 'configs/H_frozen_recipes.json', 'H recipe'),
        training_code_sha256=trainer.code_hash(), h_code_bundle_sha256=worker_h.h_code_hash(),
        environment_sha256=_required_hash(root / 'configs/requirements.execution.lock.txt', 'environment lock'),
        cache_code_sha256=_cache_code_hash(), cache_helper_code_sha256=_helper_code_hash(),
        input_validation=INPUT_VALIDATION, **authorization)
    _, weight, bias = _load_final(base, task)
    base['original_head_sha256'] = _head_hash(weight, bias, base['dimension'])
    if trainer.digest(path) != config_hash:
        raise ValueError('H config changed during context validation')
    return base, config, task


def _role_identity(base, role, manifest):
    root = Path(base['artifact_root'])
    expected = root / f'data/H/{role}.parquet'
    if role not in ROLES or Path(manifest).resolve() != expected:
        raise ValueError('unregistered accepted H role identity')
    freeze = _read_json(root / 'data/H/freeze.json', 'H freeze')
    item = freeze.get('tables', {}).get(role)
    if not item or Path(item['path']).resolve() != expected or _required_hash(expected, 'H role table') != item['sha256']:
        raise ValueError('accepted H role table path/hash identity mismatch')
    return dict(base, role=role, source_manifest_path=str(expected), source_manifest_sha256=item['sha256'],
                accepted_table_rows=item['rows'], accepted_table_videos=item['videos'])


def _canonical_cache(root, identity):
    expected = Path(identity['artifact_root']) / 'features/H' / identity['run_id'] / identity['role']
    if Path(root).resolve() != expected:
        raise ValueError('H cache must use its canonical run/role location')


def _validate_live(identity):
    base, _, _ = _context(identity['config_path'], identity['run_id'])
    current = _role_identity(base, identity['role'], identity['source_manifest_path'])
    if identity != current:
        raise ValueError('H cache live provenance identity/hash mismatch')


def _validate_inputs(frame):
    """Fresh byte AND decoded RGB verification; no stat-only acceptance path."""
    columns = ['image_path', 'content_sha256', 'rgb_sha256']
    unique = frame[columns].drop_duplicates()
    if unique.image_path.duplicated().any():
        raise ValueError('H input path has conflicting byte/RGB hash identity')
    for row in unique.itertuples(index=False):
        path = Path(row.image_path)
        try:
            before = path.stat()
            if trainer.digest(path) != row.content_sha256:
                raise ValueError('H input image byte hash mismatch: ' + str(path))
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.format != 'PNG':
                    raise ValueError('H input image must be saved RGB PNG: ' + str(path))
                rgb = np.ascontiguousarray(image, dtype=np.uint8)
            value = hashlib.sha256(str(rgb.shape).encode() + b'\0' + rgb.tobytes()).hexdigest()
            if value != row.rgb_sha256:
                raise ValueError('H input image RGB hash mismatch: ' + str(path))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError('H input image changed during validation')
        except OSError as error:
            raise ValueError('H input image unavailable: ' + str(path)) from error


def _read_manifest(path, role):
    frame = pd.read_parquet(path)
    data_h._table_check(role, frame, set())
    expected_domain = 'target' if role.endswith('/TARGET_TEST') else 'source'
    if not frame.domain.eq(expected_domain).all():
        raise ValueError('H role requires exact source/target domain identity')
    for name in ('content_sha256', 'rgb_sha256', 'mother_rgb_sha256'):
        if not frame[name].map(lambda value: isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None).all():
            raise ValueError('H image hash identity is missing or malformed')
    _validate_inputs(frame)
    return frame


def _capacity(root, identities):
    # Estimate all remaining FP32 rows, conservatively including partial caches,
    # then preserve the frozen plan's 50GiB free reserve before the first export.
    estimated = sum(i['accepted_table_rows'] * i['dimension'] * 4 + (2 * i['dimension'] + 2) * 4
                    + Path(i['source_manifest_path']).stat().st_size + 65536 for i in identities.values())
    remaining = sum(i['accepted_table_rows'] * i['dimension'] * 4 + (2 * i['dimension'] + 2) * 4
                    + Path(i['source_manifest_path']).stat().st_size + 65536
                    for role, i in identities.items()
                    if not (Path(root) / 'features/H' / i['run_id'] / role / 'complete.json').is_file())
    if remaining and shutil.disk_usage(root).free <= 50 * 1024**3 + remaining:
        raise ValueError('H cache capacity would violate the 50GiB reserve')
    return dict(initial_estimated_bytes=estimated, remaining_estimated_bytes=remaining, reserve_GiB=50)


def _cache_role(root, manifest, identity, weight, bias, feature_batches, chunk_rows=4096):
    """Write/resume one role. feature_batches(start) must start at that row."""
    if not isinstance(chunk_rows, int) or chunk_rows < 4096:
        raise ValueError('cache chunks must contain at least 4096 rows, except the final chunk')
    root, manifest = Path(root), Path(manifest)
    _canonical_cache(root, identity)
    if _head_hash(weight, bias, identity['dimension']) != identity['original_head_sha256']:
        raise ValueError('H original checkpoint head identity mismatch')
    dimension = identity['dimension']
    weight, bias = _validate_head(weight, bias, dimension)
    if _required_hash(manifest, 'role manifest') != identity['source_manifest_sha256']:
        raise ValueError('source role manifest hash mismatch')
    frame = _read_manifest(manifest, identity['role'])
    if 'role' in frame and not frame.role.eq(identity['role'].split('/')[-1]).all():
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
                precision='FP32', autocast=False, epsilon=EPSILON, input_validation=INPUT_VALIDATION,
                zero_norm_count=sum(item['zero_norm_count'] for item in chunks), chunk_rows=chunk_rows,
                cached_manifest_sha256=trainer.digest(cached_manifest), head=head, chunks=chunks)
    _validate_inputs(frame)
    trainer.save_json(root / 'complete.json', meta)
    return meta


def verify_cache(root, identity=None, *, expected_run_id=None, expected_role=None, config_path=None):
    root = Path(root)
    meta = _read_json(root / 'complete.json', 'H cache completion marker')
    if meta.get('schema') != SCHEMA or meta.get('status') != 'CACHE_COMPLETE':
        raise ValueError('incomplete or unknown H feature cache schema')
    saved = meta.get('identity', {})
    if identity is not None and identity != saved:
        raise ValueError('requested H cache identity mismatch')
    if ((expected_run_id is not None and saved.get('run_id') != expected_run_id)
            or (expected_role is not None and saved.get('role') != expected_role)
            or (config_path is not None and saved.get('config_path') != str(Path(config_path).resolve()))):
        raise ValueError('requested H run/role/config identity mismatch')
    if saved.get('schema') != SCHEMA or saved.get('role') not in ROLES:
        raise ValueError('invalid H cache identity schema/role')
    _canonical_cache(root, saved)
    _validate_live(saved)
    if _read_json(root / 'identity.json', 'H cache identity header') != saved:
        raise ValueError('H cache identity header mismatch')
    manifest = root / 'manifest.parquet'
    if (_required_hash(manifest, 'H cached manifest') != saved['source_manifest_sha256']
            or meta.get('cached_manifest_sha256') != saved['source_manifest_sha256']):
        raise ValueError('H cached manifest hash mismatch')
    frame = _read_manifest(manifest, saved['role'])
    if len(frame) != saved['accepted_table_rows'] or frame.video_uid.nunique() != saved['accepted_table_videos']:
        raise ValueError('H accepted table row/video identity mismatch')
    chunk_rows = meta.get('chunk_rows')
    if not isinstance(chunk_rows, int) or chunk_rows < 4096:
        raise ValueError('H cache chunk row budget invalid')
    chunks, rows = _read_chunks(root, frame, saved, chunk_rows)
    if chunks != meta.get('chunks') or rows != len(frame) or rows != meta.get('rows'):
        raise ValueError('H feature cache chunk/row completion mismatch')
    if (meta.get('dimension') != saved['dimension'] or meta.get('precision') != 'FP32'
            or meta.get('autocast') is not False or meta.get('epsilon') != EPSILON
            or meta.get('input_validation') != INPUT_VALIDATION
            or meta.get('zero_norm_count') != sum(c['zero_norm_count'] for c in chunks)):
        raise ValueError('H feature cache precision/dimension/zero-count mismatch')
    arrays = []
    head = meta.get('head', {})
    for key in ('weight', 'bias'):
        if head.get(key + '_file') != f'head_{key}.npy':
            raise ValueError('unsafe H original head path')
        path = root / head[key + '_file']
        if _required_hash(path, 'H original head') != head.get(key + '_sha256'):
            raise ValueError('H original head hash mismatch')
        arrays.append(np.load(path, allow_pickle=False))
    if _head_hash(*arrays, saved['dimension']) != saved['original_head_sha256']:
        raise ValueError('H original checkpoint head identity mismatch')
    return meta, frame


def load_features(root, identity=None, *, expected_run_id=None, expected_role=None, config_path=None):
    """Return exact manifest, raw FP32 h, ORIGINAL FP32 W[2,d] and b[2]."""
    root = Path(root)
    meta, frame = verify_cache(root, identity, expected_run_id=expected_run_id, expected_role=expected_role, config_path=config_path)
    h = np.concatenate([np.load(root / c['file'], allow_pickle=False) for c in meta['chunks']])
    w = np.load(root / meta['head']['weight_file'], allow_pickle=False)
    b = np.load(root / meta['head']['bias_file'], allow_pickle=False)
    return frame, h, w, b


def extract(config_path, run_id, device='cuda'):
    if not isinstance(run_id, str) or not re.fullmatch(r'H_[A-Za-z0-9_.-]+', run_id):
        raise ValueError('unsafe H run identity')
    config = _read_json(config_path, 'H config')
    root = Path(config['artifact_root']).resolve()
    with trainer.exclusive_file(root / 'features/H' / run_id / 'extract.lock'):
        cpu = torch.device(device).type == 'cpu'
        if not cpu:
            worker_h.verify_resource_boundary(config)
        with contextlib.nullcontext() if cpu else trainer.owned_gpu(root, config):
            if not cpu:
                worker_h.verify_resource_boundary(config)
            return _extract(config_path, run_id, device)


def _extract(config_path, run_id, device='cuda'):
    from .models import build_model
    base, config, task = _context(config_path, run_id)
    identities = {r: _role_identity(base, r, Path(base['artifact_root']) / f'data/H/{r}.parquet') for r in ROLES}
    root = Path(base['artifact_root'])
    _capacity(root, identities)
    # Validate every evaluation input before any encoder allocation.
    frames = {r: _read_manifest(i['source_manifest_path'], r) for r, i in identities.items()}
    checkpoint, weight, bias = _load_final(base, task)
    if _head_hash(weight, bias, base['dimension']) != base['original_head_sha256']:
        raise ValueError('H original head changed before extraction')
    chunk_rows = config.get('cache_chunk_rows', 4096)
    batch = config.get('cache_batch_size', task['microbatch'])
    workers = config.get('cache_workers', config.get('workers', 4))
    if (not isinstance(chunk_rows, int) or chunk_rows < 4096 or not isinstance(batch, int) or batch < 1
            or not isinstance(workers, int) or workers < 0):
        raise ValueError('invalid H cache chunk/batch/workers configuration')
    trainer.policy()
    model = None
    outputs = {}
    for role, identity in identities.items():
        folder = root / 'features/H' / run_id / role
        if (folder / 'complete.json').exists():
            meta, _ = verify_cache(folder, identity)
        else:
            if model is None:
                if torch.device(device).type != 'cpu':
                    worker_h.verify_resource_boundary(config)
                model = build_model(task['backbone'], Path(config['backbones'][task['backbone']]['weight_path']))
                model.load_state_dict(checkpoint['model'], strict=True)
                model = model.float().to(device=device, dtype=torch.float32).eval()
            frame = frames[role]
            def batches(start, frame=frame):
                data = trainer.loader(frame.iloc[start:].reset_index(drop=True), task['backbone'], batch, workers=workers)
                with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
                    for x, _ in data:
                        h = model.encoder(x.to(device=device, dtype=torch.float32))
                        if h.dtype != torch.float32:
                            raise ValueError('H encoder must export raw FP32 features without autocast')
                        yield h.detach().cpu().numpy().copy()
            meta = _cache_role(folder, identity['source_manifest_path'], identity, weight, bias, batches, chunk_rows)
            verify_cache(folder, identity)
        outputs[role] = dict(path=str(folder), rows=meta['rows'], complete_sha256=trainer.digest(folder / 'complete.json'))
    current, _, _ = _context(config_path, run_id)
    if current != base:
        raise ValueError('H formal provenance changed during extraction')
    for role, identity in identities.items():
        if _role_identity(base, role, identity['source_manifest_path']) != identity:
            raise ValueError('H role manifest changed during extraction')
    result = dict(status='CACHE_COMPLETE', run_id=run_id, identity=base, roles=outputs)
    trainer.save_json(root / 'features/H' / run_id / 'complete.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(json.dumps(extract(args.config, args.run_id, args.device), ensure_ascii=False, allow_nan=False))
