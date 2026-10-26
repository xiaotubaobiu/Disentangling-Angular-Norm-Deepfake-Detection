"""Fail-closed, source-only H qualification. Importing this module starts no jobs.

Only an external resource queue may invoke ``worker`` on one idle approved GPU.
``prepare_probe`` writes under qualification/H/probes only; production acceptance
requires independent AMP and FP32 evidence for both histories and all backbones.
"""
from pathlib import Path
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import struct
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from . import trainer

BACKBONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')
DIMENSIONS = dict(zip(BACKBONES, (512, 768, 1280)))
METHODS = ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures')
PRECISIONS = ('AMP', 'FP32')
MICROBATCHES = (32, 16, 8)
SCHEMA = 'H_SOURCE_QUALIFICATION_V1'
SEED = 1007
STATE_KEYS = ('model', 'optimizer', 'scaler', 'python_rng', 'numpy_rng',
    'torch_rng', 'cuda_rng', 'epoch', 'attempts', 'scheduler_clock', 'identity', 'history')
SMALL_EVIDENCE_LIMIT = 8 * 1024 * 1024


def qualification_code_hash():
    return trainer.digest(__file__)


def _h_code_hash():
    from .worker_h import h_code_hash
    return h_code_hash()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise ValueError(f'required qualification evidence missing: {path}') from error


def _context(root, *, deep=False):
    from .data_h import validate_h_freeze
    root = Path(root)
    freeze = validate_h_freeze(root, deep=deep)
    if freeze.get('status') != 'FROZEN':
        raise ValueError('accepted H freeze required')
    if not isinstance(freeze.get('training_draw_count'), int) or freeze['training_draw_count'] <= 0 or freeze['training_draw_count'] % 128:
        raise ValueError('invalid accepted H training draw count')
    config = _read(root / 'configs/base.json')
    if config.get('allowed_gpus') != [0, 1, 2, 3] or len(set(config.get('allowed_gpu_uuids', []))) != 4:
        raise ValueError('H qualification requires exactly approved GPU0-3 UUIDs')
    if config.get('workers') != 4 or config.get('effective_batch') != 128:
        raise ValueError('qualification requires four workers/effective128')
    architecture = root / 'qualification/gpu_gate.json'
    gate = _read(architecture)
    if gate.get('status') != 'PASS' or gate.get('training_code_sha256') != trainer.code_hash():
        raise ValueError('D architecture GPU gate missing or stale')
    for backbone in BACKBONES:
        if gate.get('backbones', {}).get(backbone, {}).get('status') != 'PASS':
            raise ValueError('architecture backbone qualification missing: ' + backbone)
        weight = config['backbones'][backbone]
        if trainer.digest(weight['weight_path']) != weight['weight_sha256']:
            raise ValueError('pretrained initialization byte hash mismatch')
    binding = dict(schema=SCHEMA, training_code_sha256=trainer.code_hash(),
        qualification_code_sha256=qualification_code_hash(), h_code_bundle_sha256=_h_code_hash(),
        H_data_freeze_sha256=trainer.digest(root / 'data/H/freeze.json'),
        architecture_gate_sha256=trainer.digest(architecture),
        base_config_sha256=trainer.digest(root / 'configs/base.json'),
        environment_sha256=trainer.digest(root / 'configs/requirements.execution.lock.txt'))
    return freeze, config, binding


def _source(frame, role, version):
    required = {'image_path', 'video_uid', 'method_id', 'label', 'domain', 'role',
        'mother_id', 'sample_id', 'mother_rgb_sha256', 'mother_frame_index',
        'mother_timestamp_seconds', 'group_id', 'processing_version', 'content_sha256', 'rgb_sha256'}
    if not required <= set(frame):
        raise ValueError('missing H source columns: ' + str(sorted(required - set(frame))))
    if frame.empty or not frame.domain.eq('source').all() or not frame.role.eq(role).all():
        raise ValueError('qualification requires source-only ' + role)
    if not frame.processing_version.eq(version).all():
        raise ValueError('wrong H processing history')
    if not frame.label.isin([0, 1]).all() or not frame.method_id.isin(['real', *METHODS]).all():
        raise ValueError('unregistered H method or label')
    if not frame.label.eq(frame.method_id.ne('real').astype(int)).all():
        raise ValueError('H method/label conflict')
    if frame.mother_id.isna().any() or frame.mother_id.duplicated().any() or not frame.sample_id.eq(frame.mother_id).all():
        raise ValueError('mother/sample identities must be unique and identical')
    if not frame.groupby('video_uid').size().eq(8).all():
        raise ValueError('source videos require exactly eight mother rows')
    for column in ['label', 'method_id', 'role', 'domain', 'group_id']:
        if not frame.groupby('video_uid')[column].nunique(dropna=False).eq(1).all():
            raise ValueError('inconsistent source video metadata')
    return frame.sort_values(['video_uid', 'mother_id']).reset_index(drop=True)


def _verify_bytes(frame):
    for row in frame.itertuples():
        path = Path(row.image_path)
        if not path.is_file() or trainer.digest(path) != row.content_sha256:
            raise ValueError('source image byte/content hash changed: ' + str(path))
        with Image.open(path) as image:
            rgb = np.ascontiguousarray(image.convert('RGB'), dtype=np.uint8)
        pixel_hash = hashlib.sha256(str(rgb.shape).encode() + b'\0' + rgb.tobytes()).hexdigest()
        if pixel_hash != row.rgb_sha256:
            raise ValueError('source image RGB hash changed: ' + str(path))
        stat = path.stat()
        if hasattr(row, 'bytes') and stat.st_size != row.bytes:
            raise ValueError('source image byte count changed')
        if hasattr(row, 'mtime_ns') and stat.st_mtime_ns != row.mtime_ns:
            raise ValueError('source image mtime changed')


def _paired_source_probes(b, ab, role):
    b, ab = _source(b, role, 'B'), _source(ab, role, 'AB')
    columns = ['mother_id', 'video_uid', 'method_id', 'label', 'domain', 'role',
        'mother_rgb_sha256', 'mother_frame_index', 'mother_timestamp_seconds', 'group_id']
    if not b[columns].equals(ab[columns]):
        raise ValueError('H source common B/AB mother metadata differs')
    selected = []
    for method in ['real', *METHODS]:
        count = (8 if method == 'real' else 2) if role == 'TRAIN' else (2 if method == 'real' else 1)
        videos = sorted(b.loc[b.method_id.eq(method), 'video_uid'].unique())
        if len(videos) < count:
            raise ValueError(f'insufficient source {role} videos for {method}')
        selected.extend(videos[:count])
    results = []
    for frame in [b, ab]:
        probe = frame[frame.video_uid.isin(selected)].copy().reset_index(drop=True)
        # Recompute the same protocol masses within the tiny source probe.
        counts = probe.groupby('method_id').video_uid.nunique().to_dict()
        probe['sampling_mass'] = [(.5 if m == 'real' else .125) / counts[m] / 8 for m in probe.method_id]
        probe['method_contributions'] = [json.dumps({m: mass}, sort_keys=True) for m, mass in zip(probe.method_id, probe.sampling_mass)]
        _verify_bytes(probe)
        results.append(probe)
    if role == 'TRAIN' and (len(results[0]) != 128 or results[0].label.value_counts().to_dict() != {0: 64, 1: 64}):
        raise ValueError('H TRAIN probe must have64real/64fake rows')
    return tuple(results)


def _relative(root, path):
    return str(Path(path).relative_to(root))


def _manifest(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    if not path.is_relative_to(root / 'qualification/H/probes'):
        raise ValueError('probe path escapes qualification/H/probes')
    if trainer.digest(path) != record['sha256']:
        raise ValueError('qualification probe manifest changed')
    frame = pd.read_parquet(path)
    if len(frame) != record['rows'] or frame.video_uid.nunique() != record['videos']:
        raise ValueError('qualification probe count changed')
    _verify_bytes(frame)
    return frame


def _probe_receipt(root, binding):
    receipt = _read(Path(root) / 'qualification/H/probes/receipt.json')
    if receipt.get('status') != 'PROBES_READY' or receipt.get('binding') != binding or receipt.get('target_used') is not False:
        raise ValueError('stale H qualification probe identity')
    bootstrap = _read(Path(root) / 'qualification/H/probes/bootstrap_gate.json')
    if bootstrap != dict(status='PASS', approved_routes=['H_QUALIFICATION'], source_only=True, target_used=False, **binding):
        raise ValueError('H qualification bootstrap gate identity mismatch')
    if set(receipt['tables']) != {'B/train', 'AB/train', 'B/dev', 'AB/dev', 'B/SOURCE_CAL'}:
        raise ValueError('wrong source probe manifest set')
    freeze = _read(Path(root) / 'data/H/freeze.json')
    if receipt.get('accepted_source_tables') != {k: freeze['tables'][k] for k in receipt['tables']}:
        raise ValueError('qualification probe accepted-source table identity changed')
    expected = _probe_frames(root, freeze)
    for key, record in receipt['tables'].items():
        pd.testing.assert_frame_equal(_manifest(root, record), expected[key])
    return receipt


def _probe_frames(root, freeze):
    root = Path(root)
    frames = {}
    for role in ['TRAIN', 'DEV']:
        paths = [freeze['tables'][f'{v}/{role.lower()}']['path'] for v in ['B', 'AB']]
        b, ab = _paired_source_probes(*(pd.read_parquet(root / p) for p in paths), role)
        frames[f'B/{role.lower()}'], frames[f'AB/{role.lower()}'] = b, ab
    cal = _source(pd.read_parquet(root / freeze['tables']['B/SOURCE_CAL']['path']), 'SOURCE_CAL', 'B')
    if not cal.label.eq(0).all() or not cal.method_id.eq('real').all():
        raise ValueError('B SOURCE_CAL requires real-only source videos')
    cal = cal[cal.video_uid.isin(sorted(cal.video_uid.unique())[:8])].copy().reset_index(drop=True)
    _verify_bytes(cal)
    frames['B/SOURCE_CAL'] = cal
    return frames


def prepare_probe(root):
    """Build accepted-source tiny probes, never data/H tables or a production gate."""
    root = Path(root)
    freeze, config, binding = _context(root)
    out = root / 'qualification/H/probes'
    if (out / 'receipt.json').exists():
        return _probe_receipt(root, binding)
    frames = _probe_frames(root, freeze)
    # All validation above precedes the first write.
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for key, frame in frames.items():
        path = out / (key.replace('/', '_') + '.parquet')
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
        else:
            frame.to_parquet(path, index=False)
        tables[key] = dict(path=_relative(root, path), sha256=trainer.digest(path), rows=len(frame), videos=int(frame.video_uid.nunique()))
    trainer.save_json(out / 'bootstrap_gate.json', dict(status='PASS', approved_routes=['H_QUALIFICATION'], source_only=True, target_used=False, **binding))
    receipt = dict(status='PROBES_READY', binding=binding, tables=tables, source_only=True, target_used=False,
        accepted_source_tables={k: freeze['tables'][k] for k in frames}, effective_batch=128, loader_workers=4,
        probe_draw_count=128, training_draw_count=freeze['training_draw_count'])
    trainer.save_json(out / 'receipt.json', receipt)
    return receipt


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _add_tree(h, value):
    if torch.is_tensor(value):
        h.update(b'tensor' + str(value.dtype).encode() + repr(tuple(value.shape)).encode() + _tensor_bytes(value))
    elif isinstance(value, np.ndarray):
        h.update(b'numpy' + value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes())
    elif isinstance(value, dict):
        h.update(b'dict')
        for key in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
            _add_tree(h, key); _add_tree(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(type(value).__name__.encode())
        for item in value: _add_tree(h, item)
    elif isinstance(value, float):
        h.update(b'float' + struct.pack('!d', value))
    else:
        h.update(type(value).__name__.encode() + repr(value).encode())


def tree_hash(value):
    h = hashlib.sha256(); _add_tree(h, value); return h.hexdigest()


def assert_byte_equal(a, b, path=''):
    if type(a) is not type(b):
        raise AssertionError('state type mismatch ' + path)
    if torch.is_tensor(a):
        good = a.dtype == b.dtype and a.shape == b.shape and _tensor_bytes(a) == _tensor_bytes(b)
    elif isinstance(a, np.ndarray):
        good = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise AssertionError('state keys mismatch ' + path)
        for key in a: assert_byte_equal(a[key], b[key], path + '/' + str(key))
        return
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): raise AssertionError('state sequence mismatch ' + path)
        for i, (x, y) in enumerate(zip(a, b)): assert_byte_equal(x, y, path + '/' + str(i))
        return
    elif isinstance(a, float):
        good = struct.pack('!d', a) == struct.pack('!d', b)
    else:
        good = a == b
    if not good: raise AssertionError('state byte mismatch ' + path)


def recompute_source_metrics(dev, cal):
    """Independent pair-count DEV AUC; real-only CAL never enters an AUC call."""
    from sklearn.metrics import roc_auc_score
    for frame in [dev, cal]:
        if frame.empty or frame.video_uid.duplicated().any() or not np.isfinite(frame.score.to_numpy()).all():
            raise ValueError('unique finite source video scores required')
    if set(dev.label) != {0, 1} or set(dev.loc[dev.label.eq(1), 'method_id']) != set(METHODS):
        raise ValueError('source DEV needs real and all four fake methods')
    if not cal.label.eq(0).all() or not cal.method_id.eq('real').all():
        raise ValueError('SOURCE_CAL must contain real-only scores')
    real = dev[dev.label.eq(0)]
    methods = {}
    for method in METHODS:
        frame = pd.concat([real, dev[dev.method_id.eq(method)]])
        value = trainer.auc_independent(frame.label, frame.score)
        if abs(value - roc_auc_score(frame.label, frame.score)) > 1e-12:
            raise ValueError('independent source DEV AUC mismatch')
        methods[method] = value
    pooled = trainer.auc_independent(dev.label, dev.score)
    if abs(pooled - roc_auc_score(dev.label, dev.score)) > 1e-12:
        raise ValueError('independent pooled source DEV AUC mismatch')
    # Derive the empirical boundary independently of trainer.empirical_threshold.
    values = np.sort(cal.score.to_numpy(dtype=np.float64))[::-1]
    limit = math.floor(.05 * len(values) + 1e-12)
    threshold = float(np.nextafter(values[limit], np.inf))
    if threshold != trainer.empirical_threshold(values):
        raise ValueError('independent real-only source threshold mismatch')
    fp = int(np.count_nonzero(values >= threshold))
    if fp > limit: raise ValueError('real-only empirical FPR constraint failed')
    return dict(dev=dict(macro_auc=float(np.mean(list(methods.values()))), pooled_auc=pooled, method_auc=methods),
        source_cal=dict(threshold=threshold, alpha=.05, videos=len(values), false_positives=fp,
            empirical_fpr=fp / len(values), rule='score>=threshold', auc_called=False))


def _export(model, frame, backbone, micro, device, path):
    model.eval(); parts = []
    with torch.inference_mode():
        for x, _ in trainer.loader(frame, backbone, micro, workers=4):
            parts.append(model.encoder(x.to(device)).float().cpu().numpy())
    value = np.concatenate(parts)
    if value.dtype != np.float32 or value.shape != (len(frame), DIMENSIONS[backbone]) or not np.isfinite(value).all():
        raise ValueError('invalid FP32 Raw feature export')
    np.save(path, value, allow_pickle=False)
    return value


def _video_scores(frame, features, weight, bias):
    margin = features.astype(np.float64) @ (weight[1].astype(np.float64) - weight[0].astype(np.float64)) + (np.float64(bias[1]) - np.float64(bias[0]))
    result = frame[['video_uid', 'method_id', 'label']].copy(); result['score'] = margin
    return result.groupby(['video_uid', 'method_id', 'label'], sort=True, as_index=False).score.mean()


def _run_history(frame, dev, cal, backbone, weight, identity, out, micro, precision,
                 device='cuda', model_factory=None):
    """Real trainer path, injectable only for CPU unit tests; not a public worker."""
    from .models import build_model
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model_factory = build_model if model_factory is None else model_factory
    trainer.policy(); trainer.set_random(SEED)
    lr = weight['learning_rate']
    cuda = str(device).startswith('cuda')
    if precision == 'AMP' and not cuda: raise ValueError('AMP qualification requires CUDA')

    def make():
        model = model_factory(backbone, weight['weight_path']).to(device).float()
        if any(not p.requires_grad for p in model.parameters()):
            raise ValueError('all encoder/head parameters must be trainable')
        if tuple(model.classifier.weight.shape) != (2, DIMENSIONS[backbone]):
            raise ValueError('Raw feature/head dimension mismatch')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scaler = torch.amp.GradScaler('cuda', init_scale=128) if precision == 'AMP' else None
        return model, optimizer, scaler

    def epoch(model, optimizer, scaler, number, prefix):
        model.train(); draws, methods = trainer.epoch_draws(frame, SEED, number, 128)
        augmentations = np.asarray([[p['flip'], p['brightness'], p['contrast']] for p in (
            trainer.augmentation_parameters(SEED, number, str(frame.iloc[int(index)].mother_id), i)
            for i, index in enumerate(draws))], dtype=np.float64)
        for name, value in [('samples', draws), ('methods', methods), ('augmentations', augmentations)]:
            np.save(out / f'{prefix}_{name}.npy', value, allow_pickle=False)
        events, batches = [], []
        loader = trainer.loader(frame, backbone, 128, draws, SEED, number, workers=4)
        if loader.num_workers != 4: raise AssertionError('actual loader requires four workers')
        for x, y in loader:
            if len(y) != 128: raise AssertionError('effective batch128 required')
            batches.append(dict(inputs=tree_hash(x), labels=tree_hash(y)))
            event = trainer.advance_step(model, optimizer, scaler, x, y, number - 1, 1, 2, lr, micro, device)
            missing = [name for name, p in model.named_parameters() if p.grad is None]
            if missing: raise ValueError('parameters without gradients: ' + repr(missing))
            if event['skip'] or not event['gradient_finite']:
                raise FloatingPointError('qualification numeric failure: skip/nonfinite requires protocol review')
            events.append(event)
        if len(events) != 1: raise AssertionError('qualification epoch must attempt exactly one update')
        metrics, video = trainer.source_metrics(model, dev, backbone, micro, device, workers=4)
        video.to_parquet(out / f'{prefix}_source_dev.parquet', index=False)
        result = dict(epoch=number, attempts_after=number, events=events, batch_hashes=batches,
            samples_sha256=trainer.digest(out / f'{prefix}_samples.npy'),
            methods_sha256=trainer.digest(out / f'{prefix}_methods.npy'),
            augmentations_sha256=trainer.digest(out / f'{prefix}_augmentations.npy'), dev=metrics)
        trainer.save_json(out / f'{prefix}_record.json', result)
        return result

    model, optimizer, scaler = make()
    initial_model_state_sha256 = tree_hash(model.state_dict())
    first = epoch(model, optimizer, scaler, 1, 'epoch_01')
    trainer.save_state(out / 'epoch_01_state.pt', model, optimizer, scaler, 1, identity, [first], 1)
    continuous = epoch(model, optimizer, scaler, 2, 'epoch_02_continuous')
    trainer.save_state(out / 'continuous_state.pt', model, optimizer, scaler, 2, identity, [first, continuous], 2)
    target = torch.load(out / 'continuous_state.pt', map_location='cpu', weights_only=False)
    features = _export(model, frame, backbone, micro, device, out / 'features_continuous.npy')
    assert_byte_equal(target['torch_rng'], torch.get_rng_state(), 'exportCPU_RNG')
    if cuda: assert_byte_equal(target['cuda_rng'], torch.cuda.get_rng_state_all(), 'exportCUDA_RNG')
    del model, optimizer, scaler; gc.collect()
    if cuda: torch.cuda.empty_cache()
    model, optimizer, scaler = make()
    restored = trainer.load_state(out / 'epoch_01_state.pt', model, optimizer, scaler, identity)
    if restored['epoch'] != 1 or restored['attempts'] != 1:
        raise ValueError('restored attempt clock differs')
    resumed = epoch(model, optimizer, scaler, 2, 'epoch_02_resumed')
    assert_byte_equal(continuous, resumed, 'epoch2')
    trainer.save_state(out / 'resumed_state.pt', model, optimizer, scaler, 2, identity, [first, resumed], 2)
    again_state = torch.load(out / 'resumed_state.pt', map_location='cpu', weights_only=False)
    comparisons = {}
    for key in STATE_KEYS:
        assert_byte_equal(target[key], again_state[key], key)
        comparisons[key] = tree_hash(target[key])
    again = _export(model, frame, backbone, micro, device, out / 'features_resumed.npy')
    assert_byte_equal(features, again, 'Raw_FP32_features')
    rejected = []
    for field in ['config', 'data', 'code', 'initialization']:
        try:
            trainer.load_state(out / 'epoch_01_state.pt', model, optimizer, scaler, {**identity, field: 'intentional_changed_identity'})
        except ValueError as error:
            if 'identity mismatch' not in str(error): raise
            rejected.append(field)
        else:
            raise AssertionError('changed identity accepted: ' + field)
    weight_array = model.classifier.weight.detach().cpu().numpy()
    bias_array = model.classifier.bias.detach().cpu().numpy()
    np.savez(out / 'head_fp32.npz', weight=weight_array, bias=bias_array)
    scores = {}
    for name, subset in [('dev', dev), ('cal', cal)]:
        h = _export(model, subset, backbone, micro, device, out / f'{name}_features_fp32.npy')
        scores[name] = _video_scores(subset, h, weight_array, bias_array)
        scores[name].to_parquet(out / f'source_{name}.parquet', index=False)
    recomputed = recompute_source_metrics(scores['dev'], scores['cal'])
    for key in ['macro_auc', 'pooled_auc', 'method_auc']:
        assert_byte_equal(resumed['dev'][key], recomputed['dev'][key], 'savedDEV/' + key)
    assert_byte_equal(again_state['torch_rng'], torch.get_rng_state(), 'final_exportCPU_RNG')
    if cuda: assert_byte_equal(again_state['cuda_rng'], torch.cuda.get_rng_state_all(), 'final_exportCUDA_RNG')
    trainer.save_json(out / 'metrics.json', recomputed)
    result = dict(status='PASS', identity=identity, precision=precision, microbatch=micro,
        effective_batch=128, loader_workers=4, epochs=2, attempts_per_epoch=1,
        continuous_attempts=2, resumed_attempts=2, all_trainable_parameters_received_gradients=True,
        resume_parameter_optimizer_scaler_rng_byte_exact=True, resume_FP32_features_byte_exact=True,
        epoch2_sampler_methods_augmentations_batches_byte_exact=True, features_dimension=DIMENSIONS[backbone],
        changed_identity_fields_rejected=rejected, state_comparisons=comparisons,
        source_metrics=recomputed, initialization_model_sha256=initial_model_state_sha256, target_used=False)
    trainer.save_json(out / 'result.json', result)
    del model, optimizer, scaler, target, again_state; gc.collect()
    if cuda: torch.cuda.empty_cache()
    return result


def _attempt_identity(root, binding, probe, config, backbone, microbatch, precision):
    return dict(**binding, config=tree_hash(dict(backbone=backbone, microbatch=microbatch, precision=precision,
        seed=SEED, epochs=2, effective_batch=128, workers=4, draw_count=128,
        learning_rate=config['backbones'][backbone]['learning_rate'])),
        data=trainer.digest(Path(root) / 'qualification/H/probes/receipt.json'), code=trainer.code_hash(),
        initialization=config['backbones'][backbone]['weight_sha256'], backbone=backbone,
        microbatch=microbatch, precision=precision)


def worker(root, backbone, microbatch=32, precision='AMP', *, admission_policy=None):
    """Run one future owned GPU job; OOM alone permits caller retry32->16->8."""
    if backbone not in BACKBONES: raise ValueError('unknown backbone')
    if precision not in PRECISIONS: raise ValueError('precision must be AMP or FP32')
    if microbatch not in MICROBATCHES: raise ValueError('microbatch must be32,16,8')
    root = Path(root)
    with trainer.exclusive_file(root / 'qualification/H' / backbone / precision / 'qualification.lock'):
        freeze, config, binding = _context(root)
        probe = _probe_receipt(root, binding)
        identity = _attempt_identity(root, binding, probe, config, backbone, microbatch, precision)
        out = root / 'qualification/H' / backbone / precision / f'micro{microbatch}'
        if (out / 'receipt.json').exists():
            return _verify_attempt(root, out, identity, probe)
        if (out / 'failure.json').exists() or out.exists():
            raise ValueError('existing incomplete/failed qualification evidence requires review; refuse overwrite')
        out.mkdir(parents=True)
        try:
            # The direct CLI must obey the same physical GPU identity and
            # D-priority boundary as queue_h before the trainer takes ownership.
            from .worker_h import verify_resource_boundary
            if admission_policy is None:
                verify_resource_boundary(config, require_D_priority=True)
                ownership = trainer.owned_gpu(root, config)
            else:
                from .admission_h import owned_drain_qualification
                ownership = owned_drain_qualification(root, config, _read(admission_policy))
            with ownership as admission_receipt:
                if admission_policy is not None:
                    trainer.save_json(out / 'resource_admission.json', admission_receipt)
                trainer.policy(); torch.set_num_threads(2)
                if torch.cuda.device_count() != 1: raise ValueError('exactly one explicitly assigned CUDA GPU required')
                torch.cuda.reset_peak_memory_stats(); start = time.time()
                histories = {}
                cal = _manifest(root, probe['tables']['B/SOURCE_CAL'])
                for version in ['B', 'AB']:
                    frame = _manifest(root, probe['tables'][f'{version}/train'])
                    dev = _manifest(root, probe['tables'][f'{version}/dev'])
                    histories[version] = _run_history(frame, dev, cal, backbone, config['backbones'][backbone],
                        {**identity, 'history': version}, out / version, microbatch, precision)
                _verify_history_pair(out)
                if _context(root)[2] != binding: raise ValueError('qualification identity changed during worker')
                _probe_receipt(root, binding)
                torch.cuda.synchronize()
                receipt = dict(status='PASS', identity=identity, backbone=backbone, precision=precision,
                    microbatch=microbatch, effective_batch=128, loader_workers=4, source_only=True, target_used=False,
                    histories=histories, device_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    device_name=torch.cuda.get_device_name(), peak_memory_bytes=torch.cuda.max_memory_allocated(),
                    elapsed_seconds=time.time() - start,
                    evidence_sha256={str(p.relative_to(out)): trainer.digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
                trainer.save_json(out / 'receipt.json', receipt)
                return receipt
        except BaseException as error:
            oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'cuda out of memory' in str(error).lower()
            trainer.save_json(out / 'failure.json', dict(status='OOM_RETRY_ALLOWED' if oom else 'REVIEW_REQUIRED',
                identity=identity, error=repr(error), source_only=True, target_used=False))
            raise


def _verify_history_pair(out):
    histories = [_read(Path(out) / version / 'result.json') for version in ['B', 'AB']]
    if not histories[0].get('initialization_model_sha256') or histories[0]['initialization_model_sha256'] != histories[1].get('initialization_model_sha256'):
        raise ValueError('B/AB encoder and new head initialization differ')
    for prefix in ['epoch_01', 'epoch_02_continuous', 'epoch_02_resumed']:
        for name in ['samples', 'methods', 'augmentations']:
            a, b = [np.load(Path(out) / v / f'{prefix}_{name}.npy', allow_pickle=False) for v in ['B', 'AB']]
            assert_byte_equal(a, b, f'B_AB/{prefix}/{name}')


def _receipt_header(root, out, identity):
    receipt = _read(out / 'receipt.json')
    if receipt.get('status') != 'PASS' or receipt.get('identity') != identity or receipt.get('target_used') is not False or receipt.get('source_only') is not True:
        raise ValueError('qualification receipt missing, failed, or stale')
    if any(receipt.get(k) != identity[k] for k in ['backbone', 'precision', 'microbatch']):
        raise ValueError('qualification receipt task identity mismatch')
    if receipt.get('device_uuid') not in _read(Path(root) / 'configs/base.json')['allowed_gpu_uuids']:
        raise ValueError('qualification receipt GPU not approved')
    if receipt.get('effective_batch') != 128 or receipt.get('loader_workers') != 4 or set(receipt.get('histories', {})) != {'B', 'AB'}:
        raise ValueError('qualification receipt contract incomplete')
    return receipt


def _attempt_stats(out):
    """Fresh metadata fingerprints; intentionally not a fresh content hash."""
    result = {}
    for path in sorted(Path(out).rglob('*')):
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(out))] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def _verify_attempt(root, out, identity, probe):
    initial_stats = _attempt_stats(out)
    receipt = _receipt_header(root, out, identity)
    files = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name != 'receipt.json'}
    if files != set(receipt['evidence_sha256']): raise ValueError('qualification evidence file set changed')
    for name, expected in receipt['evidence_sha256'].items():
        if trainer.digest(out / name) != expected: raise ValueError('qualification evidence byte hash changed: ' + name)
    for version in ['B', 'AB']:
        history = receipt['histories'][version]
        if history != _read(out / version / 'result.json') or history.get('identity') != {**identity, 'history': version}:
            raise ValueError('qualification history identity mismatch')
        expected = dict(status='PASS', precision=identity['precision'], microbatch=identity['microbatch'],
            effective_batch=128, loader_workers=4, epochs=2, attempts_per_epoch=1,
            continuous_attempts=2, resumed_attempts=2, features_dimension=DIMENSIONS[identity['backbone']], target_used=False)
        if any(history.get(k) != v for k, v in expected.items()) or set(history.get('state_comparisons', {})) != set(STATE_KEYS):
            raise ValueError('qualification history clock/precision/dimension proof incomplete')
        required = ['all_trainable_parameters_received_gradients', 'resume_parameter_optimizer_scaler_rng_byte_exact',
            'resume_FP32_features_byte_exact', 'epoch2_sampler_methods_augmentations_batches_byte_exact']
        if any(history.get(key) is not True for key in required) or history.get('changed_identity_fields_rejected') != ['config', 'data', 'code', 'initialization']:
            raise ValueError('qualification resume/gradient/identity proof incomplete')
        states = [torch.load(out / version / name, map_location='cpu', weights_only=False) for name in ['continuous_state.pt', 'resumed_state.pt']]
        if any(state['identity'] != {**identity, 'history': version} or state['epoch'] != 2 or state['attempts'] != 2 for state in states):
            raise ValueError('saved qualification state identity/clock differs')
        for key in history['state_comparisons']:
            assert_byte_equal(states[0][key], states[1][key], 'accept/' + key)
            if tree_hash(states[0][key]) != history['state_comparisons'][key]:
                raise ValueError('saved state comparison evidence changed')
        del states
        features = [np.load(out / version / f'features_{name}.npy', allow_pickle=False) for name in ['continuous', 'resumed']]
        assert_byte_equal(*features, 'accept/features')
        if features[0].dtype != np.float32 or features[0].shape != (128, DIMENSIONS[identity['backbone']]) or not np.isfinite(features[0]).all():
            raise ValueError('qualification Raw FP32 export contract changed')
        train = _manifest(root, probe['tables'][f'{version}/train'])
        for prefix, epoch in [('epoch_01', 1), ('epoch_02_continuous', 2), ('epoch_02_resumed', 2)]:
            draws, methods = trainer.epoch_draws(train, SEED, epoch, 128)
            augmentations = np.asarray([[p['flip'], p['brightness'], p['contrast']] for p in (
                trainer.augmentation_parameters(SEED, epoch, str(train.iloc[int(index)].mother_id), i)
                for i, index in enumerate(draws))], dtype=np.float64)
            for name, expected_array in [('samples', draws), ('methods', methods), ('augmentations', augmentations)]:
                assert_byte_equal(expected_array, np.load(out / version / f'{prefix}_{name}.npy', allow_pickle=False), 'accept/draw/' + name)
        scores = {}
        head = np.load(out / version / 'head_fp32.npz', allow_pickle=False)
        if head['weight'].dtype != np.float32 or head['bias'].dtype != np.float32 or head['weight'].shape != (2, DIMENSIONS[identity['backbone']]) or head['bias'].shape != (2,):
            raise ValueError('saved source FP32 head contract changed')
        for name, key in [('dev', f'{version}/dev'), ('cal', 'B/SOURCE_CAL')]:
            frame = _manifest(root, probe['tables'][key])
            h = np.load(out / version / f'{name}_features_fp32.npy', allow_pickle=False)
            if h.dtype != np.float32 or h.shape != (len(frame), DIMENSIONS[identity['backbone']]) or not np.isfinite(h).all():
                raise ValueError('saved source FP32 features invalid')
            scores[name] = pd.read_parquet(out / version / f'source_{name}.parquet')
            pd.testing.assert_frame_equal(scores[name], _video_scores(frame, h, head['weight'], head['bias']))
        if recompute_source_metrics(scores['dev'], scores['cal']) != history['source_metrics']:
            raise ValueError('saved source DEV/CAL independent recomputation failed')
    _verify_history_pair(out)
    if initial_stats != _attempt_stats(out):
        raise ValueError('qualification evidence changed during full verification')
    return receipt


def verified_attempt(root, backbone, precision, microbatch):
    """Fresh full validation of one completed attempt; no GPU query or launch."""
    if backbone not in BACKBONES or precision not in PRECISIONS or microbatch not in MICROBATCHES:
        raise ValueError('invalid qualification attempt identity')
    root = Path(root)
    _, config, binding = _context(root)
    probe = _probe_receipt(root, binding)
    identity = _attempt_identity(root, binding, probe, config, backbone, microbatch, precision)
    out = root / 'qualification/H' / backbone / precision / f'micro{microbatch}'
    return _verify_attempt(root, out, identity, probe)


def next_attempt(root, backbone, precision):
    """Read-only resume plan; fully verify completed work without launching it."""
    if backbone not in BACKBONES or precision not in PRECISIONS:
        raise ValueError('invalid qualification attempt identity')
    root = Path(root)
    _, config, binding = _context(root)
    probe = _probe_receipt(root, binding)
    oom_failures = []
    for microbatch in MICROBATCHES:
        out = root / 'qualification/H' / backbone / precision / f'micro{microbatch}'
        identity = _attempt_identity(root, binding, probe, config, backbone, microbatch, precision)
        result = dict(backbone=backbone, precision=precision, microbatch=microbatch, oom_failures=oom_failures)
        if (out / 'receipt.json').exists():
            _verify_attempt(root, out, identity, probe)
            return dict(result, status='COMPLETE', receipt=_relative(root, out / 'receipt.json'),
                receipt_sha256=trainer.digest(out / 'receipt.json'))
        if (out / 'failure.json').exists():
            failure = _read(out / 'failure.json')
            if failure.get('identity') != identity or failure.get('status') != 'OOM_RETRY_ALLOWED':
                raise ValueError('qualification failure is stale or requires numeric/protocol review')
            oom_failures.append(dict(path=_relative(root, out / 'failure.json'), sha256=trainer.digest(out / 'failure.json')))
            continue
        if out.exists():
            raise ValueError('incomplete qualification attempt requires review')
        return dict(result, status='READY')
    raise ValueError('no allowed H qualification microbatch passed')


def _verify_small_evidence(out, receipt):
    """Rehash small proof files even when their accepted size/mtime is intact."""
    files = {str(p.relative_to(out)) for p in Path(out).rglob('*') if p.is_file() and p.name != 'receipt.json'}
    if files != set(receipt.get('evidence_sha256', {})):
        raise ValueError('qualification evidence file set changed')
    for name, expected in receipt['evidence_sha256'].items():
        path = Path(out) / name
        if path.stat().st_size <= SMALL_EVIDENCE_LIMIT and trainer.digest(path) != expected:
            raise ValueError('small qualification evidence byte hash changed: ' + name)


def _qualified_backbones(root, binding, probe, config, *, deep=True, accepted_stats=None, stats_out=None):
    records = {}
    seen = set()
    for backbone in BACKBONES:
        precisions = {}
        for precision in PRECISIONS:
            oom_failures = []
            for micro in MICROBATCHES:
                out = Path(root) / 'qualification/H' / backbone / precision / f'micro{micro}'
                identity = _attempt_identity(root, binding, probe, config, backbone, micro, precision)
                if (out / 'receipt.json').exists():
                    key = _relative(root, out / 'receipt.json')
                    before = _attempt_stats(out)
                    if accepted_stats is not None and accepted_stats.get(key) != before:
                        raise ValueError('accepted qualification evidence stat fingerprint changed: ' + key)
                    if deep: _verify_attempt(root, out, identity, probe)
                    else: _verify_small_evidence(out, _receipt_header(root, out, identity))
                    if before != _attempt_stats(out):
                        raise ValueError('qualification evidence changed during verification')
                    if stats_out is not None: stats_out[key] = before
                    seen.add(key)
                    precisions[precision] = dict(microbatch=micro, receipt=key, receipt_sha256=trainer.digest(out / 'receipt.json'), oom_failures=oom_failures)
                    break
                failure = _read(out / 'failure.json') if (out / 'failure.json').exists() else {}
                if failure.get('status') != 'OOM_RETRY_ALLOWED' or failure.get('identity') != identity:
                    raise ValueError(f'qualification receipt required for {backbone}/{precision}/micro{micro}; only verified OOM permits smaller batch')
                oom_failures.append(dict(path=_relative(root, out / 'failure.json'), sha256=trainer.digest(out / 'failure.json')))
            else:
                raise ValueError('no allowed H qualification microbatch passed')
        records[backbone] = dict(microbatch=precisions['AMP']['microbatch'], fp32_microbatch=precisions['FP32']['microbatch'], precisions=precisions)
    if accepted_stats is not None and set(accepted_stats) != seen:
        raise ValueError('accepted qualification evidence stat receipt set changed')
    return records


def accept_qualification(root):
    """Create H-only production gate after six complete two-history GPU proofs."""
    root = Path(root)
    with trainer.exclusive_file(root / 'qualification/H/accept.lock'):
        freeze, config, binding = _context(root, deep=False)
        probe = _probe_receipt(root, binding)
        stats = {}
        backbones = _qualified_backbones(root, binding, probe, config, deep=True, stats_out=stats)
        gate = dict(status='PASS', approved_routes=['H'], source_only=True, target_used=False, **binding,
            probe_receipt_sha256=trainer.digest(root / 'qualification/H/probes/receipt.json'), backbones=backbones,
            accepted_evidence_stats=stats,
            acceptance_verification='fresh_cryptographic_full_GPU_evidence_and_accepted_H_freeze_metadata_bindings')
        path = root / 'qualification/H_gate.json'
        if path.exists():
            if _read(path) != gate: raise ValueError('existing H gate differs; refuse overwrite')
            return gate
        trainer.save_json(path, gate)
        return gate


def validate_qualification(root, *, deep=False):
    """Validate bindings and accepted evidence freshness without loading models.

    Default: freshly hash receipts, proof files up to8MiB and selected PNGs, and require
    every accepted evidence file's size/mtime to match its full-acceptance stat.
    ``deep=True`` additionally freshly hashes/deserializes all GPU proof files.
    The default is metadata freshness of previously cryptographically accepted
    proof, not a new cryptographic verification of large checkpoint bytes.
    """
    root = Path(root)
    path = root / 'qualification/H_gate.json'
    if not path.is_file(): raise ValueError('H qualification gate missing')
    gate = _read(path)
    freeze, config, binding = _context(root)
    if gate.get('status') != 'PASS' or gate.get('approved_routes') != ['H'] or gate.get('target_used') is not False or any(gate.get(k) != v for k, v in binding.items()):
        raise ValueError('H qualification gate missing or stale identity')
    probe = _probe_receipt(root, binding)
    if gate.get('probe_receipt_sha256') != trainer.digest(root / 'qualification/H/probes/receipt.json'):
        raise ValueError('H gate source probe identity changed')
    if not isinstance(gate.get('accepted_evidence_stats'), dict):
        raise ValueError('H gate lacks accepted evidence stat fingerprints')
    if gate.get('backbones') != _qualified_backbones(root, binding, probe, config, deep=deep,
            accepted_stats=gate['accepted_evidence_stats']):
        raise ValueError('H gate qualification receipt identity changed')
    return dict(gate, _validation=dict(mode='fresh_cryptographic_full_evidence' if deep else 'accepted_evidence_stat_freshness',
        large_checkpoints_deserialized=deep, selected_source_PNG_byte_and_RGB_hashes_fresh=True,
        small_evidence_hash_limit_bytes=SMALL_EVIDENCE_LIMIT, full_source_videos_rehashed=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker', 'accept', 'validate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backbone', choices=BACKBONES)
    parser.add_argument('--micro', type=int, choices=MICROBATCHES, default=32)
    parser.add_argument('--precision', choices=PRECISIONS, default='AMP')
    parser.add_argument('--admission-policy', type=Path, help='Reviewed qualification-only post-lock D drain lease policy')
    parser.add_argument('--deep', action='store_true', help='Fresh full GPU proof verification for validate')
    args = parser.parse_args()
    if args.action == 'worker': result = worker(args.root, args.backbone, args.micro, args.precision, admission_policy=args.admission_policy)
    elif args.action == 'validate': result = validate_qualification(args.root, deep=args.deep)
    else: result = {'prepare': prepare_probe, 'accept': accept_qualification, 'validate': validate_qualification}[args.action](args.root)
    print(json.dumps(result, allow_nan=False), flush=True)
