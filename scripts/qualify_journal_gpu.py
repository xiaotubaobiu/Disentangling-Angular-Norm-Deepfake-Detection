#!/usr/bin/env python3
"""Identity-bound, disposable full-batch AMP qualification on physical GPU 0.

Uses the production model, optimizer step, clipping, source image loader and EMA.
Each backbone runs in a fresh subprocess so all CUDA allocations are released.
No epochs, checkpoints, target images or recipe changes are performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def inventory():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in text.strip().splitlines():
        index, uuid, name, total, used = [v.strip() for v in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, name=name,
                         total_memory_mib=int(total), used_memory_mib=int(used)))
    return rows


def validate_device_binding(visible, config, devices):
    if (config['allowed_gpus'] != [0, 1, 2, 3]
            or visible != config['allowed_gpu_uuids'][0]):
        raise ValueError('Qualification requires only physical GPU 0 UUID')
    physical_zero = [row for row in devices if row['index'] == 0]
    if len(physical_zero) != 1 or physical_zero[0]['uuid'] != visible:
        raise ValueError('GPU 0 UUID differs from the configured physical device')


def validate_result(result):
    if result['batch_size'] != 128 or result['microbatch'] != 128 or result['precision'] != 'AMP':
        raise ValueError('Full batch 128 AMP is required without accumulation')
    if result['target_used'] or not result['source_batch_checked']:
        raise ValueError('A source-only real image batch is required')
    if not all(result[key] for key in ['initial_raw_equals_ema', 'raw_changed', 'ema_changed']):
        raise ValueError('Raw and EMA parameter evidence is incomplete')
    if result['parameter_count'] != result['trainable_parameter_count']:
        raise ValueError('All encoder and head parameters must be trainable')
    events = result['events']
    if sum(bool(row['success']) for row in events) < 2:
        raise ValueError('At least two successful optimizer updates are required')
    for row in events:
        if not math.isfinite(row['loss']):
            raise ValueError('Nonfinite loss')
        if bool(row['success']) == bool(row['skip']):
            raise ValueError('Optimizer success and AMP skip evidence disagree')
        if row['success'] and not row['gradient_finite']:
            raise ValueError('Nonfinite gradient on successful optimizer update')
    if not 0 < result['peak_memory_allocated_bytes'] <= result['total_device_memory_bytes']:
        raise ValueError('GPU memory evidence is invalid')


def tensor_digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def source_batch(backbone, seed):
    """Select only released FF train records whose known sources are TRAIN."""
    import numpy as np
    import torch
    from PIL import Image
    from journal_detector import data_d, trainer
    definition = data_d.load_definition()
    roles, _ = data_d.pair_partition(definition['splits'])
    path = data_d.DATA_ROOT / 'dataset_json_update/fsgan_ff.json'
    expected = definition['indices_sha256'][path.name]
    if digest(path) != expected:
        raise ValueError('Official source index hash mismatch')
    index = json.loads(path.read_text())
    selected = []
    for label_key, role_map in next(iter(index.values())).items():
        label = int(label_key.endswith('_Fake'))
        if not label_key.endswith(('_Real', '_Fake')):
            raise ValueError('Unexpected source index label')
        class_rows = []
        for original_id, record in role_map.get('train', {}).items():
            if record['label'] != label_key:
                raise ValueError('Source record label mismatch')
            for original in record['frames'][:1]:
                image = data_d.resolve_path(original, 'fsgan', 'train', label)
                sources = data_d.source_ids('ff', image.parent.name)
                if not sources or any(roles.get(value) != 'train' for value in sources):
                    continue
                with Image.open(image) as value:
                    pixels = np.asarray(value.convert('RGB'))
                rgb_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
                class_rows.append(dict(image_path=str(image), label=label,
                    source_ids=sources, source_role='train', domain='ff',
                    official_index_id=original_id, content_sha256=digest(image),
                    rgb_sha256=rgb_hash))
            if len(class_rows) == 64:
                break
        if len(class_rows) != 64:
            raise ValueError('Need 64 actual source train images per class')
        selected.extend(class_rows)
    if len(selected) != 128:
        raise ValueError('Expected a balanced real source batch of 128')
    tensors = []
    for i, row in enumerate(selected):
        params = trainer.augmentation_parameters(seed, 1, row['rgb_sha256'], i)
        row['augmentation_seed'] = params['seed']
        tensors.append(trainer.image_tensor(row['image_path'], backbone, params))
    return torch.stack(tensors), torch.tensor([row['label'] for row in selected]), dict(
        official_index=str(path), official_index_sha256=expected,
        data_definition_sha256=digest(data_d.DEFINITION_ROOT / 'definition.json'),
        selection='first frame of 64 released FF TRAIN videos per class with all known FF sources assigned TRAIN',
        graph_audit_scope='capacity check; full RGB graph audit is a separate prerequisite for formal training',
        samples=selected)


def worker(args):
    import torch
    import torchvision
    from journal_detector import trainer
    from journal_detector.models import build_model
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    backbone = args.worker
    spec = config['backbones'][backbone]
    result_path = Path(args.artifact_root) / 'operations' / ('gpu_qualification_' + backbone + '.json')
    result = dict(backbone=backbone, status='RUNNING', seed=7, batch_size=128,
        microbatch=128, precision='AMP', target_used=False, source_batch_checked=False,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), events=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
        identity=dict(config_sha256=digest(args.config), code_sha256=trainer.code_hash(),
            qualifier_sha256=digest(__file__), initialization_sha256=digest(BASE / spec['weight_path'])),
        environment=dict(python=sys.version, torch=torch.__version__,
            torchvision=torchvision.__version__, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version()),
        recipe=dict(optimizer='AdamW', learning_rate=spec['learning_rate'],
            weight_decay=5e-4, gradient_clip=1., ema_decay=.999, amp_init_scale=128,
            input_shape=[128, 3, 224, 224], output_l2=False))
    try:
        validate_device_binding(result['gpu_uuid'], config, inventory())
        if result['identity']['initialization_sha256'] != spec['weight_sha256']:
            raise ValueError('Official initialization hash mismatch')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        trainer.policy()
        trainer.set_random(7)
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one UUID-bound CUDA device must be visible')
        properties = torch.cuda.get_device_properties(0)
        actual_uuid = str(properties.uuid)
        if actual_uuid.removeprefix('GPU-') != result['gpu_uuid'].removeprefix('GPU-'):
            raise ValueError('PyTorch CUDA UUID differs from requested physical GPU 0')
        result.update(device_name=properties.name, torch_device_uuid=actual_uuid,
                      total_device_memory_bytes=properties.total_memory)
        torch.cuda.reset_peak_memory_stats()
        model = build_model(backbone, BASE / spec['weight_path']).to('cuda')
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=5e-4)
        scaler = torch.amp.GradScaler('cuda', init_scale=128)
        ema = trainer.EMA(model)
        result['parameter_count'] = sum(p.numel() for p in model.parameters())
        result['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        before_raw = tensor_digest(model.named_parameters())
        before_ema = tensor_digest(ema.shadow.items())
        result.update(initial_raw_parameter_sha256=before_raw,
                      initial_ema_parameter_sha256=before_ema,
                      initial_raw_equals_ema=before_raw == before_ema)
        generator = torch.Generator().manual_seed(7)
        rgb = torch.randint(0, 256, (128, 3, 224, 224), generator=generator, dtype=torch.uint8)
        mean, std = trainer.normalization(backbone)
        x = (rgb.float().div_(255) - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None]
        y = torch.arange(128) % 2
        result['synthetic_input_sha256'] = tensor_digest([('normalized_RGB', x), ('labels', y)])

        def update(x, y, kind):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            event = trainer.advance_step(model, optimizer, scaler, x, y,
                len(result['events']), 1000000, 15, spec['learning_rate'], 128, 'cuda')
            if event['success']:
                ema.update(model)
            torch.cuda.synchronize()
            event.update(input_kind=kind, batch_size=len(y), seconds=time.perf_counter() - tick,
                         peak_memory_allocated_bytes=torch.cuda.max_memory_allocated())
            result['events'].append(event)
            print(json.dumps(dict(backbone=backbone, **event), allow_nan=False), flush=True)
            save_json(result_path, result)

        while sum(row['success'] for row in result['events']) < 2:
            if len(result['events']) >= 8:
                raise FloatingPointError('Eight attempts without two successful full-batch updates')
            update(x, y, 'synthetic_normalized_RGB')
        x, y, source = source_batch(backbone, 7)
        result['source_batch'] = source
        source_start = len(result['events'])
        while not any(row['success'] for row in result['events'][source_start:]):
            if len(result['events']) - source_start >= 8:
                raise FloatingPointError('Eight actual source batch AMP skips')
            update(x, y, 'actual_source_TRAIN_official_augmentation')
        result['source_batch_checked'] = True
        model.eval()
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.float16):
                raw_logits = model(x[:8].to('cuda'))
            with ema.applied(model):
                with torch.autocast('cuda', dtype=torch.float16):
                    ema_logits = model(x[:8].to('cuda'))
        result['raw_logits_finite'] = bool(torch.isfinite(raw_logits).all())
        result['ema_logits_finite'] = bool(torch.isfinite(ema_logits).all())
        if not result['raw_logits_finite'] or not result['ema_logits_finite']:
            raise FloatingPointError('Nonfinite actual source raw/EMA logits')
        result['final_raw_parameter_sha256'] = tensor_digest(model.named_parameters())
        result['final_ema_parameter_sha256'] = tensor_digest(ema.shadow.items())
        result.update(raw_changed=result['final_raw_parameter_sha256'] != before_raw,
                      ema_changed=result['final_ema_parameter_sha256'] != before_ema,
                      peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                      success_updates=sum(row['success'] for row in result['events']),
                      amp_skips=sum(row['skip'] for row in result['events']))
        validate_result(result)
        result['status'] = 'QUALIFIED'
    except BaseException as error:
        result.update(status='FAILED', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            result.update(peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        traceback.print_exc()
    result.update(elapsed_seconds=time.perf_counter() - started,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  final_code_sha256=trainer.code_hash())
    result['gpu_hours'] = result['elapsed_seconds'] / 3600
    if result['final_code_sha256'] != result['identity']['code_sha256']:
        result.update(status='FAILED', error='Scientific source code changed during GPU qualification')
    save_json(result_path, result)
    print(json.dumps(dict(backbone=backbone, status=result['status'],
        elapsed_seconds=result['elapsed_seconds'], result_path=str(result_path))), flush=True)
    return 0 if result['status'] == 'QUALIFIED' else 1


def main(args):
    from journal_detector import trainer
    root = Path(args.artifact_root)
    config = json.loads(Path(args.config).read_text())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    validate_device_binding(visible, config, inventory())
    output = root / 'operations/gpu_qualification.json'
    receipt = dict(status='RUNNING', started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_uuid=visible, target_used=False, backbone_order=list(config['backbones']), results={},
        identity=dict(config_sha256=digest(args.config), code_sha256=trainer.code_hash(),
            qualifier_sha256=digest(__file__)), command=sys.argv)
    save_json(output, receipt)
    start = time.perf_counter()
    with trainer.owned_gpu(root, config):
        for backbone in config['backbones']:
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                '--config', str(Path(args.config).resolve()), '--artifact-root', str(root),
                '--worker', backbone]
            log_path = root / 'operations' / ('gpu_qualification_' + backbone + '.stdout.log')
            with log_path.open('w') as log:
                process = subprocess.Popen(command, cwd=BASE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                code = process.wait()
            result_path = root / 'operations' / ('gpu_qualification_' + backbone + '.json')
            result = json.loads(result_path.read_text()) if result_path.exists() else dict(status='FAILED')
            result.update(returncode=code, stdout_path=str(log_path), stdout_sha256=digest(log_path))
            receipt['results'][backbone] = result
            save_json(output, receipt)
        receipt.update(status='QUALIFIED' if all(row['status'] == 'QUALIFIED' for row in receipt['results'].values()) else 'FAILED',
            elapsed_seconds=time.perf_counter() - start,
            gpu_hours=sum(row.get('gpu_hours', 0) for row in receipt['results'].values()),
            finished_utc=datetime.now(timezone.utc).isoformat(), post_run_inventory=inventory())
        save_json(output, receipt)
    print(json.dumps(dict(status=receipt['status'], receipt=str(output))), flush=True)
    return 0 if receipt['status'] == 'QUALIFIED' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BASE / 'configs/journal.json'))
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--worker', choices=['clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0'])
    arguments = parser.parse_args()
    raise SystemExit(worker(arguments) if arguments.worker else main(arguments))
