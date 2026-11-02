"""H worker provenance checks around the unchanged, qualified trainer.

No target inference. All checks precede CUDA allocation. A valid H data freeze
and runtime gate are separate from the running D configuration and gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from . import trainer


def h_code_hash():
    root = Path(__file__).parent
    names = ('data_h.py', 'qualification_h.py', 'queue_h.py', 'worker_h.py', 'queue_d.py')
    return hashlib.sha256(b''.join(name.encode() + b'\0' + (root / name).read_bytes() for name in names)).hexdigest()


def expected_training_identity(config, task):
    root = Path(config['artifact_root'])
    return dict(config=hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
        train_manifest=trainer.digest(root / task['train_manifest']),
        dev_manifest=trainer.digest(root / task['dev_manifest']),
        initialization=trainer.digest(config['backbones'][task['backbone']]['weight_path']),
        environment=trainer.digest(root / 'configs/requirements.execution.lock.txt'),
        code=trainer.code_hash(), qualification=trainer.digest(root / task['qualification_gate']))


def verify_resource_boundary(config, *, require_D_priority=True):
    """Fresh physical index/UUID and D-priority checks before CUDA allocation."""
    from .queue_d import gpu_memory
    from .queue_h import d_priority_ready
    import os
    if config.get('allowed_gpus') != [0, 1, 2, 3]:
        raise ValueError('H resource boundary only permits GPU0-3')
    memory = gpu_memory()
    expected = [memory[i]['uuid'] for i in (0, 1, 2, 3)]
    if config.get('allowed_gpu_uuids') != expected:
        raise ValueError('H approved UUID list differs from physical GPU0-3')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in expected:
        raise ValueError('H requires exactly one approved GPU0-3 UUID')
    if require_D_priority and not d_priority_ready(config['artifact_root'])['ready']:
        raise ValueError('D GPU queue retains priority; no H GPU work is allowed')
    return dict(physical_gpu_uuids=expected, D_priority_checked=require_D_priority)


def validate_task(config_path, run_id, *, verify_inputs=False, require_dispatch=False):
    from .data_h import validate_h_freeze
    from .qualification_h import validate_qualification
    if not re.fullmatch(r'H_[A-Za-z0-9_.-]+', run_id):
        raise ValueError('unsafe or non-H run identity')
    path = Path(config_path).resolve(); config = json.loads(path.read_text())
    root = Path(config['artifact_root']).resolve()
    if path != root / 'configs/H_execution.json' or config.get('schema') != 'H_EXECUTION_V2':
        raise ValueError('H worker requires its dedicated H_execution configuration')
    freeze = validate_h_freeze(root); gate = validate_qualification(root)
    base = json.loads((root / 'configs/base.json').read_text())
    for key in ('artifact_root', 'backbones', 'allowed_gpus', 'allowed_gpu_uuids', 'workers', 'effective_batch'):
        if config.get(key) != base.get(key):
            raise ValueError('H global header differs from accepted base: ' + key)
    if config.get('development_seeds') != [1007, 1042] or config.get('formal_seeds') != [7, 42, 123, 2026, 3407]:
        raise ValueError('H global seed registry changed')
    expected_micro = {b: {'AMP': g['microbatch'], 'FP32': g['fp32_microbatch']} for b, g in gate['backbones'].items()}
    if config.get('H_microbatches') != expected_micro:
        raise ValueError('H global microbatch registry differs from qualification')
    if config.get('H_training_rows') != freeze['tables']['B/train']['rows'] or config.get('H_training_draw_count') != freeze['training_draw_count']:
        raise ValueError('H global paired exposure changed')
    if require_dispatch:
        verify_resource_boundary(config)
    if trainer.digest(root / 'data/H/freeze.json') != config['H_data_freeze_sha256']:
        raise ValueError('H data freeze changed')
    if trainer.digest(root / 'qualification/H_gate.json') != config['H_qualification_sha256']:
        raise ValueError('H source qualification changed')
    if config.get('h_code_bundle_sha256') != h_code_hash() or gate['h_code_bundle_sha256'] != h_code_hash():
        raise ValueError('H runtime code differs from the qualified bundle')
    task = config.get('runs', {}).get(run_id)
    if task is None or task.get('route') != 'H' or task.get('history') not in ('B', 'AB'):
        raise ValueError('unregistered H task/history')
    if task.get('h_code_bundle_sha256') != h_code_hash() or task.get('H_data_freeze_sha256') != config['H_data_freeze_sha256']:
        raise ValueError('task provenance differs from frozen H data/code')
    if config['effective_batch'] != 128 or task['draw_count'] != freeze['training_draw_count'] or task['draw_count'] % 128:
        raise ValueError('H paired exposure or effective-batch budget changed')
    if task['epochs'] not in (30, 45, 60) or task['backbone'] not in gate['backbones']:
        raise ValueError('unregistered H epoch budget/backbone')
    if task['precision'] not in ('AMP', 'FP32'):
        raise ValueError('unregistered H precision')
    micro_key = 'microbatch' if task['precision'] == 'AMP' else 'fp32_microbatch'
    if task['microbatch'] != gate['backbones'][task['backbone']][micro_key]:
        raise ValueError('unqualified H precision/microbatch')
    if task['qualification_gate'] != 'qualification/H_gate.json' or task['selection'] != 'frozen_last':
        raise ValueError('H qualification/terminal selection changed')
    if task['phase'] == 'formal':
        if task.get('recipe_frozen') is not True or task['seed'] not in config['formal_seeds']:
            raise ValueError('H formal recipe/seed not frozen')
        recipe_path = root / 'configs/H_frozen_recipes.json'
        if trainer.digest(recipe_path) != task['source_recipe_sha256']:
            raise ValueError('H source recipe identity changed')
        recipe = json.loads(recipe_path.read_text())
        if recipe.get('target_scores_used') is not False:
            raise ValueError('H recipe does not certify source-only selection')
        selected = recipe['recipes'][task['backbone']]
        if task['epochs'] != selected['epochs'] or task['lr'] != selected['lr']:
            raise ValueError('H histories must use the same frozen source recipe')
    elif task['phase'] != 'development' or task['seed'] not in config['development_seeds']:
        raise ValueError('unregistered H development phase/seed')
    else:
        lrs = (1e-5, 3e-5, 5e-5) if task['backbone'] == 'dinov2_vit_b14' else (1e-5,) if task['backbone'] == 'clip_vit_b16' else (1e-4,)
        if task['precision'] != 'AMP' or task['lr'] not in lrs or (task['epochs'] != 30 and task['backbone'] != 'dinov2_vit_b14'):
            raise ValueError('unregistered H development grid/precision/budget')
    from .queue_h import development_id, formal_id
    canonical = (formal_id(task['backbone'], task['history'], task['seed'], task['precision'])
        if task['phase'] == 'formal' else development_id(task['backbone'], task['history'], task['seed'], task['lr'], task['epochs']))
    if run_id != canonical:
        raise ValueError('H run identity differs from canonical task fields')
    frames = {}
    for role in ('train', 'dev'):
        key = task['history'] + '/' + role
        expected = Path(freeze['tables'][key]['path']).resolve()
        actual = (root / task[role + '_manifest']).resolve()
        if actual != expected or actual != root / f'data/H/{key}.parquet':
            raise ValueError('H task must use the accepted common source cohort')
        if trainer.digest(actual) != freeze['tables'][key]['sha256']:
            raise ValueError('H source manifest changed')
        if verify_inputs:
            frames[role] = pd.read_parquet(actual)
            for row in frames[role][['image_path', 'content_sha256']].drop_duplicates().itertuples(index=False):
                if trainer.digest(row.image_path) != row.content_sha256:
                    raise ValueError('H source input byte hash changed')
    identity = expected_training_identity(config, task)
    if identity['initialization'] != config['backbones'][task['backbone']]['weight_sha256']:
        raise ValueError('pretrained initialization byte identity changed')
    return config, task, identity


def _verify_original_completion(config_path, run_id, *, context=None):
    config, task, identity = context if context is not None else validate_task(config_path, run_id)
    out = Path(config['artifact_root']) / 'runs' / run_id
    receipt = json.loads((out / 'complete.json').read_text())
    expected_attempts = task['epochs'] * (task['draw_count'] // 128)
    if (receipt.get('status') != 'TRAINING_COMPLETE' or receipt.get('identity') != identity
            or receipt.get('epochs') != task['epochs'] or receipt.get('attempts') != expected_attempts
            or receipt.get('selection') != 'frozen_last' or receipt.get('target_used') is not False
            or receipt.get('success', -1) + receipt.get('skip', -1) != expected_attempts):
        raise ValueError('H completion receipt identity/budget mismatch')
    if trainer.digest(out / 'final.pt') != receipt['final_sha256']:
        raise ValueError('H formal/development final checkpoint hash mismatch')
    if trainer.digest(out / 'source_best.pt') != receipt['source_best_sha256']:
        raise ValueError('H source-best checkpoint hash mismatch')
    history = json.loads((out / 'history.json').read_text())
    if [r['epoch'] for r in history] != list(range(1, task['epochs'] + 1)):
        raise ValueError('incomplete H history')
    if sum(r['success'] for r in history) != receipt['success'] or sum(r['skip'] for r in history) != receipt['skip']:
        raise ValueError('H history success/skip mismatch')
    return receipt


def _start_record(config_path, run_id, config, identity):
    return dict(schema='H_TRAINING_START_V1', run_id=run_id,
        config_path=str(Path(config_path).resolve()), artifact_root=str(Path(config['artifact_root']).resolve()),
        identity=identity, H_data_freeze_sha256=config['H_data_freeze_sha256'],
        h_code_bundle_sha256=config['h_code_bundle_sha256'], source_inputs_byte_verified=True,
        target_used=False)


def _completion_artifacts(epochs):
    names = ['complete.json', 'history.json', 'identity.json', 'resolved_config.json',
        'state.pt', 'source_best.pt', 'final.pt', 'H_training_start.json']
    for epoch in range(1, epochs + 1):
        names += [f'epoch_{epoch:02d}_{suffix}' for suffix in
                  ('updates.jsonl', 'samples.npy', 'methods.npy', 'augmentations.npy', 'source_dev.parquet')]
    return names


def finalize_completion_seal(config_path, run_id):
    """Seal committed evidence, also recovering a crash after terminal save.

    Recovery requires the matching pre-training H start record and authoritative
    checkpoint history. Existing seals are verified and never silently replaced.
    """
    config, task, identity = validate_task(config_path, run_id)
    out = Path(config['artifact_root']) / 'runs' / run_id
    seal_path = out / 'H_completion_seal.json'
    with trainer.exclusive_file(out / 'run.lock'):
        if seal_path.exists():
            return verify_run_completion(config_path, run_id)
        receipt = _verify_original_completion(config_path, run_id, context=(config, task, identity))
        expected_start = _start_record(config_path, run_id, config, identity)
        if not (out / 'H_training_start.json').is_file() or json.loads((out / 'H_training_start.json').read_text()) != expected_start:
            raise ValueError('unsealed H completion lacks matching pre-training start evidence')
        checkpoint = trainer.torch.load(out / 'state.pt', map_location='cpu', weights_only=False)
        history = json.loads((out / 'history.json').read_text())
        if (checkpoint.get('identity') != identity or checkpoint.get('epoch') != task['epochs']
                or checkpoint.get('attempts') != receipt['attempts'] or checkpoint.get('history') != history):
            raise ValueError('authoritative checkpoint history/identity differs from H completion')
        del checkpoint
        names = _completion_artifacts(task['epochs'])
        if any(not (out / name).is_file() for name in names):
            raise ValueError('H completion evidence inventory is incomplete')
        seal = dict(schema='H_COMPLETION_SEAL_V1', status='SEALED', run_id=run_id,
            config_path=str(Path(config_path).resolve()), artifact_root=str(Path(config['artifact_root']).resolve()),
            identity=identity, H_data_freeze_sha256=config['H_data_freeze_sha256'],
            h_code_bundle_sha256=config['h_code_bundle_sha256'],
            artifact_sha256={name: trainer.digest(out / name) for name in names}, target_used=False)
        trainer.save_json(seal_path, seal)
    return receipt


def verify_run_completion(config_path, run_id):
    config, task, identity = validate_task(config_path, run_id)
    receipt = _verify_original_completion(config_path, run_id, context=(config, task, identity))
    out = Path(config['artifact_root']) / 'runs' / run_id
    path = out / 'H_completion_seal.json'
    if not path.is_file():
        raise ValueError('H completion evidence has not been sealed')
    seal = json.loads(path.read_text())
    expected = dict(schema='H_COMPLETION_SEAL_V1', status='SEALED', run_id=run_id,
        config_path=str(Path(config_path).resolve()), artifact_root=str(Path(config['artifact_root']).resolve()),
        identity=identity, H_data_freeze_sha256=config['H_data_freeze_sha256'],
        h_code_bundle_sha256=config['h_code_bundle_sha256'], target_used=False)
    if any(seal.get(k) != value for k, value in expected.items()):
        raise ValueError('H completion seal run/root/identity changed')
    if set(seal.get('artifact_sha256', {})) != set(_completion_artifacts(task['epochs'])):
        raise ValueError('H completion seal expected inventory changed')
    for name, expected_digest in seal['artifact_sha256'].items():
        if trainer.digest(out / name) != expected_digest:
            raise ValueError('H sealed completion artifact hash changed: ' + name)
    return receipt


def read_sealed_history(root, run_id):
    out = Path(root) / 'runs' / run_id
    seal = json.loads((out / 'H_completion_seal.json').read_text())
    if seal.get('run_id') != run_id or seal.get('artifact_root') != str(Path(root).resolve()):
        raise ValueError('H sealed history run/root mismatch')
    if trainer.digest(out / 'history.json') != seal['artifact_sha256'].get('history.json'):
        raise ValueError('H sealed development history hash changed')
    return json.loads((out / 'history.json').read_text())


def run(config_path, run_id):
    # The wrapper lock also protects start/seal side effects across direct CLI
    # duplicates; the unchanged trainer retains its own run and GPU locks.
    config, _, _ = validate_task(config_path, run_id, require_dispatch=True)
    out = Path(config['artifact_root']) / 'runs' / run_id
    with trainer.exclusive_file(out / 'H_worker.lock'):
        config, task, identity = validate_task(config_path, run_id, verify_inputs=True, require_dispatch=True)
        if (out / 'H_completion_seal.json').exists():
            return verify_run_completion(config_path, run_id)
        start = _start_record(config_path, run_id, config, identity)
        start_path = out / 'H_training_start.json'
        if start_path.exists() and json.loads(start_path.read_text()) != start:
            raise ValueError('H pre-training start identity changed')
        if not start_path.exists():
            if any(path.name not in ('H_worker.lock', 'run.lock') for path in out.iterdir()):
                raise ValueError('existing H training artifacts lack pre-training start evidence; preserve and review')
            trainer.save_json(start_path, start)
        verify_resource_boundary(config)
        result = trainer.train(config_path, run_id)
        finalize_completion_seal(config_path, run_id)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--run-id', required=True)
    args = p.parse_args(); run(args.config, args.run_id)


if __name__ == '__main__':
    main()
