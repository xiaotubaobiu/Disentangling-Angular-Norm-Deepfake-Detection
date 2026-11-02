"""Finite H source qualification, shared development and formal training queue.

Uses dedicated H config/status/registry/locks. Existing D has priority until its
GPU queue completes. H never reserves an idle slot ahead of a pending D task,
avoiding the running D supervisor's dispatch race. Target export is a separate
stage, gated by all five paired numerical ledgers per backbone.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np

from .trainer import digest, save_json, exclusive_file, code_hash
from .queue_d import choose_recipe, dino_extend, gpu_memory, reserved

BACKBONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')
HISTORIES = ('B', 'AB')
DEVELOPMENT_SEEDS = (1007, 1042)
FORMAL_SEEDS = (7, 42, 123, 2026, 3407)
METHODS = ('real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures')


def utc():
    return datetime.now(timezone.utc).isoformat()


def formal_id(backbone, history, seed, precision='AMP'):
    return f'H_formal_{backbone}_{history}_seed{seed}' + ('_FP32' if precision == 'FP32' else '')


def development_id(backbone, history, seed, lr, epochs):
    return f'H_dev_{backbone}_{history}_lr{lr:.0e}_seed{seed}_e{epochs}'


def registered_task(config, backbone, history, seed, lr, epochs, phase, precision='AMP', recipe_sha256=None):
    if backbone not in BACKBONES or history not in HISTORIES or precision not in ('AMP', 'FP32'):
        raise ValueError('unregistered H task')
    if epochs not in (30, 45, 60):
        raise ValueError('unregistered H budget')
    return dict(route='H', history=history, phase=phase, backbone=backbone, seed=seed,
        lr=lr, epochs=epochs, microbatch=config['H_microbatches'][backbone][precision],
        precision=precision, draw_count=config['H_training_draw_count'],
        train_manifest=f'data/H/{history}/train.parquet', dev_manifest=f'data/H/{history}/dev.parquet',
        qualification_gate='qualification/H_gate.json', recipe_frozen=phase == 'formal',
        selection='frozen_last', H_data_freeze_sha256=config['H_data_freeze_sha256'],
        h_code_bundle_sha256=config['h_code_bundle_sha256'], source_recipe_sha256=recipe_sha256)


def standard_development_tasks(config):
    result = {}
    for backbone in BACKBONES:
        grid = (1e-5, 3e-5, 5e-5) if backbone == 'dinov2_vit_b14' else (1e-5,) if backbone == 'clip_vit_b16' else (1e-4,)
        for lr in grid:
            for seed in DEVELOPMENT_SEEDS:
                for history in HISTORIES:
                    key = development_id(backbone, history, seed, lr, 30)
                    result[key] = registered_task(config, backbone, history, seed, lr, 30, 'development')
    return result


def formal_tasks(config, recipes, *, recipe_sha256, precision='AMP', backbone=None):
    result = {}
    for b in BACKBONES if backbone is None else (backbone,):
        recipe = recipes[b]
        for seed in FORMAL_SEEDS:
            for history in HISTORIES:
                key = formal_id(b, history, seed, precision)
                result[key] = registered_task(config, b, history, seed, recipe['lr'], recipe['epochs'],
                    'formal', precision, recipe_sha256)
    return result


def choose_shared_recipe(candidates):
    if any(len(c['histories']) != 4 for c in candidates):
        raise ValueError('shared H recipe requires exactly four histories: two seeds x two histories')
    for candidate in candidates:
        for history in candidate['histories']:
            if any(not np.isfinite(row['dev']['macro_auc']) for row in history):
                raise ValueError('invalid source development metric')
    return choose_recipe(candidates)


def extend_budget(histories, budget):
    if len(histories) != 4:
        raise ValueError('H DINO extension requires four source histories')
    return dino_extend(histories, budget)


def d_priority_ready(root):
    path = Path(root) / 'queue_status.json'
    if not path.is_file():
        return dict(ready=False, reason='D_QUEUE_STATUS_MISSING')
    q = json.loads(path.read_text())
    ready = q.get('stage') == 'P3_D_CACHE_COMPLETE_ANALYSIS_PENDING' and not q.get('active')
    return dict(ready=ready, reason='D_GPU_STAGE_COMPLETE' if ready else 'D_QUEUE_HAS_PRIORITY',
                D_stage=q.get('stage'), D_owner_pid=q.get('owner_pid'))


def capacity_check(free_bytes, *, initial=False, additional_bytes=0):
    required = (250 if initial else 50) * 1024**3 + int(additional_bytes)
    return dict(allowed=free_bytes > required, free_bytes=free_bytes, required_bytes=required,
        reserve_GiB=50, initial_working_set_estimate_GiB=200 if initial else 0,
        additional_conditional_bytes=int(additional_bytes))


def dispatchable_devices(config, memory, compute_uuids, locked_uuids, *, d_ready):
    if config['allowed_gpus'] != [0, 1, 2, 3] or len(config['allowed_gpu_uuids']) != 4:
        raise ValueError('H may only use the four approved GPU0-3 devices')
    for gpu, uuid in zip(config['allowed_gpus'], config['allowed_gpu_uuids']):
        if memory[gpu]['uuid'] != uuid:
            raise ValueError('approved GPU UUID changed')
    if not d_ready:
        return []
    return [i for i in config['allowed_gpus'] if memory[i]['used_mib'] < 100
        and memory[i]['uuid'] not in compute_uuids and memory[i]['uuid'] not in locked_uuids]


def owned_worker_alive(item):
    try:
        args = Path(f'/proc/{item["pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    identity = item.get('run_id') or item.get('qualification_key', '')
    if not identity:
        return False
    if 'reliability_v2.worker_h' in args:
        return identity in args
    return 'reliability_v2.qualification_h' in args and item.get('backbone', '') in args


def prepare_execution(root):
    from .data_h import validate_h_freeze
    from .qualification_h import validate_qualification
    from .worker_h import h_code_hash
    root = Path(root); freeze = validate_h_freeze(root); gate = validate_qualification(root)
    base = json.loads((root / 'configs/base.json').read_text())
    if base['allowed_gpus'] != [0, 1, 2, 3] or set(base['backbones']) != set(BACKBONES):
        raise ValueError('base device/backbone contract changed')
    config = {key: copy.deepcopy(base[key]) for key in
        ('artifact_root', 'backbones', 'allowed_gpus', 'allowed_gpu_uuids', 'workers', 'effective_batch')}
    config.update(schema='H_EXECUTION_V2', protocol='pipeline-reliability-v2-20261023',
        development_seeds=list(DEVELOPMENT_SEEDS), formal_seeds=list(FORMAL_SEEDS),
        H_data_freeze_sha256=digest(root / 'data/H/freeze.json'),
        H_qualification_sha256=digest(root / 'qualification/H_gate.json'),
        h_code_bundle_sha256=h_code_hash(), H_training_draw_count=freeze['training_draw_count'],
        H_training_rows=freeze['tables']['B/train']['rows'],
        H_microbatches={b: {'AMP': gate['backbones'][b]['microbatch'],
                          'FP32': gate['backbones'][b]['fp32_microbatch']} for b in BACKBONES},
        resource_policy='D_PRIORITY_UNTIL_D_GPU_STAGE_COMPLETE; max4 idle GPU0-3 only', runs={})
    path = root / 'configs/H_execution.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != 'runs'} != {k: v for k, v in config.items() if k != 'runs'}:
            raise ValueError('existing H execution header identity changed')
        return previous
    save_json(path, config)
    return config


def audit_paired_ledgers(config_path, backbone, *, precision='AMP', verify=None):
    if verify is None:
        from .worker_h import verify_run_completion
        verify = verify_run_completion
    config = json.loads(Path(config_path).read_text()); root = Path(config['artifact_root'])
    pairs = []; asymmetric = 0
    for seed in FORMAL_SEEDS:
        ids = {history: formal_id(backbone, history, seed, precision) for history in HISTORIES}
        tasks = {history: config['runs'][run] for history, run in ids.items()}
        if any(t['backbone'] != backbone or t['phase'] != 'formal' or t['precision'] != precision or t['seed'] != seed
               or t['history'] != history for history, t in tasks.items()):
            raise ValueError('unregistered formal pair identity')
        for key in ('epochs', 'lr', 'draw_count', 'microbatch', 'h_code_bundle_sha256', 'H_data_freeze_sha256'):
            if tasks['B'][key] != tasks['AB'][key]:
                raise ValueError('H pair has asymmetric frozen ' + key)
        receipts = {history: verify(config_path, run) for history, run in ids.items()}
        histories = {history: json.loads((root / 'runs' / run / 'history.json').read_text()) for history, run in ids.items()}
        epochs = tasks['B']['epochs']; draws = tasks['B']['draw_count']; steps = draws // 128
        events = {history: [] for history in HISTORIES}; artifacts = {}
        for history, run in ids.items():
            out = root / 'runs' / run
            if len(histories[history]) != epochs:
                raise ValueError('incomplete formal pair history')
            for epoch, row in enumerate(histories[history], 1):
                if row['epoch'] != epoch or row['attempts'] != steps or row['draw_count'] != draws:
                    raise ValueError('paired epoch/draw/attempt budget drift')
                path = out / f'epoch_{epoch:02d}_updates.jsonl'
                logs = [json.loads(line) for line in path.open()]
                if len(logs) != steps or [e['attempt'] for e in logs] != list(range((epoch-1)*steps, epoch*steps)):
                    raise ValueError('incomplete paired attempt ledger')
                if sum(e['success'] for e in logs) != row['success'] or sum(e['skip'] for e in logs) != row['skip']:
                    raise ValueError('paired history update counters differ from logs')
                for e in logs:
                    if e['success'] == e['skip'] or not math.isfinite(e['loss']):
                        raise ValueError('invalid paired numerical event')
                    if precision == 'FP32' and (e['skip'] or not e['gradient_finite']):
                        raise ValueError('FP32 numerical failure requires source protocol review')
                    if not e['gradient_finite'] and not e['skip']:
                        raise ValueError('nonfinite successful gradient')
                events[history].extend(bool(e['skip']) for e in logs)
                artifacts[str(path.relative_to(root))] = digest(path)
                for suffix, field in [('samples', 'sample_sha256'), ('augmentations', 'augmentation_sha256'), ('methods', None)]:
                    path = out / f'epoch_{epoch:02d}_{suffix}.npy'; value = digest(path)
                    if field and value != row[field]:
                        raise ValueError('paired ' + suffix + ' hash differs from committed history')
                    artifacts[str(path.relative_to(root))] = value
                    other = root / 'runs' / ids['B'] / path.name
                    if history == 'AB' and digest(other) != value:
                        raise ValueError('paired ' + suffix + ' arrays differ')
                    x = np.load(path, allow_pickle=False)
                    if len(x) != draws:
                        raise ValueError('paired array length differs from exposure budget')
                    if suffix == 'samples' and (x.ndim != 1 or not np.issubdtype(x.dtype, np.integer)
                        or x.min() < 0 or x.max() >= config['H_training_rows']):
                        raise ValueError('invalid paired sample indices')
                    if suffix == 'augmentations' and (x.shape != (draws, 3) or not np.isfinite(x).all()
                        or not np.isin(x[:, 0], [0, 1]).all() or (x[:, 1:] < .9).any() or (x[:, 1:] > 1.1).any()):
                        raise ValueError('invalid paired augmentation stream')
                    if suffix == 'methods':
                        counts = {str(k): int(v) for k, v in zip(*np.unique(x, return_counts=True))}
                        if not set(counts) <= set(METHODS) or counts != row['method_counts']:
                            raise ValueError('paired method counts differ from stream')
            if sum(events[history]) != receipts[history]['skip'] or len(events[history]) != receipts[history]['attempts']:
                raise ValueError('paired receipt budget differs from complete ledger')
        same = events['B'] == events['AB']; asymmetric += int(not same)
        pairs.append(dict(seed=seed, run_ids=ids, same_successful_update_stream=same,
            B_success=receipts['B']['success'], AB_success=receipts['AB']['success'],
            B_skips=receipts['B']['skip'], AB_skips=receipts['AB']['skip'],
            skip_positions={key: [i for i, skipped in enumerate(values) if skipped] for key, values in events.items()},
            final_sha256={key: value['final_sha256'] for key, value in receipts.items()},
            artifact_sha256=artifacts))
    return dict(status='REQUIRES_UNIFORM_FP32_FALLBACK' if asymmetric else 'MATCHED_SUCCESSFUL_UPDATE_STREAMS',
        backbone=backbone, precision=precision, formal_seeds=list(FORMAL_SEEDS), pairs=pairs,
        asymmetric_seed_count=asymmetric, fallback_pair_count=5 if asymmetric else 0,
        training_code_sha256=code_hash(), H_data_freeze_sha256=config['H_data_freeze_sha256'],
        config_path=str(Path(config_path).resolve()), before_target_export=True,
        target_performance_used=False, utc=utc())


class HQueue:
    def __init__(self, root, code, config_path=None):
        self.root = Path(root); self.code = Path(code)
        self.path = Path(config_path) if config_path else self.root / 'configs/H_execution.json'
        self.config = json.loads(self.path.read_text()) if self.path.exists() else json.loads((self.root / 'configs/base.json').read_text())
        self.jobs = {}; self.completed = []
        status = self.root / 'H_queue_status.json'
        if status.exists() and any(owned_worker_alive(item) for item in json.loads(status.read_text()).get('active', [])):
            raise RuntimeError('existing H worker still active; refuse duplicate supervisor')
        self.logs = self.root / 'logs'; self.logs.mkdir(parents=True, exist_ok=True)

    def status(self, stage, detail=None):
        active = [dict(pid=item['process'].pid, run_id=key, gpu=item['gpu'], uuid=item['uuid'],
                       backbone=item.get('backbone')) for key, item in self.jobs.items()]
        save_json(self.root / 'H_queue_status.json', dict(owner_pid=os.getpid(), stage=stage, utc=utc(),
            active=active, completed_this_session=self.completed, detail=detail,
            resource_policy='D_PRIORITY; idle GPU0-3 only; global lifetime locks'))

    def _verified_completion(self, run):
        from .worker_h import verify_run_completion, finalize_completion_seal
        if not (self.root / 'runs' / run / 'complete.json').is_file():
            return None
        if not (self.root / 'runs' / run / 'H_completion_seal.json').is_file():
            finalize_completion_seal(self.path, run)
        return verify_run_completion(self.path, run)

    def _devices(self):
        d = d_priority_ready(self.root)
        if not d['ready']:
            return [], d
        memory = gpu_memory()
        compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        occupied = {line.split(', ')[0] for line in compute.strip().splitlines() if line.strip()}
        locked = {m['uuid'] for i, m in memory.items() if i in (0, 1, 2, 3) and reserved(self.root, m['uuid'])}
        free = dispatchable_devices(self.config, memory, occupied, locked, d_ready=True)
        return [(i, memory[i]['uuid']) for i in free if i not in {x['gpu'] for x in self.jobs.values()}], d

    def _launch(self, key, gpu, uuid, *, qualification=None):
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', PYTHONPATH=str(self.code))
        command = [str(self.root / 'environment/bin/python'), '-u', '-m']
        if qualification:
            b, precision, micro = qualification
            command += ['reliability_v2.qualification_h', 'worker', '--root', str(self.root),
                        '--backbone', b, '--precision', precision, '--micro', str(micro)]
        else:
            command += ['reliability_v2.worker_h', '--config', str(self.path), '--run-id', key]
        with (self.logs / (key + '.log')).open('a') as stream:
            process = subprocess.Popen(command, cwd=self.code, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
        self.jobs[key] = dict(process=process, gpu=gpu, uuid=uuid,
                             backbone=qualification[0] if qualification else self.config['runs'][key]['backbone'])

    def batch(self, run_ids, stage):
        pending = []
        for run in run_ids:
            if self._verified_completion(run):
                self.completed.append(run)
            else:
                pending.append(run)
        while pending or self.jobs:
            for run, item in list(self.jobs.items()):
                rc = item['process'].poll()
                if rc is None:
                    continue
                del self.jobs[run]
                if rc != 0 or not self._verified_completion(run):
                    self.status('FAILED_SOURCE_REVIEW_REQUIRED', dict(run_id=run, returncode=rc))
                    raise RuntimeError('H worker failed; source review required: ' + run)
                self.completed.append(run)
            capacity = capacity_check(shutil.disk_usage(self.root).free)
            if not capacity['allowed']:
                self.status('CAPACITY_REVIEW_REQUIRED', capacity)
                raise RuntimeError('H storage below reserve; preserve existing data and review capacity')
            free, d = self._devices() if pending else ([], d_priority_ready(self.root))
            for gpu, uuid in free:
                if not pending:
                    break
                self._launch(pending.pop(0), gpu, uuid)
            self.status(stage if d['ready'] else 'WAITING_D_GPU_QUEUE_PRIORITY', dict(pending=pending, requested_stage=stage, capacity=capacity))
            if pending or self.jobs:
                time.sleep(15)
        self.status(stage + '_COMPLETE', dict(run_ids=run_ids))

    def qualification(self):
        from . import qualification_h as qual
        qual.prepare_probe(self.root)
        pending = []; active = {}; passed = []
        for b in BACKBONES:
            for precision in ('AMP', 'FP32'):
                next_attempt = qual.next_attempt(self.root, b, precision)
                if next_attempt['status'] == 'COMPLETE':
                    passed.append(next_attempt)
                else:
                    pending.append((b, precision, next_attempt['microbatch']))
        while pending or self.jobs:
            for key, item in list(self.jobs.items()):
                rc = item['process'].poll()
                if rc is None:
                    continue
                del self.jobs[key]; b, precision, micro = active.pop(key)
                folder = self.root / f'qualification/H/{b}/{precision}/micro{micro}'
                if rc == 0 and (folder / 'receipt.json').exists():
                    passed.append(dict(backbone=b, precision=precision, microbatch=micro))
                elif (folder / 'failure.json').exists() and json.loads((folder / 'failure.json').read_text()).get('status') == 'OOM_RETRY_ALLOWED' and micro > 8:
                    pending.insert(0, (b, precision, {32: 16, 16: 8}[micro]))
                else:
                    raise RuntimeError(f'H source qualification failed {key}; preserve evidence and review')
            capacity = capacity_check(shutil.disk_usage(self.root).free)
            if not capacity['allowed']:
                raise RuntimeError('H qualification capacity below reserve')
            free, d = self._devices() if pending else ([], d_priority_ready(self.root))
            for gpu, uuid in free:
                if not pending:
                    break
                spec = pending.pop(0); b, precision, micro = spec
                key = f'H_qual_{b}_{precision}_micro{micro}'; active[key] = spec
                self._launch(key, gpu, uuid, qualification=spec)
            self.status('H_SOURCE_QUALIFICATION' if d['ready'] else 'WAITING_D_GPU_QUEUE_PRIORITY', dict(pending=pending, passed=passed))
            if pending or self.jobs:
                time.sleep(15)
        return qual.accept_qualification(self.root)

    def _register(self, tasks):
        for key, value in tasks.items():
            if key in self.config['runs'] and self.config['runs'][key] != value:
                raise ValueError('H task registration changed identity')
            self.config['runs'][key] = value
        save_json(self.path, self.config)

    def histories(self, ids):
        from .worker_h import read_sealed_history
        return [read_sealed_history(self.root, key) for key in ids]

    def run(self):
        from .data_h import validate_h_freeze
        from .qualification_h import validate_qualification
        validate_h_freeze(self.root)
        initial = capacity_check(shutil.disk_usage(self.root).free, initial=True)
        if not initial['allowed']:
            raise RuntimeError('H initial budget requires estimated200GiB plus50GiB reserve')
        if (self.root / 'qualification/H_gate.json').exists():
            validate_qualification(self.root)
        else:
            self.qualification()
        self.config = prepare_execution(self.root)
        tasks = standard_development_tasks(self.config); self._register(tasks)
        self.batch(list(tasks), 'H_STANDARD_DEVELOPMENT')
        recipes = {}
        for b in BACKBONES:
            lrs = (1e-5, 3e-5, 5e-5) if b == 'dinov2_vit_b14' else (1e-5,) if b == 'clip_vit_b16' else (1e-4,)
            candidates = []
            for lr in lrs:
                ids = [development_id(b, history, seed, lr, 30) for seed in DEVELOPMENT_SEEDS for history in HISTORIES]
                candidates.append(dict(lr=lr, epochs=30, run_ids=ids, histories=self.histories(ids)))
            chosen = choose_shared_recipe(candidates); epochs = 30; ids = chosen['run_ids']
            if b == 'dinov2_vit_b14':
                while True:
                    extension = extend_budget(self.histories(ids), epochs)
                    if extension is None:
                        break
                    epochs = extension; more = {}
                    for seed in DEVELOPMENT_SEEDS:
                        for history in HISTORIES:
                            key = development_id(b, history, seed, chosen['lr'], epochs)
                            more[key] = registered_task(self.config, b, history, seed, chosen['lr'], epochs, 'development')
                    self._register(more); ids = list(more)
                    self.batch(ids, f'H_DINO_FRESH_{epochs}_DEVELOPMENT')
            recipes[b] = dict(lr=chosen['lr'], epochs=epochs, final_development_run_ids=ids,
                source_30epoch_grid=[dict(lr=c['lr'], last5_fourway_mean=float(np.mean([
                    np.mean([r['dev']['macro_auc'] for r in hist[-5:]]) for hist in c['histories']]))) for c in candidates],
                shared_B_AB_recipe=True, selection_source_only=True)
        recipe_path = self.root / 'configs/H_frozen_recipes.json'
        recipe = dict(recipes=recipes, H_data_freeze_sha256=self.config['H_data_freeze_sha256'],
            h_code_bundle_sha256=self.config['h_code_bundle_sha256'], target_scores_used=False)
        if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
            raise ValueError('H frozen recipe changed')
        save_json(recipe_path, recipe)
        formal = formal_tasks(self.config, recipes, recipe_sha256=digest(recipe_path)); self._register(formal)
        registry_path = self.root / 'H_run_registry.json'
        save_json(registry_path, dict(schema='H_RUN_REGISTRY_V2', formal_model_count=30,
            standard_development_count=20, maximum_extra_dino_development_count=8,
            formal_run_ids=list(formal), effective_runs={}, target_export_allowed=False))
        self.batch(list(formal), 'H_FORMAL_AMP_TRAINING')
        effective = {}
        for b in BACKBONES:
            ledger = audit_paired_ledgers(self.path, b)
            save_json(self.root / f'reports/H_{b}_AMP_paired_ledger.json', ledger)
            precision = 'AMP'
            if ledger['status'] == 'REQUIRES_UNIFORM_FP32_FALLBACK':
                amp_roots = [self.root / 'runs' / key for key, t in formal.items() if t['backbone'] == b]
                projected = int(1.2 * sum(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) for folder in amp_roots))
                capacity = capacity_check(shutil.disk_usage(self.root).free, additional_bytes=projected)
                save_json(self.root / f'reports/H_{b}_FP32_capacity.json', capacity)
                if not capacity['allowed']:
                    raise RuntimeError('uniform H FP32 fallback requires capacity review; no selective rerun')
                reruns = formal_tasks(self.config, recipes, recipe_sha256=digest(recipe_path), precision='FP32', backbone=b)
                self._register(reruns); self.batch(list(reruns), f'H_{b}_UNIFORM_FIVE_PAIR_FP32')
                ledger = audit_paired_ledgers(self.path, b, precision='FP32'); precision = 'FP32'
                save_json(self.root / f'reports/H_{b}_FP32_paired_ledger.json', ledger)
            if ledger['status'] != 'MATCHED_SUCCESSFUL_UPDATE_STREAMS':
                raise RuntimeError('H numerical ledger did not permit target export')
            effective[b] = dict(precision=precision, run_ids=[formal_id(b, h, s, precision) for s in FORMAL_SEEDS for h in HISTORIES])
            save_json(self.root / f'qualification/H_target_export_{b}.json', dict(status='PASS',
                H_data_freeze_sha256=self.config['H_data_freeze_sha256'], h_code_bundle_sha256=self.config['h_code_bundle_sha256'],
                precision=precision, backbone=b, all_five_pairs_checked=True, effective_runs=effective[b]['run_ids'],
                paired_ledger_path=str(self.root / f'reports/H_{b}_{precision}_paired_ledger.json'),
                paired_ledger_sha256=digest(self.root / f'reports/H_{b}_{precision}_paired_ledger.json'),
                target_performance_used=False))
        registry = json.loads(registry_path.read_text()); registry.update(effective_runs=effective, target_export_allowed=True,
            status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING')
        save_json(registry_path, registry)
        self.status('H_FORMAL_COMPLETE_TARGET_CACHE_ANALYSIS_PENDING', dict(effective_runs=effective))


def readiness(root):
    root = Path(root)
    return dict(standard_development_models=20, formal_models=30, maximum_additional_DINO_models=8,
        maximum_conditional_FP32_formal_reruns=30, H_freeze_present=(root / 'data/H/freeze.json').is_file(),
        H_gate_present=(root / 'qualification/H_gate.json').is_file(), D_priority=d_priority_ready(root),
        GPU_launch_attempted=False, modifies_D_configuration=False,
        next='full H pixels + fresh audited freeze, then H qualification on D-priority-approved idle devices')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('status', 'run'))
    p.add_argument('--root', required=True); p.add_argument('--code')
    args = p.parse_args(); root = Path(args.root)
    if args.action == 'status':
        print(json.dumps(readiness(root), indent=2)); return
    if not args.code:
        p.error('run requires --code')
    with exclusive_file(root / 'H_pipeline.lock'):
        queue = HQueue(root, args.code)
        try:
            queue.run()
        except BaseException:
            queue.status('FAILED_SOURCE_REVIEW_REQUIRED', dict(traceback=traceback.format_exc(),
                         existing_workers_continue_until_own_completion=True))
            raise


if __name__ == '__main__':
    main()
