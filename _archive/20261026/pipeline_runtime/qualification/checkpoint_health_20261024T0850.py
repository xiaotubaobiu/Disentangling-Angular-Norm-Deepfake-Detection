"""Read-only CPU health inspection of existing, owned D training checkpoints."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import gc
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch

from reliability_v2 import trainer

ROOT = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_check(value, counts):
    if torch.is_tensor(value):
        assert value.device.type == 'cpu'
        counts['tensors'] += 1
        counts['elements'] += value.numel()
        if value.is_floating_point() or value.is_complex():
            assert bool(torch.isfinite(value).all()), 'nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for part in value.values():
            tensor_check(part, counts)
    elif isinstance(value, (tuple, list)):
        for part in value:
            tensor_check(part, counts)


def inspect_run(item, config, hashes):
    started = time.perf_counter()
    run = item['run_id']; task = config['runs'][run]; out = ROOT / 'runs' / run
    if item.get('process_required', True):
        assert Path(f'/proc/{item["pid"]}').exists()
        cmdline = Path(f'/proc/{item["pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert run in cmdline and 'reliability_v2.trainer' in cmdline
    expected = dict(config=hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
        train_manifest=hashes['train'], dev_manifest=hashes['dev'],
        initialization=hashes[task['backbone']], environment=hashes['environment'],
        code=trainer.code_hash(), qualification=hashes['gate'])
    assert expected['initialization'] == config['backbones'][task['backbone']]['weight_sha256']
    assert json.loads((out / 'identity.json').read_text()) == expected
    assert json.loads((out / 'resolved_config.json').read_text()) == task
    # Open the committed inode once. Atomic replacement by the live writer
    # cannot make the digest and deserialized contents refer to different files.
    checkpoint = out / 'state.pt'
    with checkpoint.open('rb') as stream:
        stat = os.fstat(stream.fileno()); h = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=False)
    assert state['identity'] == expected
    assert state['scheduler_clock'] == 'attempt_count'
    epoch = state['epoch']; steps = 799
    assert state['attempts'] == epoch * steps
    assert [row['epoch'] for row in state['history']] == list(range(1, epoch + 1))
    external = json.loads((out / 'history.json').read_text())
    common = min(len(external), epoch)
    assert external[:common] == state['history'][:common]
    assert abs(len(external) - epoch) <= 1, 'external/checkpoint history drift'
    assert 0 <= state['consecutive_skips'] < 8
    best = max(row['dev']['macro_auc'] for row in state['history'])
    assert best == state['best_score']
    assert state['history'][state['best_epoch'] - 1]['dev']['macro_auc'] == best
    assert set(state['best_model']) == set(state['model'])
    for name in state['model']:
        assert state['best_model'][name].shape == state['model'][name].shape
    dimension = {'clip_vit_b16': 512, 'dinov2_vit_b14': 768, 'efficientnet_b0': 1280}[task['backbone']]
    assert tuple(state['model']['classifier.weight'].shape) == (2, dimension)
    counts = Counter(); tensor_check(state['model'], counts)
    tensor_check(state['best_model'], counts); tensor_check(state['optimizer'], counts)
    opt = state['optimizer']; ids = [i for group in opt['param_groups'] for i in group['params']]
    assert set(ids) == set(opt['state']), 'optimizer does not cover every parameter'
    success = sum(row['success'] for row in state['history'])
    optsteps = sorted(set(int(x['step'].item()) for x in opt['state'].values()))
    assert optsteps == [success]
    lr = trainer.attempt_lr(state['attempts'] - 1, steps, task['epochs'], task['lr'])
    assert all(math.isclose(group['lr'], lr, rel_tol=0, abs_tol=1e-15) for group in opt['param_groups'])
    assert state['scaler'] is not None and state['scaler']['scale'] > 0
    random.Random().setstate(state['python_rng'])
    np.random.RandomState().set_state(state['numpy_rng'])
    torch.Generator().set_state(state['torch_rng'])
    assert len(state['cuda_rng']) == 1 and state['cuda_rng'][0].dtype == torch.uint8
    # Validate every committed epoch's complete step ledger. The current epoch
    # is observed but never edited or treated as a committed recovery point.
    events_count = 0; skipped = 0; nonfinite_skipped = 0
    for row in state['history']:
        logs = [json.loads(line) for line in (out / f'epoch_{row["epoch"]:02d}_updates.jsonl').open()]
        assert len(logs) == steps
        assert [e['attempt'] for e in logs] == list(range((row['epoch'] - 1) * steps, row['epoch'] * steps))
        assert sum(e['success'] for e in logs) == row['success']
        assert sum(e['skip'] for e in logs) == row['skip']
        for e in logs:
            assert math.isfinite(e['loss']) and e['success'] != e['skip']
            assert math.isclose(e['lr'], trainer.attempt_lr(e['attempt'], steps, task['epochs'], task['lr']), rel_tol=0, abs_tol=1e-15)
            assert e['scale_before'] > 0 and e['scale_after'] > 0
            if not e['gradient_finite']:
                assert e['skip']; nonfinite_skipped += 1
            else:
                assert math.isfinite(e['gradient_norm'])
        events_count += len(logs); skipped += row['skip']
    array_audits = []
    for row in state['history'][-2:]:
        e = row['epoch']
        for field, suffix in [('sample_sha256', 'samples'), ('augmentation_sha256', 'augmentations')]:
            p = out / f'epoch_{e:02d}_{suffix}.npy'
            assert digest(p) == row[field]
            array_audits.append(dict(epoch=e, kind=suffix, sha256=row[field]))
    updates = sorted(out.glob('epoch_*_updates.jsonl'))
    newest = updates[-1]
    lines = newest.read_text().splitlines()
    if lines:
        try: current = json.loads(lines[-1])
        except json.JSONDecodeError: current = json.loads(lines[-2]) if len(lines) > 1 else None
    else: current = None
    recent = state['history'][-min(3, epoch):]
    recent_seconds = float(np.mean([row['seconds'] for row in recent]))
    report = dict(**item, status='PASS', checkpoint_epoch=epoch, history_epoch_at_observation=len(external),
        attempts=state['attempts'], successful_updates=success, AMP_skips=skipped,
        checkpoint_sha256=h.hexdigest(), checkpoint_inode=stat.st_ino, checkpoint_bytes=stat.st_size,
        checkpoint_mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        checkpoint_age_seconds=time.time() - stat.st_mtime,
        live_path_replaced_during_inspection=checkpoint.stat().st_ino != stat.st_ino,
        identity=expected, best_epoch=state['best_epoch'], optimizer_parameter_states=len(opt['state']),
        optimizer_step_counts=optsteps, scaler=state['scaler'], tensor_finiteness=dict(counts),
        RNG_restoration_checked='independent Python/NumPy/CPU Torch generators; CUDA RNG shape checked without CUDA initialization',
        committed_step_log_records=events_count, skipped_nonfinite_gradients=nonfinite_skipped,
        recent_array_hashes=array_audits, current_update_record=current,
        newest_update_file=newest.name, newest_update_age_seconds=time.time() - newest.stat().st_mtime,
        latest_epoch_seconds=state['history'][-1]['seconds'], recent_3_epoch_mean_seconds=recent_seconds,
        recent_effective_input_rate_per_second=102272 / recent_seconds,
        timing_scope='sampling+training+full DEV; following checkpoint write excluded',
        verification_seconds=time.perf_counter() - started)
    # Do not include source validation performance in the progress summary.
    if report['current_update_record'] is not None:
        report['current_update_record'] = {k: report['current_update_record'][k] for k in
            ('epoch', 'attempt', 'lr', 'loss', 'gradient_finite', 'scale_after', 'success', 'skip')}
    del state; gc.collect()
    return report


def expected_identity(task,config,hashes):
    return dict(config=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),train_manifest=hashes['train'],dev_manifest=hashes['dev'],initialization=hashes[task['backbone']],environment=hashes['environment'],code=trainer.code_hash(),qualification=hashes['gate'])


def first_epoch(item,config,hashes):
    run=item['run_id'];out=ROOT/'runs'/run;task=config['runs'][run]
    p=Path('/proc')/str(item['pid']);assert p.exists()
    cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert run in cmd and 'reliability_v2.trainer' in cmd
    expected=expected_identity(task,config,hashes)
    assert json.loads((out/'identity.json').read_text())==expected
    assert json.loads((out/'resolved_config.json').read_text())==task
    assert not (out/'complete.json').exists()
    files=[dict(name=x.name,bytes=x.stat().st_size,mtime_utc=datetime.fromtimestamp(x.stat().st_mtime,timezone.utc).isoformat()) for x in out.glob('epoch_01_*')]
    return dict(**item,status='PENDING_FIRST_EPOCH_RECOVERY_POINT',identity=expected,files=files,checkpoint_epoch=0,committed_attempts=0,scope='Identity and live PID verified; no committed checkpoint yet, not corruption and not a recovery-integrity PASS')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    config=json.loads((ROOT/'configs/execution.json').read_text());queue=json.loads((ROOT/'queue_status.json').read_text())
    hashes={key:digest(path) for key,path in {'train':ROOT/'data/D/train.parquet','dev':ROOT/'data/D/dev.parquet','environment':ROOT/'configs/requirements.execution.lock.txt','gate':ROOT/'qualification/D_gate.json',**{key:Path(v['weight_path']) for key,v in config['backbones'].items()}}.items()}
    items=[]
    for run,task in config['runs'].items():
        if (ROOT/'runs'/run/'complete.json').is_file():items.append(dict(run_id=run,pid=None,process_required=False,lifecycle_at_inspection='COMPLETE'))
    for item in queue['active']:
        if item['run_id'] not in [x['run_id'] for x in items]:items.append(dict(**item,process_required=True,lifecycle_at_inspection='RUNNING'))
    results=[]
    for item in items:
        try:
            out=ROOT/'runs'/item['run_id']
            if (out/'complete.json').exists():
                item={**item,'process_required':False,'lifecycle_at_inspection':'COMPLETE'}
            if (out/'state.pt').exists():report=inspect_run(item,config,hashes)
            else:report=first_epoch(item,config,hashes)
        except Exception as error:report=dict(**item,status='FAILED_INSPECTION',exception_type=type(error).__name__,error=str(error))
        results.append(report)
        print(json.dumps({k:report[k] for k in ('run_id','status','lifecycle_at_inspection','checkpoint_epoch','attempts','AMP_skips','recent_effective_input_rate_per_second','error') if k in report}),flush=True)
    ok=all(r['status'] in ['PASS','PENDING_FIRST_EPOCH_RECOVERY_POINT'] for r in results)
    receipt=dict(utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,status='PASS_WITH_EXPLICIT_FIRST_EPOCH_PENDING' if ok and any(r['status']!='PASS' for r in results) else 'PASS' if ok else 'INSPECTION_REQUIRES_REVIEW',checks=results,queue_snapshot=queue,scope='read-only pinned committed checkpoint inspection including completed runs; new first epoch pending separately; no live restore',CUDA_initialized=torch.cuda.is_initialized())
    trainer.save_json(ROOT/'reports/checkpoint_health_20261024T0850.json',receipt)
    assert ok,receipt

if __name__=='__main__':main()
