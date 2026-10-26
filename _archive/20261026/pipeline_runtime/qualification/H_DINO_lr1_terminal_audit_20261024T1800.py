"""Read-only CPU audit of atomic H development checkpoints and paired streams."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import gc
import random
import time
import numpy as np
import pandas as pd
import torch
import bisect, math
from reliability_v2 import worker_h

R = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
REPORT = R / 'reports/H_DINO_lr1_terminal_acceptance_20261024T1800.json'
assert not REPORT.exists()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tensor_tree(value):
    count = 0
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
        if value.is_floating_point() or value.is_complex():
            assert torch.isfinite(value).all().item()
        return 1
    if isinstance(value, dict):
        for item in value.values():
            count += tensor_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            count += tensor_tree(item)
    return count


queue = {'active': [{'run_id': 'H_dev_dinov2_vit_b14_B_lr1e-05_seed1007_e30', 'pid': 2035767}, {'run_id': 'H_dev_dinov2_vit_b14_AB_lr1e-05_seed1007_e30', 'pid': 2035768}, {'run_id': 'H_dev_dinov2_vit_b14_B_lr1e-05_seed1042_e30', 'pid': 2035769}, {'run_id': 'H_dev_dinov2_vit_b14_AB_lr1e-05_seed1042_e30', 'pid': 2036656}]}
config = read(R / 'configs/H_execution.json')
gate_hash = sha(R / 'qualification/H_gate.json')
results = []
start = time.monotonic()
for job in queue['active']:
    run = job['run_id']
    task = config['runs'][run]
    folder = R / 'runs' / run
    assert read(folder/'complete.json')['status']=='TRAINING_COMPLETE'
    original_receipt = worker_h.verify_run_completion(R/'configs/H_execution.json',run)
    with (folder / 'state.pt').open('rb') as stream:
        stat = os.fstat(stream.fileno())
        state = torch.load(stream, map_location='cpu', weights_only=False)
        stream.seek(0)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(stream.fileno())
    assert (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    identity = read(folder / 'identity.json')
    assert state['identity'] == identity
    assert identity['qualification'] == gate_hash
    assert identity['train_manifest'] == sha(R / task['train_manifest'])
    assert identity['dev_manifest'] == sha(R / task['dev_manifest'])
    epoch, attempts = int(state['epoch']), int(state['attempts'])
    assert 1 <= epoch <= task['epochs'] == 30
    assert attempts == epoch * 191 and state['scheduler_clock'] == 'attempt_count'
    history = state['history']
    assert len(history) == epoch and [h['epoch'] for h in history] == list(range(1, epoch + 1))
    assert all(h['draw_count'] == 24448 for h in history)
    events = []
    for e in range(1, epoch + 1):
        path = folder / f'epoch_{e:02}_updates.jsonl'
        raw = path.read_bytes()
        assert raw.endswith(b'\n')
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(records) == 191 and all(r['epoch'] == e for r in records)
        events.extend(records)
    assert [r['attempt'] for r in events] == list(range(attempts))
    success = sum(bool(r['success']) for r in events)
    skips = [r for r in events if r['skip']]
    assert success + len(skips) == attempts
    ids = [p for group in state['optimizer']['param_groups'] for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(state['optimizer']['state'])
    optimizer_steps = []
    for parameter in state['optimizer']['state'].values():
        step = parameter['step']
        step = step.item() if isinstance(step, torch.Tensor) else step
        assert int(step) == success
        optimizer_steps.append(int(step))
    for group in state['optimizer']['param_groups']:
        assert group['lr'] == events[-1]['lr'] and group['weight_decay'] == 5e-4
    finite_tensors = tensor_tree(state['model']) + tensor_tree(state['best_model']) + tensor_tree(state['optimizer'])
    assert tuple(state['model']['classifier.weight'].shape) == (2, 768)
    assert tuple(state['model']['classifier.bias'].shape) == (2,)
    assert state['scaler'] and float(state['scaler']['scale']) > 0
    py = random.Random(); py.setstate(state['python_rng'])
    nr = np.random.RandomState(); nr.set_state(state['numpy_rng'])
    tg = torch.Generator(device='cpu'); tg.set_state(state['torch_rng'])
    cuda = state['cuda_rng']
    assert isinstance(cuda, list) and len(cuda) == 1
    assert all(x.dtype == torch.uint8 and x.ndim == 1 and x.numel() > 0 for x in cuda)
    trailing = 0
    for event in reversed(events):
        if not event['skip']:
            break
        trailing += 1
    assert state['consecutive_skips'] == trailing
    assert state['best_model'] is not None and 1 <= state['best_epoch'] <= epoch
    recovery = []
    for event in skips:
        following = events[event['attempt'] + 1:]
        recovery.append(dict(epoch=event['epoch'], attempt=event['attempt'],
                             next_success=next((r['attempt'] for r in following if r['success']), None)))
    result = dict(run_id=run, pid=job['pid'], snapshot_epoch=epoch, snapshot_attempts=attempts,
                  snapshot_successful_updates=success, snapshot_AMP_skips=len(skips),
                  skip_recovery_within_snapshot=recovery, state_sha256=digest,
                  state_inode_at_open=stat.st_ino, state_bytes=stat.st_size,
                  path_still_same_inode=(folder / 'state.pt').stat().st_ino == stat.st_ino,
                  finite_model_optimizer_best_tensors=finite_tensors, optimizer_parameter_states=len(ids),
                  optimizer_step_min=min(optimizer_steps), optimizer_step_max=max(optimizer_steps),
                  scaler_scale=state['scaler']['scale'], scheduler_clock=state['scheduler_clock'],
                  python_numpy_cpu_torch_RNG_restore_to_local_generators=True,
                  saved_CUDA_RNG_count=1,
                  CUDA_RNG_restored_on_GPU_in_this_audit=False,
                  CUDA_RNG_sha256=hashlib.sha256(cuda[0].numpy().tobytes()).hexdigest(),
                  mean_committed_epoch_seconds=float(np.mean([h['seconds'] for h in history])),
                  frozen_train_draws_per_epoch=24448, frozen_attempts_per_epoch=191,
                  status='PASS_COMPLETE_ATOMIC_CHECKPOINT_CPU_SNAPSHOT')
    assert epoch == 30 and attempts == 5730
    assert original_receipt['success']==success and original_receipt['skip']==len(skips) and original_receipt['target_used'] is False
    final=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    best=torch.load(folder/'source_best.pt',map_location='cpu',weights_only=False)
    assert final['identity']==identity and final['epoch']==30
    assert set(final['model'])==set(state['model']) and all(torch.equal(v,final['model'][k]) for k,v in state['model'].items())
    assert set(best['model'])==set(state['best_model']) and all(torch.equal(v,best['model'][k]) for k,v in state['best_model'].items())
    assert original_receipt['source_best_epoch']==state['best_epoch']
    del final,best
    def independent_auc(scores,labels):
        negative=sorted(float(v) for v,y in zip(scores,labels) if int(y)==0)
        positive=[float(v) for v,y in zip(scores,labels) if int(y)==1]
        assert negative and positive and all(math.isfinite(v) for v in negative+positive)
        twice=sum(bisect.bisect_left(negative,v)+bisect.bisect_right(negative,v) for v in positive)
        return twice/(2*len(negative)*len(positive))
    dev_errors=[]
    for entry in history:
        table=pd.read_parquet(folder/f"epoch_{entry['epoch']:02}_source_dev.parquet")
        assert list(table.columns)==['video_uid','method_id','label','score'] and len(table)==535 and table.video_uid.is_unique
        actual=independent_auc(table.score,table.label);dev_errors.append(abs(actual-entry['dev']['pooled_auc']))
        methods=[]
        for method,reported in entry['dev']['method_auc'].items():
            select=table[(table.label==0)|(table.method_id==method)]
            value=independent_auc(select.score,select.label);methods.append(value);dev_errors.append(abs(value-reported))
        assert len(methods)==4
        dev_errors.append(abs(math.fsum(methods)/4-entry['dev']['macro_auc']))
    assert max(dev_errors)<=3e-15
    result.update(terminal_complete_sha256=sha(folder/'complete.json'),terminal_seal_sha256=sha(folder/'H_completion_seal.json'),seal_current_artifacts_verified=158,final_matches_complete_state=True,source_best_matches_best_state=True,source_best_epoch=state['best_epoch'],source_DEV_tables_independently_recomputed=30,source_DEV_metric_checks=len(dev_errors),source_DEV_max_auc_abs_error=max(dev_errors),target_used=False)
    results.append(result)
    print(json.dumps(result), flush=True)
    del state, events, history
    gc.collect()

tables = {h: pd.read_parquet(R / f'data/H/{h}/train.parquet') for h in ['B', 'AB']}
assert len(tables['B']) == len(tables['AB']) == 24424
assert tables['B'].mother_id.tolist() == tables['AB'].mother_id.tolist()
pairs = []
for seed in [1007, 1042]:
    members = [x for x in results if f'seed{seed}_' in x['run_id']]
    assert len(members) == 2
    epochs = min(x['snapshot_epoch'] for x in members)
    hashes = []
    for epoch in range(1, epochs + 1):
        arrays = []
        for item in members:
            folder = R / 'runs' / item['run_id']
            arrays.append({key: np.load(folder / f'epoch_{epoch:02}_{key}.npy', allow_pickle=False)
                           for key in ['samples', 'methods', 'augmentations']})
        for key in arrays[0]:
            assert np.array_equal(arrays[0][key], arrays[1][key])
        assert len(arrays[0]['samples']) == 24448
        np.testing.assert_array_equal(tables['B'].mother_id.to_numpy()[arrays[0]['samples']],
                                      tables['AB'].mother_id.to_numpy()[arrays[1]['samples']])
        hashes.append(dict(epoch=epoch, samples_sha256=sha(R/'runs'/members[0]['run_id']/f'epoch_{epoch:02}_samples.npy')))
    pairs.append(dict(seed=seed, common_snapshot_epochs_checked=epochs,
                      samples_methods_augmentations_and_actual_mother_sequence_equal=True, hashes=hashes))

assert not torch.cuda.is_initialized()
report = dict(utc=datetime.now(timezone.utc).isoformat(), status='PASS_H_DINO_LR1_DEVELOPMENT_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS',
              source_only=True, records=results, paired_streams=pairs, audit_seconds=time.monotonic()-start,
              CUDA_initialized=False, production_processes_restarted=False,
              scope='First four standard H development runs only, all E30 terminal. Original158-artifact seals revalidated; independent complete state/final/best/counts/AdamW/RNG/pair mother streams and120 source DEV tables recomputed by stdlib bisect rank. CPU RNG restored locally; saved CUDA RNG not restored onto GPU here. No H formal or target performance acceptance.')
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(status=report['status'], report=str(REPORT), audit_seconds=report['audit_seconds'])), flush=True)
