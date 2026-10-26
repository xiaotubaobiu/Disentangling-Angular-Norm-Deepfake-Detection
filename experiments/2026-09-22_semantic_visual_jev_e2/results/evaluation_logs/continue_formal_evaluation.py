#!/usr/bin/env python3
"""Operational coordinator only; does not change scientific source or settings."""
import csv
import datetime as dt
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path('/home/z/project/deepfake_detection')
EXP = REPO / 'experiments/2026-09-22_semantic_visual_jev_e2'
RESULTS = EXP / 'results'
LOGS = RESULTS / 'evaluation_logs'
PROGRESS = RESULTS / 'progress'
DOMAINS = ['ffpp', 'cdf', 'dfd']
CONDITIONS = ['fixed_h1', 'seen', 'heldout', 'permutation', 'counterfactual', 'meaningless_candidates', 'meaningless_questions']
WORKERS = [2701438, 2701439, 2701440, 2701441]
RUN = LOGS / dt.datetime.now(dt.timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
RUN.mkdir(parents=True, exist_ok=False)
EVENTS = RUN / 'events.jsonl'
ENV = os.environ.copy()
ENV.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def event(kind, **data):
    row = dict(timestamp_utc=now(), event=kind, **data)
    with EVENTS.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush()
    print(json.dumps(row, sort_keys=True), flush=True)
    write_json(LOGS / 'current_run.json', dict(run_directory=str(RUN), coordinator_pid=os.getpid(), **row))

def worker_alive(pid):
    try:
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        return 'train_semantic.py' in cmdline and '--seed 2048' in cmdline
    except FileNotFoundError:
        return False

def training_records():
    paths = list((RESULTS / 'training_logs/seed_2048').glob('*/attempt_0/0/stdout.log'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one seed2048 rank0 stdout log, found {paths}')
    records = []
    for line in paths[0].read_text().splitlines():
        if line.startswith('{'):
            records.append(json.loads(line))
    return paths[0], records

def save_training_history(epochs):
    path = PROGRESS / 'seed_2048_history.csv'
    temporary = path.with_suffix('.csv.tmp')
    keys = ['epoch', 'train_ce', 'val_seen_ce', 'val_seen_accuracy', 'val_seen_count', 'source']
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in epochs:
            writer.writerow({key: ('rank_0_persistent_stdout_log' if key == 'source' else row[key]) for key in keys})
    temporary.replace(path)

def wait_and_validate_training():
    last_epoch = 0
    event('waiting_for_existing_training', worker_pids=WORKERS)
    while True:
        path, records = training_records()
        epochs = [row for row in records if 'epoch' in row]
        if len(epochs) != last_epoch:
            save_training_history(epochs)
            for row in epochs[last_epoch:]:
                event('training_epoch_completed', seed=2048, **{key: row[key] for key in ['epoch', 'train_ce', 'val_seen_ce', 'val_seen_accuracy']})
            last_epoch = len(epochs)
        if not any(worker_alive(pid) for pid in WORKERS):
            break
        time.sleep(10)
    path, records = training_records()
    epochs = [row for row in records if 'epoch' in row]
    summaries = [row for row in records if 'best_epoch' in row and 'seed' in row]
    if len(summaries) != 1 or not epochs:
        raise RuntimeError('Training exited without exactly one final rank-zero summary and nonempty history')
    summary = summaries[0]
    if summary['seed'] != 2048 or [row['epoch'] for row in epochs] != list(range(1, len(epochs) + 1)):
        raise RuntimeError('Training history is incomplete or belongs to a different seed')
    for row in epochs:
        if not all(math.isfinite(row[key]) for key in ['train_ce', 'val_seen_ce', 'val_seen_accuracy']):
            raise RuntimeError(f'Nonfinite training diagnostics at epoch {row["epoch"]}')
        ranks = row['rank_diagnostics']
        if len(ranks) != 4 or {rank['rank'] for rank in ranks} != {0, 1, 2, 3}:
            raise RuntimeError('Training epoch lacks the complete four-rank diagnostics')
        for rank in ranks:
            if rank['train_batches'] != 292 or rank['val_batches'] != 250:
                raise RuntimeError('Unexpected rank batch coverage in the full training run')
            if not 0 < rank['peak_cuda_allocated_bytes'] < 30 * 1024**3:
                raise RuntimeError('Invalid training peak CUDA allocation')
            if any(rank['candidate_counts'].get(width, 0) <= 0 for width in ('2', '3')):
                raise RuntimeError('Training rank lacks both candidate widths')
    for log in (RESULTS / 'training_logs/seed_2048').glob('*/attempt_0/*/stderr.log'):
        if re.search(r'Traceback|RuntimeError|ChildFailedError|DataLoader worker.*(?:exit|kill)|does not have a deterministic', log.read_text()):
            raise RuntimeError(f'Training stderr contains an error requiring investigation: {log}')
    best = min(epochs, key=lambda row: row['val_seen_ce'])
    if summary['best_epoch'] != best['epoch'] or summary['best_val_seen_ce'] != best['val_seen_ce']:
        raise RuntimeError('Final training summary does not agree with the observed history')
    module_path = EXP / 'scripts/evaluate_semantic.py'
    spec = importlib.util.spec_from_file_location('e2_evaluator_checkpoint_validation', module_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    cfg_path = EXP / 'configs/semantic_jev.yaml'
    cfg = evaluator.load_config(str(cfg_path))
    checkpoint_path = RESULTS / 'training/seed_2048/best_model.pth'
    checkpoint = evaluator._load_checkpoint(checkpoint_path)
    expected_hashes = {
        'config_sha256': evaluator.file_sha256(cfg_path),
        'language_manifest_sha256': evaluator.file_sha256(REPO / cfg['language']['manifest']),
        'visual_checkpoint_sha256': evaluator.file_sha256(REPO / cfg['train']['init_visual_checkpoint']),
    }
    evaluator.validate_evaluation_checkpoint(checkpoint, seed=2048, expected_hashes=expected_hashes, allow_smoke=False)
    if checkpoint['epoch'] + 1 != best['epoch'] or checkpoint['val_seen_ce'] != best['val_seen_ce']:
        raise RuntimeError('Actual checkpoint is not the best observed training epoch')
    if checkpoint['rank_diagnostics'] != best['rank_diagnostics']:
        raise RuntimeError('Checkpoint rank diagnostics differ from best-epoch log')
    if len(epochs) != cfg['train']['epochs'] and len(epochs) - best['epoch'] != cfg['train']['patience']:
        raise RuntimeError('Training stopped without satisfying configured patience or maximum epochs')
    stop_reason = ('five_consecutive_epochs_without_seen_validation_ce_improvement' if len(epochs) - best['epoch'] == cfg['train']['patience'] else 'configured_maximum_epochs_completed')
    completion = dict(seed=2048, launcher_exit_code=None, launcher_observation_lost=True,
        all_four_worker_processes_terminated=True, final_rank_zero_summary_present=True,
        all_epoch_rank_diagnostics_verified=True, completed_epochs=len(epochs), stop_reason=stop_reason,
        best_epoch=best['epoch'], best_val_seen_ce=best['val_seen_ce'],
        best_epoch_val_seen_accuracy=best['val_seen_accuracy'], checkpoint=str(checkpoint_path),
        checkpoint_sha256=evaluator.file_sha256(checkpoint_path), reportable=checkpoint['reportable'],
        world_size=checkpoint['world_size'], visible_devices=checkpoint['visible_devices'],
        evaluation_checkpoint_validation_passed=True, verified_at_utc=now(),
        evidence_source='persistent_rank0_epoch_and_final_logs_process_inventory_and_saved_checkpoint_validation',
        rank0_log=str(path))
    save_training_history(epochs)
    write_json(PROGRESS / 'seed_2048_completion.json', completion)
    event('training_completed_and_validated', **completion)

def metric_paths():
    return {str(path) for path in (RESULTS / 'evaluation').glob('seed_*/*/*_metrics.json')}

def report_new_metrics(previous):
    current = metric_paths()
    for filename in sorted(current - previous):
        path = Path(filename)
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f'Empty or invalid metric rows: {path}')
        event('evaluation_group_completed', seed=int(path.parent.parent.name.removeprefix('seed_')),
            condition=path.parent.name, domain=path.name.removesuffix('_metrics.json'),
            metric_rows=len(rows), metric_file=filename, completed_groups=len(current))
    return current

def run_evaluation(seed, conditions, phase, previous):
    output = RESULTS / f'evaluation/seed_{seed}'
    for condition in conditions:
        for domain in DOMAINS:
            expected = output / condition / f'{domain}_metrics.json'
            if expected.exists():
                raise RuntimeError(f'Refusing to overwrite already completed evaluation group: {expected}')
    invocation = RUN / f'{phase}_seed_{seed}'
    invocation.mkdir()
    command = [shutil.which('torchrun') or '/home/z/anaconda3/bin/torchrun', '--standalone', '--nproc_per_node=4',
        '--log-dir', str(invocation / 'rank_logs'), '--redirects', '3',
        str(EXP / 'scripts/evaluate_semantic.py'), '--config', str(EXP / 'configs/semantic_jev.yaml'),
        '--seed', str(seed), '--checkpoint', str(RESULTS / f'training/seed_{seed}/best_model.pth'),
        '--domains', *DOMAINS, '--conditions', *conditions,
        '--feature-cache-root', str(RESULTS / 'feature_cache'), '--output-dir', str(output)]
    manifest = dict(seed=seed, phase=phase, conditions=conditions, domains=DOMAINS,
        argv=command, shell_display=shlex.join(command), cwd=str(REPO),
        environment={key: ENV[key] for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONUNBUFFERED']},
        configured_per_gpu_batch=128, started_at_utc=now())
    write_json(invocation / 'command.json', manifest)
    with (invocation / 'launcher.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        event('evaluation_invocation_started', seed=seed, phase=phase, conditions=conditions,
            launcher_pid=process.pid, command_manifest=str(invocation / 'command.json'))
        while process.poll() is None:
            previous = report_new_metrics(previous)
            time.sleep(10)
        code = process.wait()
    result = dict(**manifest, launcher_pid=process.pid, finished_at_utc=now(), launcher_exit_code=code)
    write_json(invocation / 'completion.json', result)
    previous = report_new_metrics(previous)
    event('evaluation_invocation_finished', seed=seed, phase=phase, launcher_exit_code=code,
        completion_file=str(invocation / 'completion.json'))
    if code != 0:
        raise RuntimeError(f'Evaluation invocation failed with exit code {code}: {invocation}')
    for condition in conditions:
        for domain in DOMAINS:
            if not (output / condition / f'{domain}_metrics.json').exists():
                raise RuntimeError(f'Successful launcher did not produce {seed}/{condition}/{domain}')
    return previous

def main():
    with (LOGS / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        event('coordinator_started', gpu_allocation='0,1,2,3', training_worker_pids=WORKERS)
        previous = metric_paths()
        if previous:
            raise RuntimeError('Preexisting formal metrics require explicit resume planning; no outputs overwritten')
        event('concurrent_frozen_checkpoint_evaluation_authorized', training_seed=2048,
            frozen_evaluation_seeds=[42, 7], free_gpu_memory_mib=[28947, 28977, 28941, 28961],
            available_system_memory_gib=314, note='No training, configuration, checkpoint selection, or scientific source changes')
        for seed in (42, 7):
            previous = run_evaluation(seed, ['fixed_h1'], 'fixed_h1', previous)
        wait_and_validate_training()
        previous = run_evaluation(2048, ['fixed_h1'], 'fixed_h1', previous)
        for seed in (42, 7, 2048):
            previous = run_evaluation(seed, CONDITIONS[1:], 'semantic_controls', previous)
        expected = {str(RESULTS / f'evaluation/seed_{seed}' / condition / f'{domain}_metrics.json') for seed in (42,7,2048) for condition in CONDITIONS for domain in DOMAINS}
        if metric_paths() != expected:
            raise RuntimeError('Final metric matrix differs from the required 63 groups')
        event('formal_evaluation_completed', completed_groups=63, expected_groups=63, all_launcher_exit_codes_zero=True)

try:
    main()
except BaseException as error:
    event('coordinator_failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    raise
