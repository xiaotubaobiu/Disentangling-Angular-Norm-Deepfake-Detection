#!/usr/bin/env python3
"""Operational two-run handoff; preserves every actual torchrun exit status."""
import ast
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

REPO=Path('/home/z/project/deepfake_detection')
EXP=REPO/'experiments/2026-09-22_semantic_visual_jev_e2'
RESULTS=EXP/'results'
LOGS=RESULTS/'evaluation_logs'
DOMAINS=['ffpp','cdf','dfd']
CONDITIONS=['fixed_h1','seen','heldout','permutation','counterfactual','meaningless_candidates','meaningless_questions']
OLD_RUN=LOGS/'run_20260923T034810Z'
OLD_COORDINATOR=2875501
SCRIPT=Path(__file__).resolve()
STARTED_WORKERS={}
ENV=os.environ.copy()
ENV.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')

# Reuse exactly the already-used operational launcher and event functions,
# without evaluating the old module's main, wait loop, or top-level mutations.
selected={'now','write_json','event','metric_paths','report_new_metrics','run_evaluation'}
tree=ast.parse((LOGS/'continue_formal_evaluation.py').read_text())
subset=ast.Module(body=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,ast.FunctionDef) and node.name in selected],type_ignores=[])
exec(compile(subset,str(LOGS/'continue_formal_evaluation.py'),'exec'))

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)

def matching_eval_pids(seed):
    import psutil
    matches=[]
    for process in psutil.process_iter(['pid','cmdline']):
        argv=process.info['cmdline'] or []
        if '--seed' not in argv or '--conditions' not in argv:
            continue
        if not any(arg.endswith('/evaluate_semantic.py') or arg.endswith('scripts/evaluate_semantic.py') for arg in argv):
            continue
        if argv[argv.index('--seed')+1] == str(seed):
            matches.append(process.info['pid'])
    return matches

def launcher_completion(seed):
    path=RUN/f'semantic_controls_seed_{seed}/completion.json'
    return json.loads(path.read_text()) if path.exists() else None

def old_completion():
    path=OLD_RUN/'semantic_controls_seed_42/completion.json'
    return json.loads(path.read_text()) if path.exists() else None

def old_handoff_record():
    rows=[json.loads(line) for line in (OLD_RUN/'events.jsonl').read_text().splitlines() if line]
    failures=[row for row in rows if row.get('event')=='coordinator_failed']
    return failures[-1] if failures else None

def start_worker(seed):
    if matching_eval_pids(seed):
        raise RuntimeError(f'Refusing concurrent evaluation writers for seed {seed}')
    if (OLD_RUN/f'semantic_controls_seed_{seed}/command.json').exists():
        raise RuntimeError(f'Old coordinator already scheduled seed {seed}; investigate before handoff')
    stream=(RUN/f'worker_seed_{seed}.log').open('w')
    previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
    try:
        process=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--worker',str(seed),str(RUN)],
            cwd=REPO,env=ENV,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask))
        STARTED_WORKERS[seed]=process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
        stream.close()
    event('parallel_worker_started',seed=seed,worker_pid=process.pid,max_concurrent_evaluation_launchers=2)
    return process

def run_supervisor():
    profile_path=LOGS/'resource_profile_seed_42_seen_cdf/summary.json'
    profile=json.loads(profile_path.read_text())
    assert profile['seed']==42 and profile['condition']=='seen' and profile['domain']=='cdf'
    peak_pss=profile['sampled_tree_peak_pss_bytes']
    if profile['sampled_min_host_available_bytes'] < 1.6*peak_pss + 32*1024**3:
        raise RuntimeError('Measured host-memory headroom does not satisfy conservative two-run margin')
    if max(row['sampled_peak_memory_used_mib'] for row in profile['gpu_peaks']) > 14*1024:
        raise RuntimeError('Measured GPU-memory headroom does not satisfy conservative two-run margin')
    if old_completion() is not None:
        raise RuntimeError('Seed42 already finished; review old scheduling before introducing another runner')
    cmdline=Path(f'/proc/{OLD_COORDINATOR}/cmdline').read_bytes().decode()
    if 'continue_formal_evaluation.py' not in cmdline:
        raise RuntimeError('Old coordinator PID no longer identifies the expected process')
    if not matching_eval_pids(42):
        raise RuntimeError('Expected active seed42 evaluation is missing')
    event('parallel_handoff_started',max_concurrent_evaluation_launchers=2,
        profile_summary=str(profile_path),old_coordinator_pid=OLD_COORDINATOR,
        note='No evaluator, config, model, threshold, checkpoint selection, batch, or active evaluation process changes')
    # Freeze only the scheduling/monitoring parent until seed7 has a real
    # completed group. Seed42 torchrun is in its own session and runs normally.
    # Its parent keeps the wait handle, so resumption still records its real exit.
    import psutil
    scheduler_identity=psutil.Process(OLD_COORDINATOR).create_time()
    scheduler_held=False
    try:
        scheduler_held=True
        os.kill(OLD_COORDINATOR,signal.SIGSTOP)
        while True:
            parent=psutil.Process(OLD_COORDINATOR)
            if parent.create_time()!=scheduler_identity:
                raise RuntimeError('Old scheduling PID identity changed during hold')
            if parent.status()==psutil.STATUS_STOPPED:
                break
            time.sleep(0.05)
        if old_completion() is not None or (OLD_RUN/'semantic_controls_seed_7/command.json').exists():
            raise RuntimeError('Old schedule advanced during hold; no new seed7 writer started')
        event('old_scheduler_temporarily_held',old_scheduler_pid=OLD_COORDINATOR,
            reason='Guarantee the old overwrite guard is armed before its future seed7 dispatch; active seed42 evaluator continues unchanged')
        workers=STARTED_WORKERS
        start_worker(7)
        finished_workers={}
        handoff=None
        previous=metric_paths()
        while True:
            previous=report_new_metrics(previous)
            first7=RESULTS/'evaluation/seed_7/seen/ffpp_metrics.json'
            if scheduler_held and first7.exists():
                json.loads(first7.read_text())
                os.kill(OLD_COORDINATOR,signal.SIGCONT)
                scheduler_held=False
                event('old_scheduler_resumed',old_scheduler_pid=OLD_COORDINATOR,
                    overwrite_guard_metric=str(first7),note='Its seed42 launcher retains the original parent and real wait exit code')
            old=old_completion()
            if old is not None and old['launcher_exit_code']!=0:
                raise RuntimeError('Actual seed42 semantic torchrun failed; no further runs dispatched')
            for seed,process in list(workers.items()):
                code=process.poll()
                if code is None or seed in finished_workers:
                    continue
                actual=launcher_completion(seed)
                finished_workers[seed]=code
                event('parallel_worker_finished',seed=seed,worker_pid=process.pid,worker_exit_code=code,
                    actual_launcher_exit_code=None if actual is None else actual['launcher_exit_code'])
                if code!=0 or actual is None or actual['launcher_exit_code']!=0:
                    raise RuntimeError(f'Actual seed{seed} parallel evaluation failed; investigate preserved logs')
            actual_active=sum(seed not in finished_workers for seed in workers) + int(old is None)
            if 2048 not in workers and actual_active<2 and not scheduler_held:
                workers[2048]=start_worker(2048)
            if handoff is None:
                handoff=old_handoff_record()
                if handoff is not None:
                    if ('Refusing to overwrite already completed evaluation group' not in handoff.get('error','')
                            or '/evaluation/seed_7/seen/ffpp_metrics.json' not in handoff.get('error','')):
                        raise RuntimeError('Old coordinator failed unexpectedly; investigate instead of treating it as handoff')
                    write_json(RUN/'old_coordinator_handoff.json',dict(original_failure=handoff,
                        classification='planned_schedule_handoff_after_preserved_seed42_completion',
                        coordinator_exit_code=None,expected_exit_code_from_uncaught_runtime_error=1,coordinator_exit_code_evidence='Not a child of this supervisor; actual wait status unavailable. Preserved traceback proves the uncaught RuntimeError path. This is not a torchrun exit code.',
                        note='Original coordinator refused a completed seed7 group before launching a duplicate. Its seed42 actual launcher completion remains authoritative.'))
                    event('old_schedule_handoff_confirmed',old_failure_event=handoff,
                        note='Expected nonzero scheduler refusal; no scientific evaluation failure or masked exit code')
            if old is not None and set(finished_workers)=={7,2048} and handoff is not None:
                break
            time.sleep(5)
        expected={str(RESULTS/f'evaluation/seed_{seed}'/condition/f'{domain}_metrics.json') for seed in (42,7,2048) for condition in CONDITIONS for domain in DOMAINS}
        if metric_paths()!=expected:
            raise RuntimeError('Final inventory is not exactly the required 63 metrics groups')
        completion_paths=[OLD_RUN/f'fixed_h1_seed_{seed}/completion.json' for seed in (42,7,2048)]
        completion_paths += [OLD_RUN/'semantic_controls_seed_42/completion.json',RUN/'semantic_controls_seed_7/completion.json',RUN/'semantic_controls_seed_2048/completion.json']
        actual_launchers=[json.loads(path.read_text()) for path in completion_paths]
        if any(row['launcher_exit_code']!=0 for row in actual_launchers):
            raise RuntimeError('At least one actual evaluation launcher has a nonzero exit code')
        write_json(RUN/'formal_evaluation_completion.json',dict(completed_at_utc=now(),completed_groups=63,
            expected_groups=63,actual_evaluation_launcher_count=6,all_actual_launcher_exit_codes_zero=True,
            actual_launcher_completion_files=[str(path) for path in completion_paths],
            old_scheduler_handoff=str(RUN/'old_coordinator_handoff.json'),matrix={str(seed):{domain:7 for domain in DOMAINS} for seed in (42,7,2048)}))
        event('formal_evaluation_completed',completed_groups=63,expected_groups=63,
            all_launcher_exit_codes_zero=True,completion_file=str(RUN/'formal_evaluation_completion.json'))
    finally:
        if scheduler_held:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            while True:
                first7=RESULTS/'evaluation/seed_7/seen/ffpp_metrics.json'
                worker7=STARTED_WORKERS.get(7)
                worker7_done=worker7 is None or worker7.poll() is not None
                if first7.exists() or (worker7_done and not matching_eval_pids(7)):
                    break
                time.sleep(5)
            try:
                cmdline=Path(f'/proc/{OLD_COORDINATOR}/cmdline').read_bytes().decode()
                if 'continue_formal_evaluation.py' in cmdline and psutil.Process(OLD_COORDINATOR).create_time()==scheduler_identity:
                    os.kill(OLD_COORDINATOR,signal.SIGCONT)
                    event('old_scheduler_resumed_in_finally',old_scheduler_pid=OLD_COORDINATOR,
                        note='Cleanup restored only the scheduling parent; active evaluation processes were never stopped')
            except (FileNotFoundError,ProcessLookupError):
                event('old_scheduler_absent_during_cleanup',old_scheduler_pid=OLD_COORDINATOR)


if len(sys.argv)>1 and sys.argv[1]=='--worker':
    seed=int(sys.argv[2]);RUN=Path(sys.argv[3]);EVENTS=RUN/'events.jsonl'
    launcher_handles=[]
    real_popen=subprocess.Popen
    def tracked_popen(*args,**kwargs):
        previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
        try:
            # These operational observers are single-threaded. Restore the
            # child's original mask before exec; only the parent registration
            # window blocks INT/TERM. Evaluator signal behavior stays unchanged.
            kwargs['preexec_fn']=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
            process=real_popen(*args,**kwargs)
            launcher_handles.append(process)
            return process
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
    def worker_signal(signum,frame):
        raise SystemExit(f'Worker observer received signal {signum}; actual evaluator will finish and its exit code will be saved')
    signal.signal(signal.SIGTERM,worker_signal)
    signal.signal(signal.SIGINT,worker_signal)
    with (LOGS/f'parallel_seed_{seed}.lock').open('a') as seed_lock:
        fcntl.flock(seed_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.Popen=tracked_popen
        try:
            run_evaluation(seed,CONDITIONS[1:],'semantic_controls',metric_paths())
        except BaseException as error:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            if launcher_handles:
                launcher=launcher_handles[-1]
                actual_code=launcher.wait()
                invocation=RUN/f'semantic_controls_seed_{seed}'
                manifest=json.loads((invocation/'command.json').read_text())
                write_json(invocation/'completion.json',dict(**manifest,launcher_pid=launcher.pid,
                    finished_at_utc=now(),launcher_exit_code=actual_code,
                    observer_error=str(error),exit_code_preserved_after_observer_error=True))
            event('parallel_evaluation_worker_failed',seed=seed,error=str(error),traceback=traceback.format_exc())
            raise

else:
    RUN=LOGS/dt.datetime.now(dt.timezone.utc).strftime('parallel_%Y%m%dT%H%M%SZ')
    RUN.mkdir(exist_ok=False);EVENTS=RUN/'events.jsonl'
    def cleanup_signal(signum, frame):
        raise SystemExit(f'Supervisor received signal {signum}; finally restores its scheduling parent')
    signal.signal(signal.SIGTERM,cleanup_signal)
    signal.signal(signal.SIGINT,cleanup_signal)
    try:
        with (LOGS/'parallel_supervisor.lock').open('a') as supervisor_lock:
            fcntl.flock(supervisor_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_supervisor()
    except BaseException as error:
        event('parallel_supervisor_failed',error=str(error),traceback=traceback.format_exc(),
            note='Active evaluators have not been terminated; inspect preserved launcher/worker logs and scheduler state')
        raise
