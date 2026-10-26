#!/usr/bin/env python3
"""Publish primary-domain readiness without controlling existing evaluation jobs."""
import datetime as dt
import json
import os
from pathlib import Path
import time
import psutil

EXP=Path('/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2')
RESULTS=EXP/'results'
LOGS=RESULTS/'evaluation_logs'
CONDITIONS=('fixed_h1','seen','heldout','permutation','counterfactual','meaningless_candidates','meaningless_questions')
READY=LOGS/'primary_report_ready.json'
EVENTS=LOGS/'primary_readiness_events.jsonl'
EVENTS.touch(exist_ok=True)
primary=[(seed,condition,domain,RESULTS/f'evaluation/seed_{seed}'/condition/f'{domain}_metrics.json') for seed in (42,7,2048) for condition in CONDITIONS for domain in ('ffpp','cdf')]
dfd_permutation=[RESULTS/f'evaluation/seed_{seed}/permutation/dfd_metrics.json' for seed in (42,7,2048)]
while not READY.exists():
    if not all(path.is_file() for *_,path in primary) or not all(path.is_file() for path in dfd_permutation):
        time.sleep(5);continue
    inventory=[]
    for seed,condition,domain,metric in primary:
        records=json.loads(metric.read_text())
        if not isinstance(records,list) or not records:
            raise RuntimeError(f'Invalid primary metrics: {metric}')
        metric_time=metric.stat().st_mtime_ns
        artifacts=[metric]
        if condition!='permutation':
            artifacts.extend([metric.parent/f'{domain}_frames.npz',metric.parent/f'{domain}_videos.csv'])
        if condition.startswith('meaningless_'):
            artifacts.append(metric.parent/f'{domain}_paired_videos.csv')
        elif condition=='counterfactual':
            artifacts.append(metric.parent/f'{domain}_controls.csv')
        for path in artifacts:
            stat=path.stat()
            if stat.st_size<=0 or stat.st_mtime_ns>metric_time:
                raise RuntimeError(f'Primary artifact is empty or changed after its metrics commit: {path}')
            inventory.append(dict(path=str(path),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    for path in dfd_permutation:
        rows=json.loads(path.read_text())
        if not isinstance(rows,list) or not rows:
            raise RuntimeError(f'Invalid DFD permutation audit: {path}')
    launchers=[]
    evaluator=str(EXP/'scripts/evaluate_semantic.py')
    for process in psutil.process_iter(['pid','cmdline']):
        argv=process.info['cmdline'] or []
        if evaluator in argv and '--seed' in argv and any(arg.endswith('/torchrun') for arg in argv):
            launchers.append(dict(pid=process.pid,seed=int(argv[argv.index('--seed')+1])))
    completion_files=[]
    for root in (LOGS/'run_20260923T034810Z',LOGS/'parallel_20260923T043530Z'):
        for path in sorted(root.glob('*/completion.json')):
            value=json.loads(path.read_text())
            completion_files.append(dict(path=str(path),seed=value.get('seed'),phase=value.get('phase'),launcher_exit_code=value.get('launcher_exit_code')))
    ready=dict(event='primary_report_ready',timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        primary_completed_groups=42,primary_expected_groups=42,primary_domains=['ffpp','cdf'],
        primary_artifact_count=len(inventory),primary_artifacts=inventory,
        dfd_permutation_audits=[str(path) for path in dfd_permutation],
        total_metrics_current=len(list((RESULTS/'evaluation').glob('seed_*/*/*_metrics.json'))),
        actual_launcher_completion_evidence=completion_files,launchers_still_active=launchers,
        caveat='Primary metrics are committed after their NPZ/CSV outputs. Remaining DFD performance work and actual launcher exits continue to be tracked separately; readiness does not assert that every launcher has exited.')
    temporary=READY.with_suffix('.json.tmp');temporary.write_text(json.dumps(ready,indent=2,sort_keys=True)+'\n');temporary.replace(READY)
    event={key:value for key,value in ready.items() if key!='primary_artifacts'}
    event['readiness_file']=str(READY)
    with EVENTS.open('a') as stream:stream.write(json.dumps(event,sort_keys=True)+'\n')
    print(json.dumps(event,sort_keys=True),flush=True)
