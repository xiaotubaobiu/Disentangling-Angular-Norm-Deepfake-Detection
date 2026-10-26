#!/usr/bin/env python3
"""Read-only external sampling of the first complete seen/CDF evaluation."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import time
import psutil

ROOT = Path('/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2/results')
LOG = ROOT / 'evaluation_logs'
PROFILE = LOG / 'resource_profile_seed_42_seen_cdf'
PROFILE.mkdir(exist_ok=False)
START = ROOT / 'evaluation/seed_42/seen/ffpp_metrics.json'
END = ROOT / 'evaluation/seed_42/seen/cdf_metrics.json'
RUN = LOG / 'run_20260923T034810Z'

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

print(json.dumps({'event':'waiting_for_seed42_seen_cdf','timestamp_utc':stamp()}),flush=True)
while not START.exists():
    status=json.loads((LOG/'current_run.json').read_text())
    if status.get('event')=='coordinator_failed':
        raise RuntimeError('Coordinator failed before resource profiling could start')
    time.sleep(2)
launches=[json.loads(line) for line in (RUN/'events.jsonl').read_text().splitlines() if line]
launches=[row for row in launches if row.get('event')=='evaluation_invocation_started' and row.get('seed')==42 and row.get('phase')=='semantic_controls']
assert len(launches)==1, launches
launcher_pid=launches[0]['launcher_pid']
launcher=psutil.Process(launcher_pid)
previous_cpu={}
previous_at=time.monotonic()
samples=[]
psutil.cpu_percent(None)
while True:
    sampled_at=time.monotonic()
    elapsed=max(sampled_at-previous_at,1e-6)
    try:
        process_list=[launcher,*launcher.children(recursive=True)]
    except psutil.NoSuchProcess:
        process_list=[]
    processes=[]
    cpu_delta=0.0
    current_cpu={}
    for process in process_list:
        try:
            memory=process.memory_full_info()
            cpu=process.cpu_times(); total_cpu=cpu.user+cpu.system
            current_cpu[process.pid]=total_cpu
            if previous_cpu:
                cpu_delta+=max(0,total_cpu-previous_cpu.get(process.pid,0))
            processes.append({'pid':process.pid,'ppid':process.ppid(),'name':process.name(),
                'rss_bytes':memory.rss,'pss_bytes':getattr(memory,'pss',None),
                'uss_bytes':getattr(memory,'uss',None),'cpu_seconds':total_cpu,
                'status':process.status(),'cmdline':process.cmdline()})
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    gpu_raw=subprocess.check_output(['nvidia-smi','--id=0,1,2,3','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in gpu_raw.splitlines():
        index,used,free,util=[int(value.strip()) for value in line.split(',')]
        gpus.append({'index':index,'memory_used_mib':used,'memory_free_mib':free,'utilization_percent':util})
    vm=psutil.virtual_memory()
    sample={'timestamp_utc':stamp(),'elapsed_since_ffpp_metric_seconds':time.time()-START.stat().st_mtime,
        'launcher_pid':launcher_pid,'process_count':len(processes),
        'tree_rss_bytes':sum(row['rss_bytes'] for row in processes),
        'tree_pss_bytes':sum(row['pss_bytes'] or 0 for row in processes),
        'tree_uss_bytes':sum(row['uss_bytes'] or 0 for row in processes),
        'tree_cpu_percent_one_core_100':100*cpu_delta/elapsed if previous_cpu else None,
        'host_cpu_percent':psutil.cpu_percent(None),'host_available_bytes':vm.available,
        'host_used_bytes':vm.used,'gpus':gpus,'processes':processes}
    samples.append(sample)
    with (PROFILE/'samples.jsonl').open('a') as stream:
        stream.write(json.dumps(sample,sort_keys=True)+'\n')
    previous_cpu=current_cpu;previous_at=sampled_at
    if END.exists():
        break
    status=json.loads((LOG/'current_run.json').read_text())
    if status.get('event')=='coordinator_failed':
        raise RuntimeError('Coordinator failed during resource profiling')
    time.sleep(10)
summary={'seed':42,'condition':'seen','domain':'cdf','sample_interval_seconds':10,
    'launcher_pid':launcher_pid,'samples':len(samples),'completed_at_utc':stamp(),
    'duration_seconds_between_adjacent_metric_commits':(END.stat().st_mtime_ns-START.stat().st_mtime_ns)/1e9,
    'duration_definition':'Previous FF++ metrics atomic commit to CDF metrics atomic commit; includes CDF cache validation, loader construction, inference, aggregation and writes, plus domain transition synchronization.',
    'sampled_tree_peak_rss_bytes':max(row['tree_rss_bytes'] for row in samples),
    'sampled_tree_peak_pss_bytes':max(row['tree_pss_bytes'] for row in samples),
    'sampled_tree_peak_uss_bytes':max(row['tree_uss_bytes'] for row in samples),
    'sampled_peak_tree_cpu_percent_one_core_100':max(row['tree_cpu_percent_one_core_100'] or 0 for row in samples),
    'sampled_peak_host_cpu_percent':max(row['host_cpu_percent'] for row in samples),
    'sampled_min_host_available_bytes':min(row['host_available_bytes'] for row in samples),
    'sampled_max_process_count':max(row['process_count'] for row in samples),
    'gpu_peaks':[{ 'index':index,'sampled_peak_memory_used_mib':max(row['gpus'][index]['memory_used_mib'] for row in samples),
        'sampled_peak_utilization_percent':max(row['gpus'][index]['utilization_percent'] for row in samples)} for index in range(4)],
    'caveats':['Sampling may miss between-sample transient peaks.','Summed RSS double-counts shared pages; PSS apportions them.','CPU process churn between samples can omit terminated-worker CPU time.']}
(PROFILE/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,sort_keys=True),flush=True)
