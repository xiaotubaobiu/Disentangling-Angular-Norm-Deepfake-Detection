"""One-shot read-only runtime snapshot. Does not launch, stop or reconfigure jobs."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib
import json
import shutil
import subprocess
import time

R = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V = Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cpu():
    return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]


def main():
    started=utc(); c0=cpu(); samples=[]
    for n in range(12):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
        rows=[]
        for line in raw.splitlines():
            i,u,g,m,t=[x.strip() for x in line.split(',')]
            rows.append(dict(index=int(i),uuid=u,gpu_util_percent=int(g),memory_used_MiB=int(m),memory_total_MiB=int(t)))
        samples.append(dict(utc=utc(),gpus=rows))
        if n<11:time.sleep(2)
    c1=cpu();delta=[b-a for a,b in zip(c0,c1)];total=sum(delta)
    compute=[]
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        u,p,m=[x.strip() for x in line.split(',')]
        cmd=Path('/proc')/p/'cmdline'
        compute.append(dict(uuid=u,pid=int(p),used_MiB=int(m),cmdline=cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else 'EXITED_DURING_SNAPSHOT'))
    gpu_summary=[]
    for i in range(8):
        xs=[s['gpus'][i]['gpu_util_percent'] for s in samples]
        gpu_summary.append(dict(index=i,mean_util_percent=sum(xs)/len(xs),min_util_percent=min(xs),max_util_percent=max(xs),samples=len(xs),memory_used_MiB=samples[-1]['gpus'][i]['memory_used_MiB']))
    queue=json.loads((R/'queue_status.json').read_text());config=json.loads((R/'configs/execution.json').read_text())
    dev={k:v for k,v in config['runs'].items() if v['phase']=='development'}
    completed=[k for k in dev if (R/'runs'/k/'complete.json').is_file()]
    active=[x['run_id'] for x in queue['active']]
    pending=[k for k in dev if k not in completed and k not in active]
    progress=json.loads((R/'data/H/H_preprocessing_progress.json').read_text())
    coverage={};duplicates=0;partial=0
    for line in (R/'data/H/H_preprocessed_videos.jsonl').open():
        try:e=json.loads(line)
        except json.JSONDecodeError:partial+=1;continue
        duplicates+=int(e['video_uid'] in coverage);coverage[e['video_uid']]=e
    reasons={k:Counter() for k in ['A','B','AB']};groups=defaultdict(Counter)
    for e in coverage.values():
        m=json.loads(Path(e['manifest']).read_text());video=m['video']
        group='/'.join([video['dataset'],video['domain'],video['method']]);groups[group]['processed']+=1
        for version in ['A','B','AB']:
            if m['eligible'][version]:groups[group][version]+=1
            else:
                failures=m.get('version_failures',{}).get(version,[])
                clean={str(f).partition(':')[2] if str(f).partition(':')[0].isdigit() else str(f) for f in failures}
                if m.get('video_failure'):clean.add(str(m['video_failure']))
                if not clean:clean.add('missing_failure_reason_in_manifest')
                reasons[version].update(clean)
        groups[group]['common']+=int(m['common_B_AB'])
    partial_report=dict(utc=utc(),scope='PARTIAL_PREPROCESSING_COVERAGE_ONLY_NOT_FINAL_H_COHORTS',unique_videos=len(coverage),duplicate_lines=duplicates,trailing_partial_lines_ignored=partial,
        version_ineligible_videos={v:sum(not e['eligible'][v] for e in coverage.values()) for v in reasons},ineligible_video_reason_counts={v:dict(x) for v,x in reasons.items()},processed_groups={k:dict(x) for k,x in groups.items()},all_results_preserved=True,thresholds_and_geometry_changed=False)
    save(R/'reports/H_partial_coverage_1315.json',partial_report)
    apparent={}
    for name,path in [('artifact_root',R),('H_derived',Path(progress['image_root']))]:
        apparent[name]=int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0])
    cap=dict(working_set_estimate_GiB=200,reserve_GiB=50,workspace_free_GiB=shutil.disk_usage(R).free/1024**3,Dataset_free_GiB=shutil.disk_usage('/Dataset').free/1024**3,apparent_bytes=apparent)
    receipt=dict(utc=utc(),started_utc=started,gpu_samples=samples,gpu_summary=gpu_summary,CPU_busy_percent=100*(total-delta[3]-delta[4])/total,CPU_iowait_percent=100*delta[4]/total,
        compute_processes=compute,D_queue=queue,D_registered_development_total=len(dev),D_completed_development_run_ids=completed,D_active_run_ids=active,D_pending_development_run_ids=pending,H_progress=progress,
        capacity=cap,config_sha256=digest(R/'configs/execution.json'),D_gate_sha256=digest(R/'qualification/D_gate.json'),training_file_sha256={n:digest(V/'reliability_v2'/n) for n in ['trainer.py','models.py','geometry_h.py','cache_d.py','scoring_d.py']},
        H_freeze_present=(R/'data/H/freeze.json').exists(),H_gate_present=(R/'qualification/H_gate.json').exists(),experiment_process_restarted=False,frozen_config_modified=False,automation_retried=False)
    save(R/'reports/runtime_health_1315.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['gpu_samples','compute_processes','D_queue']},ensure_ascii=False))


if __name__=='__main__':main()
