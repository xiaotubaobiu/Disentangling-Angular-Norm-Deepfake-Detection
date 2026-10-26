#!/usr/bin/env python3
"""Two-slot primary-only evaluation; operational wrapper, no scientific changes."""
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

REPO=Path('/home/z/project/deepfake_detection')
EXP=REPO/'experiments/2026-09-22_semantic_visual_jev_e2'
BASE=EXP/'results/evaluation_logs'
OUTPUT=EXP/'results/evaluation_primary_v2'
SOURCE='0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7'
CONDITIONS=['fixed_h1','seen','heldout','permutation','counterfactual','meaningless_candidates','meaningless_questions']
ENV={'CUDA_VISIBLE_DEVICES':'0,1,2,3','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
sys.path.insert(0,str(REPO/'src'))
from evaluation.e2_provenance import evaluation_source_bundle_sha256

def now(): return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
def write(path,data):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temp.replace(path)

def main():
    lock=(BASE/'primary_v2_supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert evaluation_source_bundle_sha256(REPO)==SOURCE, 'source bundle mismatch'
    assert not OUTPUT.exists(), f'fresh output directory required: {OUTPUT}'
    run=BASE/('primary_v2_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir()
    OUTPUT.mkdir()
    active={}
    pending=[42,7,2048]
    completed={}
    ready=False
    failed=False
    def event(kind,**extra):
        data={'event':kind,'at_utc':now(),'coordinator_pid':os.getpid(),'run_directory':str(run),'evaluation_root':str(OUTPUT),'active':{str(k):v['process'].pid for k,v in active.items()},'pending_seeds':pending,'launcher_exits':completed,**extra}
        with (run/'events.jsonl').open('a') as stream: stream.write(json.dumps(data,sort_keys=True)+'\n')
        write(BASE/'current_run_v2.json',data)
        print(json.dumps(data,sort_keys=True),flush=True)
    write(run/'cohort_manifest.json',{'created_at_utc':now(),'source_bundle_sha256':SOURCE,'evaluation_root':str(OUTPUT),'seeds':[42,7,2048],'domains':['ffpp','cdf'],'conditions':CONDITIONS,'max_simultaneous_launchers':2,'environment':ENV,'configured_per_gpu_batch':128,'code_commit':'103ab5c8fd7eedf5d2ba5c1a0f9fd24ad14c2107'})
    event('primary_v2_started')
    write(BASE/'dispatch_hold.json',{'active':False,'superseded_at_utc':now(),'reason':'Old cohort stopped and preserved; fresh patched primary-only v2 cohort authorized','run_directory':str(run)})
    while active or (pending and not failed):
        while pending and len(active)<2 and not failed:
            seed=pending.pop(0)
            assert evaluation_source_bundle_sha256(REPO)==SOURCE, 'source bundle changed before launch'
            out=OUTPUT/f'seed_{seed}'
            assert not out.exists(), f'seed output already exists: {out}'
            log=run/f'seed_{seed}'
            log.mkdir()
            argv=['/home/z/anaconda3/bin/torchrun','--standalone','--nproc_per_node=4','--log-dir',str(log/'rank_logs'),'--redirects','3',str(EXP/'scripts/evaluate_semantic.py'),'--config',str(EXP/'configs/semantic_jev.yaml'),'--seed',str(seed),'--checkpoint',str(EXP/f'results/training/seed_{seed}/best_model.pth'),'--domains','ffpp','cdf','--conditions',*CONDITIONS,'--feature-cache-root',str(EXP/'results/feature_cache'),'--output-dir',str(out)]
            manifest={'seed':seed,'argv':argv,'shell_display':shlex.join(argv),'cwd':str(REPO),'environment':ENV,'configured_per_gpu_batch':128,'expected_source_bundle_sha256':SOURCE,'started_at_utc':now()}
            write(log/'command.json',manifest)
            console=(log/'launcher_console.log').open('ab',buffering=0)
            process=subprocess.Popen(argv,cwd=REPO,env={**os.environ,**ENV},stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
            manifest['launcher_pid']=process.pid
            write(log/'command.json',manifest)
            active[seed]={'process':process,'console':console,'log':log,'manifest':manifest}
            event('launcher_started',seed=seed,launcher_pid=process.pid)
        for seed,item in list(active.items()):
            if item['process'].poll() is not None:
                code=item['process'].wait()
                write(item['log']/'completion.json',{**item['manifest'],'finished_at_utc':now(),'launcher_exit_code':code})
                item['console'].close()
                completed[str(seed)]=code
                del active[seed]
                failed=failed or code!=0
                event('launcher_completed',seed=seed,launcher_exit_code=code)
        if not ready:
            groups=[]
            missing=[]
            for seed in [42,7,2048]:
                for condition in CONDITIONS:
                    for domain in ['ffpp','cdf']:
                        root=OUTPUT/f'seed_{seed}'/condition
                        paths=[root/f'{domain}_metrics.json']
                        if condition!='permutation': paths += [root/f'{domain}_frames.npz',root/f'{domain}_videos.csv']
                        if condition.startswith('meaningless_'): paths.append(root/f'{domain}_paired_videos.csv')
                        if condition=='counterfactual': paths.append(root/f'{domain}_controls.csv')
                        if all(p.is_file() and p.stat().st_size>0 for p in paths): groups.append({'seed':seed,'condition':condition,'domain':domain,'files':[str(p) for p in paths]})
                        else: missing.append([seed,condition,domain])
            if not missing:
                for seed in [42,7,2048]:
                    provenance=json.loads((OUTPUT/f'seed_{seed}/run_provenance.json').read_text())
                    assert provenance['evaluation_source_bundle_sha256']==SOURCE
                ready=True
                write(BASE/'primary_v2_report_ready.json',{'ready_at_utc':now(),'evaluation_root':str(OUTPUT),'source_bundle_sha256':SOURCE,'groups':groups,'group_count':42,'launcher_exits_at_ready':completed,'active_launchers_at_ready':{str(k):v['process'].pid for k,v in active.items()},'note':'All 42 primary metric groups and associated files exist; launcher completion is tracked separately. No DFD evaluated.'})
                event('primary_v2_artifacts_ready',group_count=42)
        if active: time.sleep(5)
    success=not failed and len(completed)==3 and ready
    write(run/'completion.json',{'finished_at_utc':now(),'launcher_exits':completed,'group_count':len(groups),'pending_seeds':pending,'success':success,'evaluation_root':str(OUTPUT)})
    event('primary_v2_completed' if success else 'primary_v2_failed')
    return 0 if success else 1

if __name__=='__main__':
    try: sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
