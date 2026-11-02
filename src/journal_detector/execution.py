"""Persistent GPU0-3 queue; data preparation proceeds independently on CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .trainer import save_json, exclusive_file


def process_identity(pid):
    try:
        raw=Path(f'/proc/{int(pid)}/stat').read_text()
        fields=raw[raw.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        return fields[19]  # Linux stat field22: process start ticks.
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def alive(pid,token):
    return bool(token) and process_identity(pid)==str(token)


def eligible_gpus(config,hardware,compute_occupied,reserved):
    if config.get('allowed_gpus')!=[0,1,2,3] or len(config.get('allowed_gpu_uuids',[]))!=4:
        raise ValueError('queue is confined to physical GPU0-3')
    expected=config['allowed_gpu_uuids'];out=[]
    for gpu in hardware:
        if gpu['index'] not in [0,1,2,3]:continue
        if gpu['uuid']!=expected[gpu['index']]:raise ValueError('physical GPU UUID map changed')
        if gpu['uuid'] not in compute_occupied|reserved and gpu['memory']<100:out.append(gpu['uuid'])
    return out


def ready(job):return all(Path(path).is_file() for path in job.get('depends_on_files',[]))


def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    hardware=[dict(index=int(a.strip()),uuid=b.strip(),memory=int(c.strip())) for a,b,c in (r.split(',') for r in rows.splitlines() if r.strip())]
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={row.split(',')[0].strip() for row in rows.splitlines() if row.strip()}
    return hardware,busy


def worker(spec_path):
    spec=json.loads(Path(spec_path).read_text());folder=Path(spec['state_dir']);folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'started.json',{'pid':os.getpid(),'pid_start':process_identity(os.getpid()),'started':time.time(),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'argv':spec['argv']})
    result={'id':spec['id'],'started':time.time(),'argv':spec['argv']}
    try:
        gate=spec.get('code_gate')
        if gate:
            from .trainer import code_hash
            record=json.loads(Path(gate).read_text())
            if record['training_code_sha256']!=code_hash(spec.get('route','D')):
                raise ValueError('training code differs from verified queue gate')
        with (folder/'stdout.log').open('a',buffering=1) as log:
            run=subprocess.run(spec['argv'],cwd=spec['cwd'],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        result.update(exit_code=run.returncode,status='SUCCEEDED' if run.returncode==0 else 'FAILED')
    except Exception as exc:
        result.update(exit_code=-1,status='FAILED',error=str(exc),traceback=traceback.format_exc())
    result['finished']=time.time();save_json(folder/'exit.json',result)
    return result['exit_code']


def dispatch(root,code_root):
    root=Path(root).resolve();code_root=Path(code_root).resolve();operations=root/'operations'
    queue=operations/'jobs.json';state_path=operations/'dispatcher_state.json'
    operations.mkdir(parents=True,exist_ok=True)
    with exclusive_file(operations/'dispatcher.lock'):
        records=json.loads(state_path.read_text()).get('jobs',{}) if state_path.exists() else {}
        save_json(operations/'dispatcher_process.json',{'pid':os.getpid(),'pid_start':process_identity(os.getpid()),'started':time.time()})
        children={}
        while True:
            registry=json.loads(queue.read_text()) if queue.exists() else {'jobs':[]}
            config=json.loads((root/'configs/development.json').read_text())
            for child in list(children):
                if children[child].poll() is not None:children.pop(child)
            for key,record in records.items():
                if record['status']!='RUNNING':continue
                exit_path=Path(record['state_dir'])/'exit.json'
                if exit_path.exists():record.update(json.loads(exit_path.read_text()))
                elif not alive(record['pid'],record['pid_start']):
                    record.update(status='FAILED_UNKNOWN',finished=time.time(),error='worker handle absent without terminal receipt; manual audit before restart')
            reserved={r['gpu_uuid'] for r in records.values() if r['status']=='RUNNING'}
            hardware,busy=inventory();idle=eligible_gpus(config,hardware,busy,reserved)
            for job in sorted(registry['jobs'],key=lambda j:(j.get('priority',10),j['id'])):
                if job['id'] in records or not ready(job) or not idle:continue
                uuid=idle.pop(0);folder=operations/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
                spec={**job,'cwd':str(code_root),'state_dir':str(folder)}
                spec_path=folder/'spec.json';save_json(spec_path,spec)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                    'OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4','PYTHONUNBUFFERED':'1',
                    'PYTHONPATH':str(code_root/'src'),'TOKENIZERS_PARALLELISM':'false'}
                with (folder/'wrapper.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-m','journal_detector.execution','--worker',str(spec_path)],
                        cwd=code_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[job['id']]=child
                records[job['id']]={'id':job['id'],'status':'RUNNING','pid':child.pid,
                    'pid_start':process_identity(child.pid),'gpu_uuid':uuid,'state_dir':str(folder),'launched':time.time()}
                save_json(state_path,{'pid':os.getpid(),'updated':time.time(),'jobs':records})
            pending=[j['id'] for j in registry['jobs'] if j['id'] not in records]
            save_json(state_path,{'pid':os.getpid(),'pid_start':process_identity(os.getpid()),'updated':time.time(),
                'jobs':records,'waiting':pending,'running':sum(r['status']=='RUNNING' for r in records.values())})
            if registry.get('shutdown_when_idle') and not pending and not any(r['status']=='RUNNING' for r in records.values()):return
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path)
    p.add_argument('--code-root',type=Path);p.add_argument('--worker',type=Path)
    a=p.parse_args()
    if a.worker:raise SystemExit(worker(a.worker))
    if not a.root or not a.code_root:p.error('--root and --code-root required')
    dispatch(a.root,a.code_root)
