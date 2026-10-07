"""Locks and physical-device/service lifecycle gates. Never touch Jev/gateway."""
from contextlib import contextmanager
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,uuid,signal
from .manifest import digest,save_json


@contextmanager
def termination_guard():
    previous=signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:yield
    finally:signal.signal(signal.SIGTERM,previous)

def owned_process(args,**kwargs):
    child=subprocess.Popen(args,**kwargs)
    try:child.wait();return child
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)

def validate_devices(devices):
    devices=list(devices)
    if not devices or len(set(devices))!=len(devices) or not set(devices)<=set([4,5,6,7]):raise ValueError('Only unique physical devices4–7 are authorized')
    return devices


@contextmanager
def exclusive_lock(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def code_hash(project):
    h=hashlib.sha256()
    for p in sorted((Path(project)/'src').rglob('*.py')):h.update(str(p.relative_to(project)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def validate_frozen_manifests(artifact):
    root=Path(artifact);data=root/'data';frozen=json.loads((data/'freeze_complete.json').read_text())
    if frozen.get('status')!='FROZEN':raise ValueError('Data policy is not frozen')
    mapping=frozen.get('manifest_sha256',{})
    if not {'train.parquet','ff_test.parquet','cdf_test.parquet'}<=set(mapping):raise ValueError('Missing mandatory frozen manifest')
    if not isinstance(frozen.get('calibration_available'),bool):raise ValueError('Calibration availability must be explicitly frozen')
    if frozen['calibration_available'] and 'calibration.parquet' not in mapping:raise ValueError('Missing mandatory independent calibration manifest')
    if not frozen['calibration_available'] and 'calibration.parquet' in mapping:raise ValueError('Unavailable calibration cannot have an active manifest')
    for name,sha in mapping.items():
        if Path(name).name!=name or not name.endswith('.parquet'):raise ValueError('Unsafe frozen manifest path')
        p=data/name
        if not p.is_file() or digest(p)!=sha:raise ValueError('Frozen manifest changed or missing')
    return frozen


def require_gpu_gates(artifact):
    root=Path(artifact)
    names=['data/freeze_complete.json','prepare_complete.json','cpu_preflight_complete.json']
    if not all((root/name).is_file() for name in names):raise ValueError('Missing data/public-initialization/CPU gates: do not stop LLM')
    frozen=validate_frozen_manifests(root);prepared=json.loads((root/names[1]).read_text());cpu=json.loads((root/names[2]).read_text())
    if frozen['status']!='FROZEN' or prepared['status']!='PUBLIC_ASSETS_READY' or cpu['status']!='CPU_PREFLIGHT_PASS':raise ValueError('Prerequisite state is not valid')
    for file,sha in frozen['manifest_sha256'].items():
        if digest(root/'data'/file)!=sha:raise ValueError('Frozen manifest changed')
    for name,v in prepared['weights'].items():
        if digest(v['path'])!=v['sha256']:raise ValueError('Public initialization changed')
    if cpu['freeze_sha256']!=digest(root/names[0]):raise ValueError('CPU tests predate frozen data')
    project=Path(cpu['project'])
    if cpu['config_sha256']!=digest(project/'configs/v1.1.json') or cpu['code_sha256']!=code_hash(project):raise ValueError('CPU tests predate current code/config')
    return frozen


def gpu_inventory():
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],check=True,capture_output=True,text=True)
    cards={}
    for line in q.stdout.splitlines():
        index,uuid,memory,util=[x.strip() for x in line.split(',')];cards[int(index)]={'uuid':uuid,'memory_MiB':int(memory.split()[0]),'utilization':int(util.split()[0])}
    q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],check=True,capture_output=True,text=True)
    apps=[]
    for line in q.stdout.splitlines():
        if not line.strip():continue
        uuid,pid,name,memory=[x.strip() for x in line.split(',')];apps.append({'uuid':uuid,'pid':int(pid),'name':name,'memory':memory})
    return cards,apps


def container_state():
    template='{"id":{{json .Id}},"state":{"Running":{{json .State.Running}},"Pid":{{json .State.Pid}},"Status":{{json .State.Status}}},"restart":{{json .HostConfig.RestartPolicy}},"runtime":{{json .HostConfig.Runtime}}}'
    q=subprocess.run(['docker','inspect','--format',template,'onecat-vllm'],check=True,capture_output=True,text=True)
    return json.loads(q.stdout)


def pause_llm(artifact,devices):
    root=Path(artifact);require_gpu_gates(root);devices=validate_devices(devices)
    journal=root/'service_state.json'
    if journal.exists():
        prior=json.loads(journal.read_text())
        old_running=prior.get('original',{}).get('state',{}).get('Running',False)
        if old_running and (prior.get('state') in ['PREPARED','STOPPED'] or (prior.get('state')=='RESTORED_CONTAINER_RUNNING_HEALTH_PENDING' and not container_state()['state']['Running'])):
            raise ValueError('Unresolved original service restore intent; verify owned workers stopped then run restore before starting a new session')
    original=container_state();cards,apps=gpu_inventory();uuids={cards[i]['uuid'] for i in devices}
    for app in apps:
        group=(Path('/proc')/str(app['pid'])/'cgroup').read_text()
        if app['uuid'] in uuids:
            if original['id'] not in group:raise ValueError(f"Unrelated process on target card: {app['pid']}")
        elif original['id'] in group:
            raise ValueError('LLM container occupies an unauthorized card; refusing container-wide stop')
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'container':'onecat-vllm','original':original,
      'physical_devices':devices,'device_uuids':{str(i):cards[i]['uuid'] for i in devices},'owner_pid':os.getpid(),'session_id':uuid.uuid4().hex,'state':'PREPARED',
      'stop_command':'docker stop --time 60 onecat-vllm','restore_command':'docker start onecat-vllm','credentials_recorded':False,'gateway_changed':False}
    save_json(root/'service_state.json',record)
    if original['state']['Running']:
        subprocess.run(['docker','stop','--time','60','onecat-vllm'],check=True,capture_output=True,text=True)
    current=container_state()
    if current['id']!=original['id'] or current['state']['Running']:raise ValueError('Container identity/state changed during pause')
    for attempt in range(30):
        cards,apps=gpu_inventory()
        if not any(v['uuid'] in uuids for v in apps) and all(cards[i]['memory_MiB']<128 and cards[i]['utilization']==0 for i in devices):break
        time.sleep(2)
    else:raise ValueError('Target GPUs did not become idle; no probe/training launch')
    record['state']='STOPPED' if original['state']['Running'] else 'ORIGINALLY_STOPPED';save_json(root/'service_state.json',record)
    return record


def restore_llm(artifact,record):
    root=Path(artifact);current=container_state()
    if current['id']!=record['original']['id']:raise ValueError('Different container: refusing to start unrelated service')
    cards,apps=gpu_inventory();uuids=set(record['device_uuids'].values())
    if any(a['uuid'] in uuids for a in apps):
        if current['state']['Running']:
            for app in apps:
                if app['uuid'] in uuids and current['id'] not in (Path('/proc')/str(app['pid'])/'cgroup').read_text():
                    raise ValueError('Running LLM shares target cards with another process; restoration is not verified')
            record['state']='RESTORED_ALREADY_RUNNING';save_json(root/'service_state.json',record);return
        raise ValueError('Project/other GPU processes still present; do not collide with LLM')
    if record['original']['state']['Running']:
        subprocess.run(['docker','start','onecat-vllm'],check=True,capture_output=True,text=True)
        after=container_state()
        if not after['state']['Running']:raise ValueError('LLM restart failed')
        record['state']='RESTORED_CONTAINER_RUNNING_HEALTH_PENDING'
    else:record['state']='LEFT_ORIGINALLY_STOPPED'
    save_json(root/'service_state.json',record)


@contextmanager
def llm_session(artifact,devices):
    """Restore even when pause fails after stopping; ignore unowned old journals.

    Callers must join their own workers before this context exits. If those
    workers remain, restore_llm refuses to overlap them and reports the blocker.
    """
    root=Path(artifact);journal=root/'service_state.json'
    previous=journal.read_bytes() if journal.exists() else None
    record=None
    try:
        record=pause_llm(root,devices)
        yield record
    finally:
        if record is None and journal.exists() and journal.read_bytes()!=previous:
            candidate=json.loads(journal.read_text())
            if candidate.get('owner_pid')==os.getpid():record=candidate
        if record is not None:restore_llm(root,record)


def verify_selected_source_stats(artifact):
    import pandas as pd
    root=Path(artifact);frozen=validate_frozen_manifests(root);checked=0
    for name in frozen['manifest_sha256']:
        df=pd.read_parquet(root/'data'/name).drop_duplicates('image_path')
        for row in df.itertuples():
            stat=Path(row.image_path).stat()
            if stat.st_size!=row.bytes or stat.st_mtime_ns!=row.mtime_ns:
                raise ValueError('Source image changed since byte/RGB audit: '+row.image_path)
            checked+=1
    return {'selected_paths_stat_checked':checked,'stat_check_is_not_new_byte_hash_proof':True}

def preflight(config,project):
    project=Path(project);artifact=Path(config['artifact_root']);frozen=artifact/'data/freeze_complete.json'
    if not frozen.exists():raise ValueError('Cannot pass full CPU preflight before data/policy freeze')
    source_checks=verify_selected_source_stats(artifact)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(project/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([str(project/'.venv/bin/python'),'-B','-m','pytest','-q'],cwd=project,env=env,capture_output=True,text=True)
    (project/'reports/cpu_preflight_tests.txt').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('CPU suite failed; see reports/cpu_preflight_tests.txt')
    record={'status':'CPU_PREFLIGHT_PASS','project':str(project),'config_sha256':digest(project/'configs/v1.1.json'),
      'code_sha256':code_hash(project),'freeze_sha256':digest(frozen),'test_output':p.stdout,'CUDA_visible':'','LLM_stopped':False,'source_checks':source_checks}
    save_json(artifact/'cpu_preflight_complete.json',record);return record
