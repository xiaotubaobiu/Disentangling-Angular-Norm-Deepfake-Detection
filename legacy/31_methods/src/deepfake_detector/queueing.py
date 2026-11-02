"""Bounded pilot-first queue and explicit dry-run plan."""
from pathlib import Path
import datetime,json,os,subprocess,time,signal
from .manifest import save_json,digest
from .orchestration import validate_devices,require_gpu_gates,exclusive_lock,llm_session
from .stages import stage_identity,config_identity


def dry_plan(config,project):
    project=Path(project);artifact=Path(config['artifact_root']);devices=validate_devices(config['physical_gpus'])
    expected=['clip_vit_b16','dinov2_vit_b14','efficientnet_b0']
    if set(config['backbones'])!=set(expected) or config['seeds']!=[7,42,123,2026,3407] or config['epochs']!=15 or config['effective_batch']!=128 or config['checkpoint_selection']!='final_epoch_only':
        raise ValueError('Configuration differs from the fixed v1.1 run protocol')
    jobs=[{'run_id':f'{b}_seed{s}','backbone':b,'seed':s,'epochs':15,'effective_batch':128,
           'learning_rate':config['backbones'][b]['learning_rate'],'checkpoint_selection':'final_epoch_only'} for s in config['seeds'] for b in expected]
    missing=[]
    for path,name in [('data/freeze_complete.json','scientific_role_freeze'),('cpu_preflight_complete.json','CPU_preflight'),('probe_complete.json','GPU_microbatch_probe'),('pilot_complete.json','end_to_end_pilot')]:
        if not (artifact/path).is_file():missing.append(name)
    return {'status':'DRY_PLAN_ONLY','config_identity':config_identity(config),'jobs':jobs,'pilot_run_id':jobs[0]['run_id'],
            'physical_devices':devices,'maximum_concurrent_workers':len(devices),'blocking_gates':missing,
            'GPU_initialized':False,'LLM_stopped':False,'data_roles_changed':False,'matched_heads':30,
            'calibration_rules':{'FPR5':'independent source real only; fake calibration is not required',
                                 'TPR95':'unavailable without independent complete fake method coverage'}}


def schedule_jobs(jobs,devices,launch,is_complete,journal,identity,poll_seconds=1.,pilot_verified=False):
    devices=validate_devices(devices);journal=Path(journal)
    if not jobs or len({j['run_id'] for j in jobs})!=len(jobs):raise ValueError('Unique nonempty jobs required')
    if journal.exists():
        old=json.loads(journal.read_text())
        if old.get('identity')!=identity:raise ValueError('Queue identity mismatch; no mixed protocol resume')
        for run in old.get('runs',{}).values():
            if run.get('state')=='RUNNING' and Path(f'/proc/{run["pid"]}/cmdline').exists():
                if Path(f'/proc/{run["pid"]}/cmdline').read_bytes():raise ValueError('Existing queue worker still alive; no duplicate launch')
    record={'status':'RUNNING','identity':identity,'runs':{},'events':[],'peak_workers':0,'owner_pid':os.getpid()}
    active={};pending=list(jobs);pilot=jobs[0]['run_id'];pilot_passed=pilot_verified
    def event(job,state,**details):
        rid=job['run_id'];record['runs'][rid]={'state':state,**details}
        record['events'].append({'run_id':rid,'state':state,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**details})
        save_json(journal,record)
    try:
        while pending or active:
            for device,(job,process) in list(active.items()):
                code=process.poll()
                if code is None:continue
                del active[device]
                if code!=0 or not is_complete(job):
                    event(job,'FAILED',physical_device=device,returncode=code)
                    raise RuntimeError(f'{job["run_id"]}: worker failed or lacks verified completion')
                event(job,'COMPLETE',physical_device=device,returncode=code)
                if job['run_id']==pilot:pilot_passed=True
            for device in devices:
                if device in active or not pending:continue
                if not pilot_passed and (active or pending[0]['run_id']!=pilot):break
                job=pending.pop(0)
                if is_complete(job):
                    event(job,'REUSED_COMPLETE')
                    if job['run_id']==pilot:pilot_passed=True
                    continue
                process=launch(job,device);active[device]=(job,process)
                record['peak_workers']=max(record['peak_workers'],len(active))
                event(job,'RUNNING',physical_device=device,pid=process.pid)
                if not pilot_passed:break
            if active:time.sleep(poll_seconds)
        record['status']='COMPLETE';save_json(journal,record);return record
    except BaseException as error:
        record['status']='FAILED';record['failure']=f'{type(error).__name__}: {error}';save_json(journal,record)
        raise
    finally:
        # These Popen handles refer only to this scheduler's direct children.
        for device,(job,process) in active.items():
            if process.poll() is None:process.terminate()
        for device,(job,process) in active.items():
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=15)
            event(job,'CANCELLED',physical_device=device,returncode=process.returncode)


def verified_run(config,job,artifact):
    from .training import run_identity
    from .cache import verify_cache
    root=Path(artifact);folder=root/'runs'/job['run_id'];marker=folder/'complete.json'
    if not marker.exists():return False
    expected=run_identity(config,job['backbone'],job['seed'],root);meta=json.loads(marker.read_text())
    checkpoint=folder/'final.pt'
    if meta.get('status')!='TRAINING_COMPLETE' or meta['identity']!=expected or digest(checkpoint)!=meta['final_sha256']:raise ValueError('Run completion identity/hash mismatch')
    frozen=json.loads((root/'data/freeze_complete.json').read_text())
    for role in ['train','ff_test','cdf_test']+(['calibration'] if frozen['calibration_available'] else []):
        cache=root/'features'/job['run_id']/role
        if not (cache/'complete.json').exists():return False
        verify_cache(cache,{'checkpoint':digest(checkpoint),'manifest':digest(root/'data'/f'{role}.parquet')})
    return True


def run_queue(config,project,config_path):
    project=Path(project);root=Path(config['artifact_root']);plan=dry_plan(config,project)
    require_gpu_gates(root)
    if plan['blocking_gates']:raise ValueError('Missing frozen preflight/probe/pilot gates')
    pilot=json.loads((root/'pilot_complete.json').read_text())
    if pilot.get('status')!='END_TO_END_PILOT_PASS' or pilot['code_sha256']!=stage_identity('code',config,project,{})['code_sha256'] or pilot['freeze_sha256']!=digest(root/'data/freeze_complete.json') or pilot['probe_sha256']!=digest(root/'probe_complete.json'):raise ValueError('Pilot binding mismatch')
    # Revalidate all model probe bindings before any container action.
    from .training import run_identity
    for job in plan['jobs']:run_identity(config,job['backbone'],job['seed'],root)
    inputs={'freeze':root/'data/freeze_complete.json','probe':root/'probe_complete.json','prepared':root/'prepare_complete.json'}
    identity=stage_identity('formal_queue',config,project,inputs);logs=[]
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Queue interrupted by signal {signum}')
    previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        with exclusive_lock(root/'queue.lock'):
            if all(verified_run(config,j,root) for j in plan['jobs']):return {'status':'ALL_RUNS_ALREADY_VERIFIED','LLM_stopped':False}
            with llm_session(root,plan['physical_devices']) as session:
                def launch(job,device):
                    path=root/'runs'/job['run_id'];path.mkdir(parents=True,exist_ok=True)
                    log=(path/'worker.log').open('a');logs.append(log)
                    env=os.environ.copy();uid=session['device_uuids'][str(device)]
                    env.update(CUDA_VISIBLE_DEVICES=uid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                    args=[str(project/'.venv/bin/python'),'-B',str(project/'run.py'),'worker','--config',str(Path(config_path).resolve()),
                          '--backbone',job['backbone'],'--seed',str(job['seed']),'--device-uuid',uid,'--session-id',session['session_id']]
                    return subprocess.Popen(args,cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                return schedule_jobs(plan['jobs'],plan['physical_devices'],launch,lambda j:verified_run(config,j,root),root/'queue_status.json',identity,pilot_verified=True)
    finally:
        for log in logs:log.close()
        signal.signal(signal.SIGTERM,previous)


def worker(config,backbone,seed,device_uuid,session_id):
    root=Path(config['artifact_root']);require_gpu_gates(root)
    session=json.loads((root/'service_state.json').read_text())
    if session['session_id']!=session_id or session['state'] not in ['STOPPED','ORIGINALLY_STOPPED'] or device_uuid not in session['device_uuids'].values():raise ValueError('Worker is not bound to this authorized service/device session')
    if not Path(f'/proc/{session["owner_pid"]}/cmdline').exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=device_uuid:raise ValueError('Worker parent/device binding invalid')
    from .qualification import validate_session
    validate_session(config,device_uuid,session_id)
    from .orchestration import gpu_inventory
    cards,apps=gpu_inventory()
    if device_uuid not in {cards[i]['uuid'] for i in validate_devices(config['physical_gpus'])}:raise ValueError('Physical GPU binding changed')
    import torch
    if torch.cuda.device_count()!=1:raise ValueError('Worker must expose exactly one authorized GPU')
    from .training import train,extract
    train(config,backbone,seed,'cuda:0',root);extract(config,backbone,seed,'cuda:0',root)
