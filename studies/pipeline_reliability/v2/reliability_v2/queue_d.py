"""D development->source recipe freeze->formal queue on idle GPU0-3 only."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
import traceback
import numpy as np
from .trainer import digest,save_json,code_hash,exclusive_file


def validate_device_indices(indices):
    if len(set(indices))!=len(indices) or not indices or not set(indices)<={0,1,2,3}:
        raise ValueError('only distinct approved GPU0-3 are permitted')
    return indices


def choose_recipe(candidates):
    scored=[]
    for candidate in candidates:
        if candidate.get('qualification','PASS')!='PASS':continue
        histories=candidate['histories']
        if not histories or any(len(h)!=candidate['epochs'] for h in histories):raise ValueError('incomplete source-development histories')
        score=float(np.mean([np.mean([r['dev']['macro_auc'] for r in h[-5:]]) for h in histories]))
        scored.append({**candidate,'source_last5_mean':score})
    if not scored:raise ValueError('no qualified recipe')
    maximum=max(c['source_last5_mean'] for c in scored)
    tied=[c for c in scored if maximum-c['source_last5_mean']<=.001+1e-15]
    return min(tied,key=lambda c:(c['epochs'],c['lr']))


def dino_extend(histories,budget):
    if budget>=60:return None
    if budget not in [30,45]:raise ValueError('unregistered DINO budget')
    if any(len(h)!=budget for h in histories):raise ValueError('incomplete DINO histories')
    deltas=[np.mean([r['dev']['macro_auc'] for r in h[-5:]])-np.mean([r['dev']['macro_auc'] for r in h[-10:-5]]) for h in histories]
    return {30:45,45:60}[budget] if max(deltas)>.002 else None


def dev_id(backbone,lr,seed,epochs):return f'D_dev_{backbone}_lr{lr:.0e}_seed{seed}_e{epochs}'


def registered_task(config,backbone,seed,lr,epochs,phase):
    root=Path(config['artifact_root'])
    gate=json.loads((root/'qualification/gpu_gate.json').read_text())
    return {'route':'D','phase':phase,'backbone':backbone,'seed':seed,'lr':lr,'epochs':epochs,
      'microbatch':gate['backbones'][backbone]['microbatch'],'precision':'AMP',
      'train_manifest':'data/D/train.parquet','dev_manifest':'data/D/dev.parquet',
      'qualification_gate':'qualification/D_gate.json',
      'recipe_frozen':phase=='formal','selection':'frozen_last'}


def initial_registry(config):
    rows=[]
    for route,histories in [('D',['released']),('H',['B','AB'])]:
        for backbone in config['backbones']:
            for history in histories:
                for seed in config['formal_seeds']:
                    rows.append({'route':route,'history':history,'backbone':backbone,'seed':seed,
                      'status':'NOT_STARTED','recipe':'RECIPE_UNFROZEN'})
    return {'formal_model_count':len(rows),'formal':rows,'standard_development_count':30,
      'maximum_additional_dino_development_count':12,'legacy_models_eligible':False}


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def gpu_memory():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(i):{'uuid':u,'used_mib':int(m)} for i,u,m in [r.split(', ') for r in result.strip().splitlines()]}


def idle_device(record,occupied):return record['used_mib']<100 and record['uuid'] not in occupied


def reserved(root,uuid):
    try:
        with exclusive_file(Path(root)/'gpu_locks'/f'{uuid}.lock'):return False
    except BlockingIOError:return True


def still_owned(item):
    path=Path('/proc')/str(item['pid'])
    try:
        args=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return any(name in args for name in ['reliability_v2.trainer','reliability_v2.cache_d']) and item['run_id'] in args
    except FileNotFoundError:return False


class Queue:
    def __init__(self,root,code,config_path):
        self.root=Path(root);self.code=Path(code);self.path=Path(config_path)
        self.config=json.loads(self.path.read_text());self.jobs={};self.completed=[]
        validate_device_indices(self.config['allowed_gpus'])
        previous=self.root/'queue_status.json'
        if previous.exists():
            for item in json.loads(previous.read_text()).get('active',[]):
                if still_owned(item):raise RuntimeError('existing owned worker still active; refuse duplicate queue')

    def status(self,stage,detail=None):
        active=[{'pid':p.pid,'run_id':run,'gpu':gpu,'uuid':uuid} for run,(p,gpu,uuid) in self.jobs.items()]
        save_json(self.root/'queue_status.json',{'owner_pid':os.getpid(),'stage':stage,'utc':utc(),
          'active':active,'completed_this_session':self.completed,'detail':detail})
        (self.root/'D_PROGRESS.md').write_text('# D queue progress\n\n'+utc()+'\nStage: '+stage+'\n\n'+json.dumps(active,indent=2)+'\n\nNo protected GPU or service operations exist in this runner.\n')

    def update_config(self):save_json(self.path,self.config)

    def batch(self,run_ids,stage,action='train'):
        if action not in ['train','cache']:raise ValueError('unregistered queue stage')
        pending=list(run_ids)
        while pending or self.jobs:
            for run,(p,gpu,uuid) in list(self.jobs.items()):
                result=p.poll()
                if result is None:continue
                del self.jobs[run]
                if result!=0:
                    self.status('FAILED',{'run_id':run,'returncode':result,'log':str(self.root/'logs'/f'{run}.log')})
                    raise RuntimeError(f'worker failed {run}, inspect log; no result-based automatic omission')
                complete=self.root/'runs'/run/'complete.json' if action=='train' else self.root/'features/D'/run/'complete.json'
                if not complete.exists():raise RuntimeError('child exited without verified completion '+run)
                record=json.loads(complete.read_text())
                if action=='train':
                    if record['status']!='TRAINING_COMPLETE' or digest(self.root/'runs'/run/'final.pt')!=record['final_sha256']:
                        raise RuntimeError('child completion hash mismatch '+run)
                else:
                    from .cache_d import verify_cache
                    if record['status']!='CACHE_COMPLETE':raise RuntimeError('child cache incomplete '+run)
                    for role,receipt in record['roles'].items():
                        role_root=Path(receipt['path']);verify_cache(role_root)
                        if digest(role_root/'complete.json')!=receipt['complete_sha256']:raise RuntimeError('cache role receipt changed')
                self.completed.append(run)
            if pending:
                occupied={gpu for _,gpu,_ in self.jobs.values()}
                memory=gpu_memory()
                query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
                compute={r.split(', ')[0] for r in query.strip().splitlines() if r.strip()}
                free=[i for i in self.config['allowed_gpus'] if i not in occupied and idle_device(memory[i],compute) and not reserved(self.root,memory[i]['uuid'])]
                while pending and free:
                    run=pending.pop(0);gpu=free.pop(0);uuid=memory[gpu]['uuid']
                    if uuid!=self.config['allowed_gpu_uuids'][self.config['allowed_gpus'].index(gpu)]:raise RuntimeError('approved device UUID changed')
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(self.code))
                    with (self.root/'logs'/f'{run}.log').open('a') as stream:
                        command=[str(self.root/'environment/bin/python'),'-u','-m',
                          'reliability_v2.trainer' if action=='train' else 'reliability_v2.cache_d','--config',str(self.path),'--run-id',run]
                        if action=='cache':command+=['--device','cuda']
                        p=subprocess.Popen(command,cwd=self.code,env=env,stdin=subprocess.DEVNULL,
                          stdout=stream,stderr=subprocess.STDOUT)
                    self.jobs[run]=(p,gpu,uuid)
            self.status(stage,{'pending':pending})
            if pending or self.jobs:time.sleep(15)

    def histories(self,ids):return [json.loads((self.root/'runs'/run/'history.json').read_text()) for run in ids]

    def run(self):
        if json.loads((self.root/'qualification/D_gate.json').read_text())['training_code_sha256']!=code_hash():raise RuntimeError('stale source qualification')
        registry_path=self.root/'run_registry.json'
        if not registry_path.exists():save_json(registry_path,initial_registry(self.config))
        self.config.setdefault('runs',{})
        grid={b:([1e-5,3e-5,5e-5] if b=='dinov2_vit_b14' else [1e-5] if b=='clip_vit_b16' else [1e-4]) for b in self.config['backbones']}
        initial=[]
        for b,lrs in grid.items():
            for lr in lrs:
                for seed in self.config['development_seeds']:
                    run=dev_id(b,lr,seed,30);initial.append(run)
                    self.config['runs'][run]=registered_task(self.config,b,seed,lr,30,'development')
        self.update_config();self.batch(initial,'P2_D_STANDARD_DEVELOPMENT')
        recipes={}
        for b,lrs in grid.items():
            candidates=[]
            for lr in lrs:
                ids=[dev_id(b,lr,s,30) for s in self.config['development_seeds']]
                candidates.append({'lr':lr,'epochs':30,'histories':self.histories(ids),'run_ids':ids})
            chosen=choose_recipe(candidates);budget=30;ids=chosen['run_ids']
            if b=='dinov2_vit_b14':
                while True:
                    extended=dino_extend(self.histories(ids),budget)
                    if extended is None:break
                    budget=extended;ids=[]
                    for seed in self.config['development_seeds']:
                        run=dev_id(b,chosen['lr'],seed,budget);ids.append(run)
                        self.config['runs'][run]=registered_task(self.config,b,seed,chosen['lr'],budget,'development')
                    self.update_config();self.batch(ids,f'P2_D_DINO_FRESH_{budget}_DEVELOPMENT')
            recipes[b]={'lr':chosen['lr'],'epochs':budget,'selection_source_only':True,
              'source_30epoch_grid':[{'lr':c['lr'],'last5_mean':float(np.mean([np.mean([r['dev']['macro_auc'] for r in h[-5:]]) for h in c['histories']]))} for c in candidates],
              'final_development_run_ids':ids,'rule':'last5 two-seed mean; <=.001 shorter then smallerLR; DINO adjacent5 windows anyseed>.002, fresh45 then60cap'}
        save_json(self.root/'configs/D_frozen_recipes.json',{'recipes':recipes,'utc':utc(),
          'train_manifest_sha256':digest(self.root/'data/D/train.parquet'),'dev_manifest_sha256':digest(self.root/'data/D/dev.parquet'),
          'code_sha256':code_hash(),'target_scores_used':False})
        formal=[]
        for b,recipe in recipes.items():
            for seed in self.config['formal_seeds']:
                run=f'D_formal_{b}_seed{seed}';formal.append(run)
                self.config['runs'][run]=registered_task(self.config,b,seed,recipe['lr'],recipe['epochs'],'formal')
        self.update_config();registry=json.loads(registry_path.read_text())
        for row in registry['formal']:
            if row['route']=='D':row.update(recipe=recipes[row['backbone']],status='QUEUED')
        save_json(registry_path,registry);self.batch(formal,'P3_D_FORMAL_TRAINING')
        registry=json.loads(registry_path.read_text())
        for row in registry['formal']:
            if row['route']=='D':
                run=f'D_formal_{row["backbone"]}_seed{row["seed"]}'
                receipt=json.loads((self.root/'runs'/run/'complete.json').read_text())
                row.update(status='TRAINING_COMPLETE',final_sha256=receipt['final_sha256'])
        save_json(registry_path,registry)
        if self.config.get('cache_enabled') is True:
            self.batch(formal,'P3_D_FP32_FEATURE_CACHE',action='cache')
            for row in registry['formal']:
                if row['route']=='D':row['status']='CACHE_COMPLETE'
            save_json(registry_path,registry)
            self.status('P3_D_CACHE_COMPLETE_ANALYSIS_PENDING',{'formal_runs':formal,
              'next':'Independent RQ2/RQ3/RQ4 analyses and H source/data acceptance'})
        else:
            self.status('P3_D_TRAINING_COMPLETE_CACHE_PENDING',{'formal_runs':formal,
              'next':'FP32 cache extraction and independent RQ2/RQ3/RQ4 analysis; H remains independent pending source qualification'})


def main(root,code):
    root=Path(root);lock=(root/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=Queue(root,code,root/'configs/execution.json')
    try:queue.run()
    except BaseException:
        queue.status('FAILED',{'traceback':traceback.format_exc(),'active_workers_continue_until_own_completion':True})
        raise
    finally:lock.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True)
    a=p.parse_args();main(a.root,a.code)
