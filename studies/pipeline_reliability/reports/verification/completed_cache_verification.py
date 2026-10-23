import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,datetime,sys,fcntl
import torch,numpy as np,pandas as pd
torch.set_num_threads(2)
root=Path('/home/z/project/deepfake_detector');sys.path.insert(0,str(root/'src'))
from deepfake_detector.training import run_identity
from deepfake_detector.cache import verify_cache
from deepfake_detector.queueing import verified_run
from deepfake_detector.manifest import digest
cfg=json.loads((root/'configs/v1.1.json').read_text())
dest=root/'reports/completed_run_cache_receipts.json';receipt=json.loads(dest.read_text())
q=json.loads((root/'artifacts/queue_status.json').read_text());new=[]
def safe_proc(pid):
 p=Path('/proc')/str(pid);args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
 script=next((a for a in args if a.endswith('/run.py')),None);i=args.index(script) if script else None
 fields=(p/'stat').read_text().split(') ',1)[1].split()
 return {'PID':pid,'PPID':int(fields[1]),'start_ticks':int(fields[19]),'cwd':str((p/'cwd').resolve()),'script':script,'stage':args[i+1] if i is not None and i+1<len(args) else None}
for rid,r in q['runs'].items():
 if r['state']!='COMPLETE' or rid in receipt['completed_runs']:continue
 backbone,seed_text=rid.rsplit('_seed',1);seed=int(seed_text)
 expected=run_identity(cfg,backbone,seed,root/'artifacts')
 cp=root/'artifacts/runs'/rid/'final.pt';cp_sha=digest(cp)
 state=torch.load(cp,map_location='cpu',weights_only=True)
 assert state['identity']==expected and state['epoch']==15
 weight=state['model']['classifier.weight'].numpy();bias=state['model']['classifier.bias'].numpy()
 dimension={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}[backbone]
 assert weight.shape==(2,dimension) and bias.shape==(2,)
 roles={}
 for role,count in {'train':250000,'ff_test':59935,'cdf_test':368922,'calibration':1120}.items():
  cache=root/'artifacts/features'/rid/role;identity={'checkpoint':cp_sha,'manifest':digest(root/'artifacts/data'/f'{role}.parquet')}
  meta,frame=verify_cache(cache,identity);original=pd.read_parquet(root/'artifacts/data'/f'{role}.parquet')
  pd.testing.assert_frame_equal(frame,original,check_exact=True)
  assert len(frame)==meta['rows']==count and meta['backbone']==backbone and meta['seed']==seed and meta['encoding']=='FP32 raw; no extra L2 normalization'
  assert np.array_equal(np.asarray(meta['head']['weight'],dtype=np.float32),weight) and np.array_equal(np.asarray(meta['head']['bias'],dtype=np.float32),bias)
  zero=0;minimum=float('inf');maximum=0.
  for item in meta['chunks']:
   x=np.load(cache/item['file'],mmap_mode='r',allow_pickle=False);assert x.shape[1]==dimension
   norms=np.linalg.norm(x.astype(np.float64),axis=1);zero+=int((norms==0).sum());minimum=min(minimum,float(norms.min()));maximum=max(maximum,float(norms.max()))
  assert zero==0
  roles[role]={'rows':count,'chunks':len(meta['chunks']),'dimension':dimension,'dtype':'float32','all_chunk_hashes_verified':True,'all_features_finite':True,'zero_vectors':zero,'min_raw_L2_norm':minimum,'max_raw_L2_norm':maximum,'input_manifest_sha256':identity['manifest'],'cached_manifest_sha256':meta['cached_manifest_sha256'],'full_manifest_columns_values_order_match_frozen_input':True,'two_class_head_exactly_matches_final_model':True}
 assert verified_run(cfg,{'run_id':rid,'backbone':backbone,'seed':seed},root/'artifacts')
 latest=json.loads((root/'artifacts/queue_status.json').read_text());events=latest['events']
 i=next(i for i,x in enumerate(events) if x['run_id']==rid and x['state']=='COMPLETE');event=events[i]
 assert event['returncode']==0 and event['physical_device']==r['physical_device']
 starts=[x for x in events[:i] if x['run_id']==rid and x['state']=='RUNNING'];assert len(starts)==1
 oldpid=starts[0]['pid'];assert not Path('/proc/'+str(oldpid)).exists()
 following=[x for x in events[i+1:] if x['state']=='RUNNING' and x['physical_device']==r['physical_device']]
 next_event=following[0] if following else None;next_proc=None
 if next_event and Path('/proc/'+str(next_event['pid'])).exists():
  next_proc=safe_proc(next_event['pid']);assert next_proc['PPID']==3984143 and next_proc['cwd']==str(root) and next_proc['script']==str(root/'run.py') and next_proc['stage']=='worker'
 detail={'run_id':rid,'backbone':backbone,'seed':seed,'verified_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':sum(x['rows'] for x in roles.values()),'dimension':dimension,'roles':roles,'checkpoint_sha256':cp_sha,'source_code_sha256':expected['code_sha256'],'canonical_verified_run':True,'complete_event':event,'old_worker_PID':oldpid,'old_worker_absent':True,'next_event_same_GPU':next_event,'next_worker':next_proc,'original_supervisor_PID':3984143,'duplicate_queue_started':False,'target_performance_reported':False}
 assert detail['rows']==679977
 receipt['completed_runs'][rid]=detail;new.append(rid)
e=json.loads((root/'artifacts/execution_status.json').read_text());assert e['owner_pid']==3984143
owner=safe_proc(3984143);assert owner['cwd']==str(root) and owner['script']==str(root/'run.py') and owner['stage']=='execute'
locks={}
for name in ['execution.lock','queue.lock']:
 with (root/'artifacts'/name).open('rb') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);locks[name]='FREE'
  except BlockingIOError:locks[name]='HELD'
assert locks['execution.lock']=='HELD'
q_latest=json.loads((root/'artifacts/queue_status.json').read_text())
if q_latest['status']=='RUNNING':assert locks['queue.lock']=='HELD'
receipt.update(checked_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_complete_runs=len(receipt['completed_runs']),planned_runs=15,owner=owner,locks=locks,CUDA_initialized=torch.cuda.is_initialized(),status='partial model/cache evidence; final analysis/report pending')
assert not torch.cuda.is_initialized()
tmp=dest.with_name(dest.name+'.tmp');tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,dest)
if new:
 with (root/'reports/audit.log').open('a') as f:f.write(json.dumps({'UTC':receipt['checked_UTC'],'event':'new_full_run_cache_and_rotation_verified','runs':new,'rows_per_run':679977,'code_config_unchanged':True,'new_queue_started':False})+'\n')
print(json.dumps({'UTC':receipt['checked_UTC'],'new_verified_complete_runs':new,'total_verified_complete_runs':len(receipt['completed_runs']),'new_details':[{k:d[k] for k in ['run_id','rows','dimension','complete_event','next_event_same_GPU','next_worker','source_code_sha256']} for rid,d in receipt['completed_runs'].items() if rid in new],'locks':locks,'CUDA_initialized':torch.cuda.is_initialized()},ensure_ascii=False))
