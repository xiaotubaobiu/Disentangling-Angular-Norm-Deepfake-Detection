import os,json,hashlib,pathlib,fcntl,time,datetime,traceback
import numpy as np,torch
from reliability_v2 import cache_d,models,trainer
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
C=pathlib.Path('/Dataset/deepfake_detector_v2_cold_20261023/staged_copies')
REPORT=R/'reports/cold_canonical_native_pilot_20261024T1845.json'
JOURNAL=R/'reports/cold_canonical_native_pilot_journal_20261024T1845.json'
RUN='D_formal_efficientnet_b0_seed3407'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 q=p.with_name(p.name+'.tmp')
 with q.open('w') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
started=time.monotonic();doc={'utc':utc(),'status':'PENDING','run_id':RUN,'SSD_released_bytes':0,'canonical_path_restored':False,'evidence_retained':True}
lock=(R/'.cold_auxiliary_migration.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not REPORT.exists() and not JOURNAL.exists(),'existing pilot; inspect first'
assert not (R/'reports'/f'{RUN}_analysis_launch.json').exists(),'pilot run is in active analysis queue'
base,config,task=cache_d._context(R/'configs/execution.json',RUN)
assert json.loads((R/'runs'/RUN/'complete.json').read_text())['status']=='TRAINING_COMPLETE'
entries=json.loads((R/'reports/D_closed_cold_copy_manifest_20261024T1510.json').read_text())['entries']
selected=[next(e for e in entries if e['relative']==f'runs/{RUN}/{name}') for name in ('state.pt','epoch_01_methods.npy')]
records=[]
for e in selected:
 s=R/e['relative'];t=pathlib.Path(e['staged_path']); st=s.lstat(); backup=s.with_name('.'+s.name+'.coldpilot.original.20261024T1845');alias=s.with_name('.'+s.name+'.coldpilot.alias.20261024T1845')
 assert s.parent.resolve()==R/'runs'/RUN and t.resolve().is_relative_to(C.resolve())
 assert s.is_file() and not s.is_symlink() and st.st_nlink==1
 assert (st.st_ino,st.st_size,st.st_mtime_ns)==(e['source_inode'],e['bytes'],e['mtime_ns'])
 assert t.is_file() and not t.is_symlink() and t.stat().st_size==e['bytes']
 assert sha(s)==sha(t)==e['sha256']
 assert not backup.exists() and not alias.exists() and not backup.is_symlink() and not alias.is_symlink()
 records.append(dict(e,source=str(s),backup=str(backup),alias=str(alias),original_inode=st.st_ino,original_nlink=st.st_nlink,cold_mtime_ns=t.stat().st_mtime_ns))
keep={str(p):sha(p) for p in [R/'runs'/RUN/'final.pt',R/'runs'/RUN/'source_best.pt']}
doc['selected']=records;doc['native_checks']=[]
def restore():
 for e in reversed(records):
  s=pathlib.Path(e['source']);b=pathlib.Path(e['backup']);a=pathlib.Path(e['alias'])
  if b.exists():
   assert b.stat().st_ino==e['original_inode'] and sha(b)==e['sha256']
   if s.is_symlink():
    assert s.resolve()==pathlib.Path(e['staged_path']).resolve();os.replace(b,s)
   else:
    assert s.stat().st_ino==b.stat().st_ino;b.unlink()
  if a.is_symlink():a.unlink()
 for e in records:
  s=pathlib.Path(e['source']);st=s.lstat()
  assert not s.is_symlink() and (st.st_ino,st.st_size,st.st_mtime_ns,st.st_nlink)==(e['original_inode'],e['bytes'],e['mtime_ns'],e['original_nlink']) and sha(s)==e['sha256']
 save(JOURNAL,{'utc':utc(),'phase':'ORIGINAL_CANONICAL_REGULAR_FILES_RESTORED','records':records,'SSD_released_bytes':0})
def install():
 save(JOURNAL,{'utc':utc(),'phase':'BEFORE_CANONICAL_ALIAS_PILOT','records':records,'SSD_released_bytes':0})
 for e in records:
  s=pathlib.Path(e['source']);b=pathlib.Path(e['backup']);a=pathlib.Path(e['alias']);os.link(s,b);a.symlink_to(e['staged_path']);os.replace(a,s)
 save(JOURNAL,{'utc':utc(),'phase':'CANONICAL_ALIASES_INSTALLED_ORIGINAL_INODES_BACKED','records':records,'SSD_released_bytes':0})
try:
 try:
  install()
  assert cache_d._context(R/'configs/execution.json',RUN)[0]==base
  model=models.build_model(task['backbone'],config['backbones'][task['backbone']]['weight_path'])
  opt=torch.optim.AdamW(model.parameters(),lr=task['lr'],weight_decay=5e-4)
  scaler=torch.amp.GradScaler('cpu',init_scale=128)
  state=trainer.load_state(R/'runs'/RUN/'state.pt',model,opt,scaler,base['training_identity'])
  assert state['epoch']==30 and state['attempts']==23970
  assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items())
  assert all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_floating_point(v))
  steps=[int(x['step'].item()) for x in opt.state.values() if 'step' in x];assert len(set(steps))==1 and steps[0]==sum(h['success'] for h in state['history'])
  assert scaler.state_dict()==state['scaler']
  e=records[1];arr=np.load(e['source'],allow_pickle=False);original=np.load(e['backup'],allow_pickle=False)
  assert len(arr)==102272 and arr.dtype==original.dtype and np.array_equal(arr,original)
  assert cache_d._context(R/'configs/execution.json',RUN)[0]==base
  assert not torch.cuda.is_initialized()
  doc['native_checks'].append({'status':'PASS_ACTUAL_CANONICAL_NATIVE_CONTEXT_LOAD_STATE_AND_NPY','epoch':state['epoch'],'attempts':state['attempts'],'success':sum(h['success'] for h in state['history']),'skips':sum(h['skip'] for h in state['history']),'optimizer_states':len(steps),'methods_rows':len(arr),'methods_dtype':str(arr.dtype),'CPU_model_optimizer_scaler_restored':True,'CUDA_initialized':False,'CUDA_RNG_restored_on_GPU':False})
 finally:restore()
 try:
  install();raise RuntimeError('EXPECTED_INJECTED_VALIDATION_FAILURE')
 except RuntimeError as exc:
  assert str(exc)=='EXPECTED_INJECTED_VALIDATION_FAILURE';doc['rollback_injected_failure']='PASS'
 finally:restore()
 assert keep=={p:sha(p) for p in keep}
 doc.update(status='PASS_CANONICAL_NATIVE_READER_AND_FAILURE_ROLLBACK',canonical_path_restored=True,primary_weights_unchanged=True,mtime_compatibility='Auxiliary state/npy native readers use verified bytes; original nanosecond timestamps restored exactly on SSD. Qualification and final/source-best paths excluded from migration.',seconds=time.monotonic()-started)
except BaseException as exc:
 doc.update(status='FAILED_WITH_RESTORE_ATTEMPT',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
 save(REPORT,doc);raise
save(REPORT,doc);print(json.dumps({k:v for k,v in doc.items() if k!='selected'},ensure_ascii=False))
