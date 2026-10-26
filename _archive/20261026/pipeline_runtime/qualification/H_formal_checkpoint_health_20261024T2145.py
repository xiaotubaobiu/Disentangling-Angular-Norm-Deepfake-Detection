"""Prepared CPU-only H formal atomic-checkpoint/paired-stream health audit.

Reuses H_EfficientNet_checkpoint_health_20261024T2045.py checks, with dynamic
formal tasks and dimensions. Only main() audits production; importing is inert.
A running checkpoint is an open-inode committed snapshot, not a terminal seal.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import hashlib,json,os,gc,random,time,math,re
import numpy as np
import pandas as pd
import torch

R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
REPORT=R/'reports/H_formal_checkpoint_health_20261024T2145.json'
DIMENSIONS={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}
METHODS={'real','Deepfakes','Face2Face','FaceSwap','NeuralTextures'}
DEV_AUDITS=('H_first4_terminal_acceptance_20261024T1630.json','H_DINO_lr1_terminal_acceptance_20261024T1800.json','H_DINO_lr3_terminal_acceptance_20261024T1925.json','H_DINO_lr5_terminal_acceptance_20261024T2010.json','H_EfficientNet_terminal_acceptance_20261024T2045.json')

def read(path):return json.loads(Path(path).read_text())

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def ensure_new_report(path):
 if Path(path).exists() or Path(path).is_symlink():raise FileExistsError('Preserve existing report: '+str(path))

def tensor_tree(value):
 count=0
 if isinstance(value,torch.Tensor):
  assert value.device.type=='cpu'
  if value.is_floating_point() or value.is_complex():assert torch.isfinite(value).all().item()
  return 1
 if isinstance(value,dict):
  for v in value.values():count+=tensor_tree(v)
 elif isinstance(value,(tuple,list)):
  for v in value:count+=tensor_tree(v)
 return count

def select_formal_jobs(queue,config):
 selected=[]
 for job in queue.get('active',[]):
  run=job.get('run_id','')
  if not run.startswith('H_formal_'):continue
  assert re.fullmatch(r'H_formal_[A-Za-z0-9_.-]+',run) and run in config['runs']
  task=config['runs'][run]
  assert task['route']=='H' and task['phase']=='formal' and task['backbone'] in DIMENSIONS
  assert task['history'] in ('B','AB') and task['seed'] in (7,42,123,2026,3407)
  assert task['precision']=='AMP','This audit is for original formal AMP stage only'
  assert task.get('recipe_frozen') is True
  assert isinstance(job['pid'],int) and job['pid']>0
  selected.append(job)
 assert len(selected)<=4 and len({j['run_id'] for j in selected})==len(selected)
 return selected

def inspect_state(state,events,task,identity,effective_batch):
 assert task['phase']=='formal' and task['backbone'] in DIMENSIONS and task['precision']=='AMP'
 assert effective_batch==128
 draws=int(task['draw_count']);assert draws>0 and draws%effective_batch==0
 steps=draws//effective_batch;epoch=int(state['epoch']);attempts=int(state['attempts'])
 assert 1<=epoch<=int(task['epochs']) and attempts==epoch*steps
 assert state['identity']==identity and state['scheduler_clock']=='attempt_count'
 history=state['history']
 assert len(history)==epoch and [h['epoch'] for h in history]==list(range(1,epoch+1))
 assert len(events)==attempts and [e['attempt'] for e in events]==list(range(attempts))
 for index,event in enumerate(events):
  assert event['epoch']==index//steps+1
  assert isinstance(event['success'],bool) and isinstance(event['skip'],bool) and event['success']!=event['skip']
  assert math.isfinite(event['loss']) and math.isfinite(event['lr']) and event['lr']>=0
  assert event['gradient_finite'] or event['skip']
  if event['gradient_finite']:assert event['gradient_norm'] is not None and math.isfinite(event['gradient_norm'])
  else:assert event['gradient_norm'] is None and event['skip']
  assert math.isfinite(event['scale_before']) and event['scale_before']>0 and math.isfinite(event['scale_after']) and event['scale_after']>0
  assert event['skip']==(event['scale_after']<event['scale_before'])
  if index:assert event['scale_before']==events[index-1]['scale_after']
 for i,h in enumerate(history):
  part=events[i*steps:(i+1)*steps]
  assert h['draw_count']==draws and h['attempts']==steps
  assert h['success']==sum(e['success'] for e in part) and h['skip']==sum(e['skip'] for e in part)
  assert h['success']+h['skip']==steps and math.isfinite(h['seconds']) and h['seconds']>=0
  assert math.isfinite(h['dev']['macro_auc']) and 0<=h['dev']['macro_auc']<=1
 success=sum(e['success'] for e in events);skips=[e for e in events if e['skip']]
 assert success+len(skips)==attempts
 ids=[p for group in state['optimizer']['param_groups'] for p in group['params']]
 assert ids and len(ids)==len(set(ids)) and set(ids)==set(state['optimizer']['state'])
 optimizer_steps=[]
 for parameter in state['optimizer']['state'].values():
  step=parameter['step'];step=step.item() if isinstance(step,torch.Tensor) else step
  assert int(step)==step==success;optimizer_steps.append(int(step))
  assert parameter['exp_avg'].shape==parameter['exp_avg_sq'].shape
 for group in state['optimizer']['param_groups']:
  assert group['lr']==events[-1]['lr'] and group['weight_decay']==5e-4
 finite=tensor_tree(state['model'])+tensor_tree(state['best_model'])+tensor_tree(state['optimizer'])
 dimension=DIMENSIONS[task['backbone']]
 for model in (state['model'],state['best_model']):
  assert tuple(model['classifier.weight'].shape)==(2,dimension)
  assert tuple(model['classifier.bias'].shape)==(2,)
 assert state['scaler'] and math.isfinite(float(state['scaler']['scale'])) and float(state['scaler']['scale'])>0
 assert float(state['scaler']['scale'])==events[-1]['scale_after']
 py=random.Random();py.setstate(state['python_rng'])
 nr=np.random.RandomState();nr.set_state(state['numpy_rng'])
 tg=torch.Generator(device='cpu');tg.set_state(state['torch_rng'])
 cuda=state['cuda_rng']
 assert isinstance(cuda,list) and len(cuda)==1
 assert all(x.device.type=='cpu' and x.dtype==torch.uint8 and x.ndim==1 and x.numel()>0 for x in cuda)
 trailing=0
 for event in reversed(events):
  if not event['skip']:break
  trailing+=1
 assert state['consecutive_skips']==trailing and trailing<8
 assert state['best_model'] is not None and 1<=state['best_epoch']<=epoch
 recovery=[dict(epoch=e['epoch'],attempt=e['attempt'],next_success=next((n['attempt'] for n in events[e['attempt']+1:] if n['success']),None)) for e in skips]
 return dict(snapshot_epoch=epoch,snapshot_attempts=attempts,snapshot_successful_updates=success,
  snapshot_AMP_skips=len(skips),skip_recovery_within_snapshot=recovery,
  finite_model_optimizer_best_tensors=finite,optimizer_parameter_states=len(ids),
  optimizer_step_min=min(optimizer_steps),optimizer_step_max=max(optimizer_steps),
  scaler_scale=state['scaler']['scale'],scheduler_clock=state['scheduler_clock'],
  python_numpy_cpu_torch_RNG_restore_to_local_generators=True,saved_CUDA_RNG_count=1,
  CUDA_RNG_restored_on_GPU_in_this_audit=False,CUDA_RNG_sha256=hashlib.sha256(cuda[0].numpy().tobytes()).hexdigest(),
  mean_committed_epoch_seconds=float(np.mean([h['seconds'] for h in history])),
  registered_epochs=task['epochs'],backbone_dimension=dimension,
  frozen_train_draws_per_epoch=draws,frozen_attempts_per_epoch=steps,
  status='PASS_COMPLETE_ATOMIC_CHECKPOINT_CPU_SNAPSHOT')

def load_events(folder,epoch,steps):
 result=[]
 for e in range(1,epoch+1):
  raw=(folder/f'epoch_{e:02}_updates.jsonl').read_bytes();assert raw.endswith(b'\n')
  records=[json.loads(line) for line in raw.splitlines() if line.strip()]
  assert len(records)==steps and all(r['epoch']==e for r in records);result.extend(records)
 return result

def verify_arrays(a,b,tables,draws):
 for arrays in (a,b):
  indices=arrays['samples'];assert indices.shape==(draws,) and np.issubdtype(indices.dtype,np.integer)
  assert indices.min()>=0 and indices.max()<len(tables['B'])
  assert arrays['methods'].shape==(draws,) and set(map(str,np.unique(arrays['methods'])))<=METHODS
  aug=arrays['augmentations'];assert aug.shape==(draws,3) and np.isfinite(aug).all()
  assert np.isin(aug[:,0],[0,1]).all() and (aug[:,1:]>=.9).all() and (aug[:,1:]<=1.1).all()
 for key in ('samples','methods','augmentations'):assert np.array_equal(a[key],b[key])
 np.testing.assert_array_equal(tables['B'].mother_id.to_numpy()[a['samples']],tables['AB'].mother_id.to_numpy()[b['samples']])


def verify_recipe(root,task,config):
 path=Path(root)/'configs/H_frozen_recipes.json'
 assert path.is_file() and not path.is_symlink()
 recipe_sha=sha(path);assert recipe_sha==task['source_recipe_sha256']
 recipe=read(path);selected=recipe['recipes'][task['backbone']]
 assert task['recipe_frozen'] is True and recipe['target_scores_used'] is False
 assert selected['lr']==task['lr'] and selected['epochs']==task['epochs']
 assert selected['shared_B_AB_recipe'] is True and selected['selection_source_only'] is True
 for key in ('H_data_freeze_sha256','h_code_bundle_sha256'):
  assert recipe[key]==task[key]==config[key]
 assert sha(Path(root)/'data/H/freeze.json')==config['H_data_freeze_sha256']
 assert sha(path)==recipe_sha
 return dict(source_recipe_path=str(path),source_recipe_sha256=recipe_sha,
  H_data_freeze_sha256=config['H_data_freeze_sha256'],h_code_bundle_sha256=config['h_code_bundle_sha256'])

def snapshot(root,run,task,job,config,gate_hash):
 lineage=verify_recipe(root,task,config)
 folder=root/'runs'/run
 assert folder.resolve().is_relative_to((root/'runs').resolve())
 with (folder/'state.pt').open('rb') as stream:
  st=os.fstat(stream.fileno());state=torch.load(stream,map_location='cpu',weights_only=False)
  stream.seek(0);digest=hashlib.file_digest(stream,'sha256').hexdigest();after=os.fstat(stream.fileno())
 assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 identity=read(folder/'identity.json');assert read(folder/'resolved_config.json')==task
 assert identity['config']==hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
 assert identity['qualification']==gate_hash and identity['train_manifest']==sha(root/task['train_manifest']) and identity['dev_manifest']==sha(root/task['dev_manifest'])
 assert task['draw_count']==config['H_training_draw_count']
 events=load_events(folder,int(state['epoch']),task['draw_count']//128)
 result=inspect_state(state,events,task,identity,config['effective_batch'])
 pid=job.get('pid') if job else None;matching=False
 if pid:
  try:words=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except FileNotFoundError:words=''
  matching='reliability_v2.worker_h' in words and run in words and str(root/'configs/H_execution.json') in words
  if not matching:assert (folder/'complete.json').is_file() and result['snapshot_epoch']==task['epochs'],'stale/nonmatching worker without terminal checkpoint'
 try:same=(folder/'state.pt').stat().st_ino==st.st_ino
 except FileNotFoundError:same=False
 result.update(run_id=run,backbone=task['backbone'],history=task['history'],seed=task['seed'],precision=task['precision'],pid=pid,
  matching_worker_command_at_check=matching,selected_from_active_status=job is not None,
  state_sha256=digest,state_inode_at_open=st.st_ino,state_bytes=st.st_size,path_still_same_inode=same,
  identity_sha256=sha(folder/'identity.json'),**lineage)
 committed_history=state['history']
 event_flags=[bool(e['skip']) for e in events]
 del state;gc.collect()
 return result,committed_history,event_flags

def development_and_D_context(root,config):
 dev_records=[];source_hashes={}
 for name in DEV_AUDITS:
  p=root/'reports'/name;a=read(p);assert a['status'].startswith('PASS_') and a['source_only'] is True and len(a['records'])==4;source_hashes[str(p)]=sha(p);dev_records.extend(a['records'])
 assert len(dev_records)==20 and len({r['run_id'] for r in dev_records})==20
 assert all(config['runs'][r['run_id']]['phase']=='development' for r in dev_records)
 development=dict(model_count=20,attempts=sum(r['snapshot_attempts'] for r in dev_records),
   successful_updates=sum(r['snapshot_successful_updates'] for r in dev_records),AMP_skips=sum(r['snapshot_AMP_skips'] for r in dev_records),
   evidence='five previously accepted standard-development terminal receipts; not a new checkpoint audit',receipt_sha256=source_hashes)
 assert development['attempts']==114600 and development['AMP_skips']==0
 dc=read(root/'configs/execution.json');counts={phase:dict(models=0,attempts=0,successful_updates=0,AMP_skips=0) for phase in ('development','formal')}
 for run,t in dc['runs'].items():
  if t.get('route')!='D' or t.get('phase') not in counts:continue
  p=root/'runs'/run/'complete.json';c=read(p);assert c['status']=='TRAINING_COMPLETE' and c['epochs']==t['epochs']
  row=counts[t['phase']];row['models']+=1;row['attempts']+=c['attempts'];row['successful_updates']+=c['success'];row['AMP_skips']+=c['skip']
 d=dict(by_phase=counts,total_AMP_skips=sum(v['AMP_skips'] for v in counts.values()),
  evidence='D completion-receipt metadata only; no D checkpoint or AMP kernel re-audit in this script')
 return development,d

def main():
 ensure_new_report(REPORT)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert all(os.environ.get(n)=='2' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 start=time.monotonic();queue=read(R/'H_queue_status.json');config=read(R/'configs/H_execution.json')
 jobs=select_formal_jobs(queue,config)
 if not jobs:raise RuntimeError('No active registered H formal AMP runs; report not written')
 byrun={j['run_id']:j for j in jobs};selected=dict(byrun);pending=[]
 # Include a current/finished partner when a queue refresh has moved one device to another seed.
 for job in jobs:
  task=config['runs'][job['run_id']]
  mate=[r for r,t in config['runs'].items() if r.startswith('H_formal_') and t.get('phase')=='formal' and
   t['backbone']==task['backbone'] and t['seed']==task['seed'] and t['precision']==task['precision'] and t['history']!=task['history']]
  assert len(mate)==1
  if mate[0] not in selected:
   if (R/'runs'/mate[0]/'state.pt').is_file():selected[mate[0]]=None
   else:pending.append(dict(backbone=task['backbone'],seed=task['seed'],missing_committed_history=config['runs'][mate[0]]['history'],run_id=mate[0]))
 for run in selected:
  verify_recipe(R,config['runs'][run],config)
  if not (R/'runs'/run/'state.pt').is_file():raise RuntimeError('First atomic formal state not committed yet: '+run+'; report not written')
 results=[];histories={};flags={};gate_hash=sha(R/'qualification/H_gate.json')
 for run,job in selected.items():
  print('Inspecting committed CPU snapshot '+run,flush=True)
  result,h,event_flags=snapshot(R,run,config['runs'][run],job,config,gate_hash);results.append(result);histories[run]=h;flags[run]=event_flags
  print(json.dumps(result),flush=True)
 tables={h:pd.read_parquet(R/f'data/H/{h}/train.parquet') for h in ('B','AB')}
 assert len(tables['B'])==len(tables['AB'])==config['H_training_rows'] and tables['B'].mother_id.tolist()==tables['AB'].mother_id.tolist()
 grouped=defaultdict(dict)
 for result in results:grouped[(result['backbone'],result['seed'])][result['history']]=result
 pairs=[];asymmetric=[]
 for (bb,seed),members in grouped.items():
  if set(members)!= {'B','AB'}:continue
  B,AB=members['B'],members['AB'];bt=config['runs'][B['run_id']];at=config['runs'][AB['run_id']]
  for key in ('epochs','draw_count','lr','microbatch','h_code_bundle_sha256','H_data_freeze_sha256'):assert bt[key]==at[key]
  epochs=min(B['snapshot_epoch'],AB['snapshot_epoch']);draws=bt['draw_count'];steps=draws//128;hashes=[]
  for e in range(1,epochs+1):
   arrays=[]
   for item in [B,AB]:
    folder=R/'runs'/item['run_id'];a={k:np.load(folder/f'epoch_{e:02}_{k}.npy',allow_pickle=False) for k in ['samples','methods','augmentations']};arrays.append(a)
    h=histories[item['run_id']][e-1]
    assert sha(folder/f'epoch_{e:02}_samples.npy')==h['sample_sha256'] and sha(folder/f'epoch_{e:02}_augmentations.npy')==h['augmentation_sha256']
    assert dict(Counter(map(str,a['methods'].tolist())))==h['method_counts']
   verify_arrays(arrays[0],arrays[1],tables,draws)
   hashes.append(dict(epoch=e,**{k+'_sha256':sha(R/'runs'/B['run_id']/f'epoch_{e:02}_{k}.npy') for k in ['samples','methods','augmentations']}))
  same=flags[B['run_id']][:epochs*steps]==flags[AB['run_id']][:epochs*steps]
  pair=dict(backbone=bb,seed=seed,B_run=B['run_id'],AB_run=AB['run_id'],common_snapshot_epochs_checked=epochs,
   samples_methods_augmentations_and_actual_mother_sequence_equal=True,same_successful_update_stream_in_common_snapshot=same,hashes=hashes)
  pairs.append(pair)
  if not same:asymmetric.append(dict(backbone=bb,seed=seed,common_epochs=epochs,requires_source_review_and_all5_pair_terminal_protocol=True))
 development,dcontext=development_and_D_context(R,config)
 assert not torch.cuda.is_initialized()
 status='PASS_SELECTED_H_FORMAL_ATOMIC_SNAPSHOTS_AND_AVAILABLE_PAIRED_STREAMS'
 if asymmetric:status='ATOMIC_SNAPSHOTS_VALID_AMP_PAIR_SKIP_ASYMMETRY_SOURCE_REVIEW_REQUIRED'
 report=dict(utc=datetime.now(timezone.utc).isoformat(),status=status,source_only=True,
  queue_snapshot_utc=queue.get('utc'),queue_owner_pid=queue.get('owner_pid'),active_formal_run_ids=list(byrun),
  records=results,paired_streams=pairs,pending_pair_states=pending,observed_pair_update_asymmetries=asymmetric,
  H_formal_observed_snapshots=dict(unique_models=len(results),attempts=sum(r['snapshot_attempts'] for r in results),
    successful_updates=sum(r['snapshot_successful_updates'] for r in results),AMP_skips=sum(r['snapshot_AMP_skips'] for r in results),
    scope='only selected unique committed formal checkpoints at separate times; not all30 or terminal totals'),
  H_completed_standard_development=development,D_completed_receipt_context=dcontext,
  audit_seconds=time.monotonic()-start,CUDA_initialized=False,production_processes_restarted=False,
  target_export_qualification_claimed=False,whole_formal_training_complete_claimed=False,
  scope='Atomic open-inode snapshots; paths can advance during audit. Only checkpoint-committed epochs are read. Local CPU generators are restored; CUDA RNG is only verified/hashed without GPU restore. Paired checks cover each observed seed at common committed epochs. Early0 skips do not imply whole-training0. All5 terminal paired ledgers and any uniformFP32 capacity admission remain future root-owned gates.')
 with REPORT.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps(dict(status=status,report=str(REPORT),audit_seconds=report['audit_seconds'])),flush=True)
if __name__=='__main__':main()
