"""CPU-only verification of newly completed, original-queue models.
Run interactively; never schedules work or changes source/config/frozen data.
"""
from pathlib import Path
import os,sys,json,datetime,math,csv,io
root=Path('/home/z/project/deepfake_detector')
artifact=root/'artifacts'
reports=root/'reports'
cfg=json.loads((root/'configs/v1.1.json').read_text())
budget=json.loads((reports/'formal_training_budget_receipts.json').read_text())
proof=json.loads((reports/'completed_model_state_receipts.json').read_text())
assert isinstance(budget['completed_models'],dict)
assert isinstance(proof['completed_models'],dict)
todo=[]
for backbone in cfg['backbones']:
 for seed in cfg['seeds']:
  rid=f'{backbone}_seed{seed}'
  if (artifact/'runs'/rid/'complete.json').exists() and rid not in budget['completed_models']:
   todo.append((backbone,seed,rid))
if not todo:
 print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_models':[],'verified_models':len(budget['completed_models']),'no_files_changed':True}))
 sys.exit(0)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
import torch,numpy as np,pandas as pd
torch.set_num_threads(2)
sys.path.insert(0,str(root/'src'))
from deepfake_detector.training import run_identity
from deepfake_detector.manifest import digest,save_json
import inspect
contract=budget['contract']
assert torch.__version__==contract['torch_version']
assert digest(inspect.getsourcefile(torch.amp.GradScaler))==contract['GradScaler_source_sha256']
assert digest(root/'src/deepfake_detector/training.py')==contract['training_source_sha256']
assert digest(Path('/home/z/obsidian/paper/deepfake-detector/实验重设计_3backbone_5seed_v1.1.md'))==contract['design_sha256']
assert cfg['epochs']==contract['epochs']==15
assert cfg['effective_batch']==contract['effective_batch']==128
train=pd.read_parquet(artifact/'data/train.parquet',columns=['sampling_mass'])
pool=len(train);mass=train.sampling_mass.to_numpy(dtype=float)
assert pool==contract['N_pool']==250000
assert np.isfinite(mass).all() and (mass>=0).all() and np.isclose(mass.sum(),1.)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
new=[]
for backbone,seed,rid in todo:
 out=artifact/'runs'/rid
 expected=run_identity(cfg,backbone,seed,artifact)
 marker=json.loads((out/'complete.json').read_text())
 assert marker['status']=='TRAINING_COMPLETE' and marker['selection']=='final_epoch_only'
 assert marker['identity']==expected and marker['epoch']==15
 final_hash=digest(out/'final.pt')
 assert final_hash==marker['final_sha256']
 state_hash=digest(out/'state.pt')
 state=torch.load(out/'state.pt',map_location='cpu',weights_only=False)
 final=torch.load(out/'final.pt',map_location='cpu',weights_only=True)
 assert state['identity']==final['identity']==expected and state['epoch']==final['epoch']==15
 assert state['resume_scope']=='completed_epoch_boundary'
 history=json.loads((out/'history.json').read_text())
 assert state['history']==history and [h['epoch'] for h in history]==list(range(1,16))
 assert all(h['sample_slots']==pool and math.isfinite(h['loss']) and math.isfinite(h['seconds']) and h['seconds']>0 for h in history)
 draws=sum(h['sample_slots'] for h in history)
 attempts=sum(math.ceil(h['sample_slots']/cfg['effective_batch']) for h in history)
 assert draws==contract['draw_budget']==3750000 and attempts==contract['attempt_budget']==29310
 steps=[]
 for param in state['optimizer']['state'].values():
  if 'step' in param:
   value=float(param['step'].item() if torch.is_tensor(param['step']) else param['step'])
   assert math.isfinite(value) and value.is_integer()
   steps.append(int(value))
  for value in param.values():
   if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
    assert bool(torch.isfinite(value).all()),f'nonfinite optimizer: {rid}'
 assert steps and len(set(steps))==1 and 0<steps[0]<=attempts
 successes=steps[0];skips=attempts-successes
 assert state['model'].keys()==final['model'].keys()
 for key,tensor in state['model'].items():
  assert torch.equal(tensor,final['model'][key]),f'final/full state mismatch: {rid} {key}'
  if tensor.is_floating_point() or tensor.is_complex():
   assert bool(torch.isfinite(tensor).all()),f'nonfinite model: {rid} {key}'
 rng_keys=['python_rng','numpy_rng','torch_rng','cuda_rng']
 assert all(key in state and state[key] is not None for key in rng_keys)
 assert isinstance(state['python_rng'],tuple) and isinstance(state['numpy_rng'],tuple)
 assert state['torch_rng'].dtype==torch.uint8 and state['torch_rng'].ndim==1
 assert isinstance(state['cuda_rng'],list) and len(state['cuda_rng'])==1
 assert all(t.dtype==torch.uint8 and t.ndim==1 for t in state['cuda_rng'])
 sequences=[]
 for h in history:
  ep=h['epoch'];path=out/f'epoch_{ep:02d}_samples.npy';sha=digest(path)
  assert sha==h['sample_sequence_sha256']
  actual=np.load(path,allow_pickle=False)
  expected_seq=np.random.default_rng(seed+100003*(ep-1)).choice(pool,size=pool,replace=True,p=mass)
  assert actual.shape==(pool,) and np.array_equal(actual,expected_seq),f'sampling sequence mismatch: {rid} epoch{ep}'
  sequences.append({'epoch':ep,'draws':pool,'sha256':sha,'exact_regeneration_match':True})
 for group in state['optimizer']['param_groups']:
  assert group['lr']==cfg['backbones'][backbone]['learning_rate']
  assert group['weight_decay']==cfg['weight_decay']
 scaler=state['scaler']
 assert scaler['growth_factor']==2. and scaler['backoff_factor']==.5 and scaler['growth_interval']==2000
 assert math.isfinite(scaler['scale']) and scaler['scale']>0
 assert 0<=scaler['_growth_tracker']<2000
 log_ratio=math.log2(scaler['scale']/128)
 assert log_ratio.is_integer()
 bn=None
 if backbone=='efficientnet_b0':
  initial=torch.load(cfg['backbones'][backbone]['weight_path'],map_location='cpu',weights_only=True)
  if 'state_dict' in initial:initial=initial['state_dict']
  counters={key:t for key,t in state['model'].items() if key.endswith('num_batches_tracked')}
  assert len(counters)==49
  initial_counts=[];final_counts=[];deltas=[]
  for key,value in counters.items():
   init_key=key.removeprefix('encoder.')
   start=int(initial[init_key].item());end=int(value.item())
   initial_counts.append(start);final_counts.append(end);deltas.append(end-start)
  assert len(set(initial_counts))==len(set(deltas))==1 and deltas[0]==attempts
  bn={'layers':len(counters),'initial_counter':initial_counts[0],'final_counter_min':min(final_counts),'final_counter_max':max(final_counts),'per_layer_forward_counter_delta':attempts,'all_attempted_forwards_counted_including_skips':True}
 detail={'verified_UTC':now,'run_id':rid,'backbone':backbone,'seed':seed,'epoch':15,'draws':draws,'attempts':attempts,'successes':successes,'AMP_skips':skips,
  'count_evidence':'successes observed in every initialized optimizer parameter step; attempts reconstructed from verified history and frozen loop; skips are their difference',
  'optimizer_parameter_states_checked':len(steps),'learning_rate':cfg['backbones'][backbone]['learning_rate'],'scheduler':'none','AMP_initial_scale':128,'AMP_final_state':scaler,
  'aggregate_growth_events_derived_from_scale_ratio_and_skip_count':int(log_ratio)+skips,
  'skip_step_trace_available':False,
  'per_skip_details':[{'ordinal':i+1,'attempt_step':None,'epoch':None,'batch_slot_range':None,'scale_before':None,'scale_after':None,'scale_relation':'after = before * 0.5 per pinned GradScaler implementation; event numeric values not recorded'} for i in range(skips)],
  'logging_gap':'Frozen training worker records epoch-level history and end-of-epoch scaler state only; no per-step overflow/scale event log. Exact skip steps and individual scale values cannot be recovered from these records without prohibited training replay.',
  'data_behavior':'All sampled batches are read, augmented, forwarded and backwarded before scaler.step; a nonfinite-gradient event skips optimizer.step only; the batch is not replayed; RNG and BatchNorm state continue; next attempt clears gradients. Exact skipped batches and their slot count are not recoverable.',
  'budget_contract_assessment':'fixed-epoch draw budget satisfied; not a claim of complete per-step AMP audit',
  'no_compensating_steps_or_hyperparameter_changes':True,'final_model_sha256':final_hash,'state_sha256':state_hash,'source_code_sha256':expected['code_sha256']}
 if bn is not None:detail['BatchNorm_evidence']=bn
 budget['completed_models'][rid]=detail
 proof['completed_models'][rid]={'verified_UTC':now,'run_id':rid,'backbone':backbone,'seed':seed,'epoch':15,'draws':draws,'attempts':attempts,'successes':successes,'AMP_skips':skips,
  'final_model_sha256':final_hash,'state_sha256':state_hash,'source_code_sha256':expected['code_sha256'],'selection':'final_epoch_only','final_training_loss':history[-1]['loss'],
  'epoch_training_seconds':sum(h['seconds'] for h in history),'all_epoch_sequences_exactly_reconstructed':True,'epoch_sequences':sequences,
  'final_model_matches_full_state':True,'model_and_optimizer_finite':True,'full_Python_NumPy_CPU_CUDA_RNG_present':True,'CUDA_RNG_state_count':len(state['cuda_rng']),
  'AMP_final_state':scaler,'per_step_AMP_logging_gap_disclosed':True}
 new.append({'run_id':rid,'epoch':15,'draws':draws,'attempts':attempts,'successes':successes,'AMP_skips':skips,'AMP_final_scale':scaler['scale']})
 del state,final
assert not torch.cuda.is_initialized()
budget.update(checked_UTC=now,completed_models_count=len(budget['completed_models']),audit_gap_disclosed=True,CUDA_initialized=False)
proof.update(checked_UTC=now,training_complete_models=len(proof['completed_models']),CUDA_initialized=False)
save_json(reports/'formal_training_budget_receipts.json',budget)
save_json(reports/'completed_model_state_receipts.json',proof)
columns=['run_id','backbone','seed','state','epoch','draws','attempts','successes','AMP_skips','AMP_final_scale','skip_step_trace_available','draw_budget','attempt_budget','fixed_successful_update_contract']
stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
for backbone in cfg['backbones']:
 for seed in cfg['seeds']:
  rid=f'{backbone}_seed{seed}';v=budget['completed_models'].get(rid)
  row={'run_id':rid,'backbone':backbone,'seed':seed,'state':'TRAINING_COMPLETE' if v else 'TRAINING_NOT_COMPLETE','skip_step_trace_available':False,'draw_budget':contract['draw_budget'],'attempt_budget':contract['attempt_budget'],'fixed_successful_update_contract':False}
  if v:
   row.update({key:v[key] for key in ['epoch','draws','attempts','successes','AMP_skips']});row['AMP_final_scale']=v['AMP_final_state']['scale']
  writer.writerow(row)
target=reports/'formal_training_budget_counts.csv';tmp=target.with_name(target.name+'.tmp')
tmp.write_text(stream.getvalue(),encoding='utf-8');os.replace(tmp,target)
with (reports/'audit.log').open('a') as log:
 log.write(json.dumps({'UTC':now,'event':'independent_completed_model_state_and_fixed_draw_budget_verified','new_models':new,'CUDA_initialized':False,'settings_changed':False},ensure_ascii=False)+'\n')
print(json.dumps({'UTC':now,'new_models':new,'verified_models':len(budget['completed_models']),'CUDA_initialized':False,'per_step_AMP_gap_disclosed':True}))
