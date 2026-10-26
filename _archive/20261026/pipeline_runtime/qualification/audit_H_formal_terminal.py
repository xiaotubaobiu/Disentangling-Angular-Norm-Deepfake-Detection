"""Prepared read-only CPU terminal audit for sealed E30 H formal AMP pairs.

Importing is inert. Root invokes main only after selected B/AB runs are sealed.
This accepts paired subsets and never grants a five-seed target gate.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,bisect,gc,hashlib,importlib.util,json,math,os,re,sys,time
import numpy as np
import pandas as pd
import torch

R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
HEALTH_FILE=R/'qualification/H_formal_checkpoint_health_20261024T2145.py'
HEALTH_SHA='c59bd1514efa1892027bab7910d25ff6a3057efaae82d3aff988e85a12058cc6'
assert hashlib.sha256(HEALTH_FILE.read_bytes()).hexdigest()==HEALTH_SHA,'Frozen health helper changed'
_spec=importlib.util.spec_from_file_location('H_frozen_formal_health',HEALTH_FILE)
health=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(health)
METHODS={'Deepfakes','Face2Face','FaceSwap','NeuralTextures'}
PAIR_FIELDS=('epochs','draw_count','lr','microbatch','h_code_bundle_sha256','H_data_freeze_sha256','source_recipe_sha256')
def parse_args(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run-id',action='append',required=True,help='Repeat for each complete B/AB pair')
 p.add_argument('--report',required=True,help='New .json basename or absolute path directly in R/reports')
 return p.parse_args(argv)

def artifact_names(epochs):
 names=['complete.json','history.json','identity.json','resolved_config.json','state.pt','source_best.pt','final.pt','H_training_start.json']
 for epoch in range(1,epochs+1):
  names += [f'epoch_{epoch:02}_{suffix}' for suffix in ('updates.jsonl','samples.npy','methods.npy','augmentations.npy','source_dev.parquet')]
 return names

def report_path(root,value):
 root=Path(root);reports=root/'reports';raw=Path(value)
 assert reports.is_dir() and not reports.is_symlink()
 if raw.is_absolute():path=raw
 else:
  assert raw.name==value,'Report must be a simple basename or absolute reports path'
  path=reports/raw
 assert path.parent.resolve()==reports.resolve() and path.suffix=='.json'
 assert re.fullmatch(r'[A-Za-z0-9_.-]+\.json',path.name) and not path.name.startswith('.')
 health.ensure_new_report(path)
 return path

def select_pairs(run_ids,config):
 assert len(run_ids)>=2 and len(set(run_ids))==len(run_ids),'Duplicate/empty run selection'
 groups={}
 for run in run_ids:
  assert re.fullmatch(r'H_formal_[A-Za-z0-9_.-]+',run) and run in config['runs'],'Unsafe/unregistered formal run'
  t=config['runs'][run]
  assert t['route']=='H' and t['phase']=='formal' and t['precision']=='AMP' and t['epochs']==30
  assert t['recipe_frozen'] is True and t['history'] in ('B','AB') and t['backbone'] in health.DIMENSIONS
  assert t['seed'] in (7,42,123,2026,3407) and t['draw_count']>0 and t['draw_count']%128==0
  assert run==f"H_formal_{t['backbone']}_{t['history']}_seed{t['seed']}",'Noncanonical formal AMP run'
  members=groups.setdefault((t['backbone'],t['seed']),{})
  assert t['history'] not in members;members[t['history']]=run
 for members in groups.values():
  assert set(members)=={'B','AB'},'Every selected backbone/seed must include both histories'
  b,a=(config['runs'][members[h]] for h in ('B','AB'))
  assert all(b[k]==a[k] for k in PAIR_FIELDS),'Pair frozen recipe/exposure differs'
 return groups

def check_terminal_ready(root,run,task,names):
 folder=Path(root)/'runs'/run
 assert folder.is_dir() and not folder.is_symlink() and folder.resolve().parent==(Path(root)/'runs').resolve()
 assert task['epochs']==30 and len(names)==158 and len(set(names))==158
 for name in ['H_completion_seal.json']+list(names):
  path=folder/name
  assert path.is_file() and not path.is_symlink(),'Missing/unsealed terminal artifact: '+str(path)
 receipt=health.read(folder/'complete.json');seal=health.read(folder/'H_completion_seal.json')
 assert receipt['status']=='TRAINING_COMPLETE' and receipt['epochs']==30
 assert receipt['attempts']==30*(task['draw_count']//128)
 assert receipt['success']>=0 and receipt['skip']>=0 and receipt['success']+receipt['skip']==receipt['attempts']
 assert receipt['selection']=='frozen_last' and receipt['target_used'] is False
 assert seal['status']=='SEALED' and seal['run_id']==run and set(seal['artifact_sha256'])==set(names)
 assert all(re.fullmatch(r'[0-9a-f]{64}',v) for v in seal['artifact_sha256'].values())
 return dict(terminal_complete_sha256=health.sha(folder/'complete.json'),terminal_seal_sha256=health.sha(folder/'H_completion_seal.json'),artifact_sha256=seal['artifact_sha256'])

def equal_model(actual,expected):
 assert set(actual)==set(expected)
 for key,value in expected.items():
  other=actual[key]
  assert isinstance(value,torch.Tensor) and isinstance(other,torch.Tensor)
  assert value.device.type==other.device.type=='cpu' and value.dtype==other.dtype and value.shape==other.shape
  assert torch.equal(value,other),'Terminal model tensor differs: '+key

def verify_completed_weights(state,final,best,identity,receipt):
 assert state['identity']==identity and state['epoch']==30
 assert final['identity']==identity and final['epoch']==30
 assert best['identity']==identity and best['epoch']==state['best_epoch']==receipt['source_best_epoch']
 health.tensor_tree(final['model']);health.tensor_tree(best['model'])
 equal_model(final['model'],state['model']);equal_model(best['model'],state['best_model'])
 scores=[float(h['dev']['macro_auc']) for h in state['history']]
 assert len(scores)==30 and all(math.isfinite(v) for v in scores)
 assert state['best_score']==max(scores) and state['best_epoch']==scores.index(max(scores))+1
 return dict(final_matches_complete_state=True,source_best_matches_best_state=True,source_best_epoch=state['best_epoch'],source_best_score=state['best_score'])

def independent_auc(scores,labels):
 values=[float(v) for v in scores];ys=[int(y) for y in labels]
 assert len(values)==len(ys) and set(ys)=={0,1} and all(math.isfinite(v) for v in values)
 negative=sorted(v for v,y in zip(values,ys) if y==0)
 positive=[v for v,y in zip(values,ys) if y==1]
 twice=sum(bisect.bisect_left(negative,v)+bisect.bisect_right(negative,v) for v in positive)
 return twice/(2*len(negative)*len(positive))

def verify_dev_table(table,expected,metrics):
 assert list(table.columns)==['video_uid','method_id','label','score']
 assert list(expected.columns)==['video_uid','method_id','label']
 assert len(table)==len(expected) and table.video_uid.is_unique and expected.video_uid.is_unique
 assert not table.isna().any().any() and np.issubdtype(table.label.dtype,np.integer) and set(table.label)=={0,1}
 assert pd.api.types.is_numeric_dtype(table.score) and np.isfinite(table.score.to_numpy()).all()
 assert set(table.loc[table.label.eq(1),'method_id'])==METHODS
 assert set(table.loc[table.label.eq(0),'method_id'])=={'real'}
 def identities(frame):
  return sorted((str(r.video_uid),str(r.method_id),int(r.label)) for r in frame.itertuples(index=False))
 assert identities(table[['video_uid','method_id','label']])==identities(expected),'Source DEV cohort changed'
 assert set(metrics['method_auc'])==METHODS
 actual=independent_auc(table.score,table.label);errors=[abs(actual-metrics['pooled_auc'])]
 methods=[]
 for method in sorted(METHODS):
  part=table[table.label.eq(0)|table.method_id.eq(method)]
  value=independent_auc(part.score,part.label);methods.append(value);errors.append(abs(value-metrics['method_auc'][method]))
 errors.append(abs(math.fsum(methods)/4-metrics['macro_auc']))
 assert all(math.isfinite(float(v)) and 0<=v<=1 for v in [metrics['pooled_auc'],metrics['macro_auc'],*metrics['method_auc'].values()])
 assert max(errors)<=3e-15,'Independent saved source DEV AUC mismatch; no repair permitted'
 return errors

def skip_alignment(b_flags,a_flags,attempts):
 assert len(b_flags)==len(a_flags)==attempts and all(type(v) is bool for v in b_flags+a_flags)
 diff=[i for i,(b,a) in enumerate(zip(b_flags,a_flags)) if b!=a]
 return dict(same_successful_update_stream=not diff,different_attempt_indices=diff,
  B_skip_attempts=[i for i,v in enumerate(b_flags) if v],AB_skip_attempts=[i for i,v in enumerate(a_flags) if v],
  all5_seed_paired_ledger_review_still_required=True)

def stable_bound(folder,bound):
 assert health.sha(folder/'H_completion_seal.json')==bound['terminal_seal_sha256']
 assert health.sha(folder/'complete.json')==bound['terminal_complete_sha256']
 for name,digest in bound['artifact_sha256'].items():
  assert (folder/name).is_file() and not (folder/name).is_symlink()
  assert health.sha(folder/name)==digest,'Sealed artifact changed during audit: '+name

def audit(run_ids,report):
 # This is the sole real-data entry. All tests call small pure helpers only.
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert all(os.environ.get(n)=='2' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 destination=report_path(R,report);start=time.monotonic()
 config_path=R/'configs/H_execution.json';config_sha=health.sha(config_path);config=health.read(config_path)
 assert Path(config['artifact_root']).resolve()==R and config['schema']=='H_EXECUTION_V2'
 assert config['effective_batch']==128 and config['H_training_draw_count']==24448 and config['H_training_rows']==24424
 pairs=select_pairs(run_ids,config)
 sys.path.insert(0,str(V))
 from reliability_v2 import worker_h
 assert worker_h.h_code_hash()==config['h_code_bundle_sha256']
 names=worker_h._completion_artifacts(30);assert names==artifact_names(30)
 bound={};receipts={}
 # Every selected original seal must pass before any checkpoint/source-score audit.
 for run in run_ids:
  print('Original terminal seal verification '+run,flush=True,file=sys.stderr)
  task=config['runs'][run];health.verify_recipe(R,task,config)
  bound[run]=check_terminal_ready(R,run,task,names)
  receipts[run]=worker_h.verify_run_completion(config_path,run)
 assert health.sha(config_path)==config_sha
 records=[];histories={};flags={};gate_hash=health.sha(R/'qualification/H_gate.json')
 for run in run_ids:
  print('CPU terminal state/weights/source DEV '+run,flush=True,file=sys.stderr)
  task=config['runs'][run];folder=R/'runs'/run
  record,history,event_flags=health.snapshot(R,run,task,None,config,gate_hash)
  assert record['snapshot_epoch']==30 and record['snapshot_attempts']==5730
  assert record['state_sha256']==bound[run]['artifact_sha256']['state.pt']
  receipt=receipts[run]
  assert receipt['success']==record['snapshot_successful_updates'] and receipt['skip']==record['snapshot_AMP_skips']
  assert receipt['target_used'] is False
  state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
  final=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
  best=torch.load(folder/'source_best.pt',map_location='cpu',weights_only=False)
  assert state['history']==history==health.read(folder/'history.json')
  weights=verify_completed_weights(state,final,best,health.read(folder/'identity.json'),receipt)
  events=health.load_events(folder,30,task['draw_count']//128)
  for event in events:
   assert event['lr']==worker_h.trainer.attempt_lr(event['attempt'],task['draw_count']//128,30,task['lr']),'Attempt-count LR differs'
  for entry in history:
   part=events[(entry['epoch']-1)*191:entry['epoch']*191]
   assert math.isclose(entry['train_loss'],sum(e['loss'] for e in part)/191,rel_tol=1e-14,abs_tol=1e-15)
  manifest=pd.read_parquet(R/task['dev_manifest'],columns=['video_uid','method_id','label'])
  expected=manifest.groupby(['video_uid','method_id','label'],sort=True,as_index=False).size()[['video_uid','method_id','label']]
  assert len(expected)==535 and expected.video_uid.is_unique
  errors=[]
  for entry in history:
   table=pd.read_parquet(folder/f"epoch_{entry['epoch']:02}_source_dev.parquet")
   errors.extend(verify_dev_table(table,expected,entry['dev']))
  assert len(errors)==180
  record.update(**weights,terminal_complete_sha256=bound[run]['terminal_complete_sha256'],terminal_seal_sha256=bound[run]['terminal_seal_sha256'],
    seal_current_artifacts_verified=158,original_worker_completion_verifier_passed=True,
    source_DEV_natural_videos_per_epoch=535,source_DEV_tables_independently_recomputed=30,
    source_DEV_metric_checks=len(errors),source_DEV_max_auc_abs_error=max(errors),
    attempt_LR_clock_verified_with_original_function=True,source_only=True,target_used=False,
    status='PASS_H_FORMAL_AMP_E30_TERMINAL_CPU_CHECKPOINT_AND_SOURCE_DEV')
  records.append(record);histories[run]=history;flags[run]=event_flags
  del state,final,best,events,manifest,expected,table;gc.collect()
 tables={h:pd.read_parquet(R/f'data/H/{h}/train.parquet') for h in ('B','AB')}
 assert len(tables['B'])==len(tables['AB'])==24424
 assert tables['B'].mother_id.tolist()==tables['AB'].mother_id.tolist()
 paired=[];asymmetries=[]
 for (bb,seed),members in sorted(pairs.items()):
  print(f'Full E30 paired source streams {bb} seed{seed}',flush=True,file=sys.stderr)
  draws=config['runs'][members['B']]['draw_count'];hashes=[]
  for epoch in range(1,31):
   arrays=[]
   for h in ('B','AB'):
    folder=R/'runs'/members[h]
    a={key:np.load(folder/f'epoch_{epoch:02}_{key}.npy',allow_pickle=False) for key in ('samples','methods','augmentations')}
    arrays.append(a);entry=histories[members[h]][epoch-1]
    assert health.sha(folder/f'epoch_{epoch:02}_samples.npy')==entry['sample_sha256']
    assert health.sha(folder/f'epoch_{epoch:02}_augmentations.npy')==entry['augmentation_sha256']
    assert dict(Counter(map(str,a['methods'].tolist())))==entry['method_counts']
    np.testing.assert_array_equal(a['methods'],tables[h].method_id.to_numpy()[a['samples']])
   health.verify_arrays(arrays[0],arrays[1],tables,draws)
   hashes.append(dict(epoch=epoch,**{key+'_sha256':health.sha(R/'runs'/members['B']/f'epoch_{epoch:02}_{key}.npy') for key in ('samples','methods','augmentations')}))
  alignment=skip_alignment(flags[members['B']],flags[members['AB']],30*(draws//128))
  paired.append(dict(backbone=bb,seed=seed,B_run=members['B'],AB_run=members['AB'],common_snapshot_epochs_checked=30,
    samples_methods_augmentations_and_actual_mother_sequence_equal=True,hashes=hashes,**alignment))
  if not alignment['same_successful_update_stream']:asymmetries.append(dict(backbone=bb,seed=seed,**alignment))
 # Sealed paths must still equal the originals after every CPU/data check.
 for run in run_ids:
  print('Rebinding unchanged terminal artifacts '+run,flush=True,file=sys.stderr)
  health.verify_recipe(R,config['runs'][run],config);stable_bound(R/'runs'/run,bound[run])
 assert health.sha(config_path)==config_sha and health.sha(HEALTH_FILE)==HEALTH_SHA and not torch.cuda.is_initialized()
 status='PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS'
 if asymmetries:status='SELECTED_TERMINAL_CHECKS_PASS_AMP_PAIR_SKIP_ASYMMETRY_ALL5_SOURCE_REVIEW_REQUIRED'
 result=dict(utc=datetime.now(timezone.utc).isoformat(),status=status,source_only=True,
  selected_run_ids=run_ids,records=records,paired_streams=paired,observed_pair_update_asymmetries=asymmetries,
  selected_terminal_totals=dict(models=len(records),attempts=sum(x['snapshot_attempts'] for x in records),
    successful_updates=sum(x['snapshot_successful_updates'] for x in records),AMP_skips=sum(x['snapshot_AMP_skips'] for x in records)),
  inputs=dict(H_execution_sha256=config_sha,frozen_health_script_sha256=HEALTH_SHA,
    exporter_script_sha256=health.sha(Path(__file__)),qualification_gate_sha256=gate_hash),
  audit_seconds=time.monotonic()-start,CUDA_initialized=False,production_processes_restarted=False,
  target_export_qualification_claimed=False,whole_formal_training_complete_claimed=False,
  automatic_FP32_started=False,
  scope='Selected complete B/AB E30 formal AMP pairs only. Original158-artifact seals revalidated before inspection and rebound afterward. Final and source-best tensor contents/identity, complete Adam/scaler/RNG/attempt logs, thirty535-video saved source DEV tables with180 bisect AUC checks per run, and full source mother/sample/method/augmentation/skip positions verified. CPU RNG restored locally; saved CUDA RNG only inspected and hashed. This subset is not allfive-seed acceptance, an all30-model conclusion, a conditional uniformFP32 admission, a paired-ledger replacement, or a target gate.')
 health.ensure_new_report(destination)
 with destination.open('x',encoding='utf-8') as stream:
  json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=status,report=str(destination),audit_seconds=result['audit_seconds'])),flush=True)
 return result

def main(argv=None):
 args=parse_args(argv);audit(args.run_id,args.report)
if __name__=='__main__':main()
