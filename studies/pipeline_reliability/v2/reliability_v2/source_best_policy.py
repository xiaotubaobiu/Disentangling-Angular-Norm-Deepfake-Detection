"""Pure source-best M metadata policy. No I/O, tensor load or inference authority.
Actual sealed checkpoints, original heads, data/roles, code/environment, ownership
and global H target proof must still be verified by the separate cache context.
"""
import copy,hashlib,json,math,re
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SEEDS=(7,42,123,2026,3407)
D_ROLES=('train','source_cal','source_test','target_full','target_cal','target_test')
H_ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST')
IDENTITY_FIELDS={'config','train_manifest','dev_manifest','initialization','environment','code','qualification'}
def require(ok,reason):
 if not ok:raise ValueError(reason)
def integer(x,minimum=0):
 require(isinstance(x,int) and not isinstance(x,bool) and x>=minimum,'integer counter/budget required');return x
def digest_string(x):
 require(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None,'SHA256 required');return x
def _validated_history(history):
 require(isinstance(history,list) and len(history)>0,'complete source DEV history required')
 for i,row in enumerate(history,1):
  require(integer(row.get('epoch'),1)==i,'contiguous source epochs required')
  a=integer(row.get('attempts'),1);u=integer(row.get('success'));sk=integer(row.get('skip'));draws=integer(row.get('draw_count'),128)
  require(a==u+sk and draws==a*128,'attempt/success/skip/draw budget drift')
  score=row.get('dev',{}).get('macro_auc');require(isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and 0<=score<=1,'finite source DEV macro AUC required')
 return history

def select_source_best(history):
 history=_validated_history(history)
 # Same strict improvement as original trainer; ties retain first epoch.
 best=max(range(len(history)),key=lambda i:history[i]['dev']['macro_auc'])
 prefix=history[:best+1]
 return dict(epoch=best+1,source_macro_auc=history[best]['dev']['macro_auc'],attempts=sum(x['attempts'] for x in prefix),successful_updates=sum(x['success'] for x in prefix),AMP_skips=sum(x['skip'] for x in prefix),draws=sum(x['draw_count'] for x in prefix))

def _H_effective(registry,task,run_id):
 require(isinstance(registry,dict) and registry.get('schema')=='H_RUN_REGISTRY_V2' and registry.get('status')=='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING' and registry.get('formal_model_count')==30 and registry.get('target_export_allowed') is True,'final all3 effective H registry prerequisite')
 formal=[f'H_formal_{b}_{h}_seed{s}' for b in BACKBONES for s in SEEDS for h in ('B','AB')]
 require(len(registry.get('formal_run_ids',[]))==30 and set(registry['formal_run_ids'])==set(formal),'formal30 registry changed')
 require(set(registry.get('effective_runs',{}))==set(BACKBONES),'all3 effective backbones required')
 for b,item in registry['effective_runs'].items():
  precision=item.get('precision');require(precision in ('AMP','FP32'),'effective precision invalid')
  expected=[f'H_formal_{b}_{h}_seed{s}'+('_FP32' if precision=='FP32' else '') for s in SEEDS for h in ('B','AB')]
  require(item.get('run_ids')==expected,'effective five-pair inventory/order changed')
 item=registry['effective_runs'][task['backbone']]
 require(item['precision']==task['precision'] and run_id in item['run_ids'],'H endpoint superseded by effective precision')
 return hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def metadata_endpoint(task,run_id,history,completion,checkpoint_metadata,*,H_registry=None):
 require(isinstance(task,dict) and task.get('phase')=='formal' and task.get('recipe_frozen') is True and task.get('selection')=='frozen_last','existing frozen formal task only')
 route=task.get('route');bone=task.get('backbone');seed=task.get('seed');epochs=integer(task.get('epochs'),1);precision=task.get('precision')
 require(route in ('D','H') and bone in BACKBONES and integer(seed,1) in SEEDS and precision in ('AMP','FP32'),'registered M model family/seed/precision required')
 if route=='D':
  require(precision=='AMP','selected D formal endpoints are AMP');expected=f'D_formal_{bone}_seed{seed}'
 else:
  require(task.get('history') in ('B','AB'),'registered H history required');expected=f"H_formal_{bone}_{task['history']}_seed{seed}"+('_FP32' if precision=='FP32' else '')
 require(run_id==expected,'canonical formal endpoint ID required')
 best=select_source_best(history);require(len(history)==epochs,'full registered source history required')
 if route=='H':
  frozen_draws=integer(task.get('draw_count'),128);require(frozen_draws%128==0,'registered H draw budget multiple128 required')
  require(all(row['draw_count']==frozen_draws and row['attempts']==frozen_draws//128 for row in history),'H per-epoch exposure differs from frozen task')
 require(completion.get('status')=='TRAINING_COMPLETE' and integer(completion.get('epochs'),1)==epochs and completion.get('selection')=='frozen_last' and completion.get('target_used') is False,'complete source-only frozen-last formal parent required')
 for key,field in [('attempts','attempts'),('success','success'),('skip','skip')]:require(integer(completion.get(key))==sum(x[field] for x in history),'whole training counts drift')
 require(integer(completion.get('source_best_epoch'),1)==best['epoch'] and integer(checkpoint_metadata.get('epoch'),1)==best['epoch'],'wrong source-best/tie epoch')
 identity=completion.get('identity');require(isinstance(identity,dict) and IDENTITY_FIELDS<=set(identity) and checkpoint_metadata.get('identity')==identity,'checkpoint identity differs from formal parent')
 for key in IDENTITY_FIELDS:digest_string(identity[key])
 require(identity['config']==hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),'metadata identity is foreign to the frozen task')
 checkpoint_sha=digest_string(completion.get('source_best_sha256'));digest_string(completion.get('final_sha256'))
 registry_hash=_H_effective(H_registry,task,run_id) if route=='H' else None
 return dict(schema='M_SOURCE_BEST_METADATA_ONLY_V1',status='POLICY_PASS_SEALED_FILE_TENSOR_CONTEXT_PENDING',endpoint_id='M_source_best_'+run_id,parent_run_id=run_id,route=route,backbone=bone,seed=seed,history=task.get('history'),effective_precision=precision,checkpoint_selection='source_best',selection_rule='earliest epoch attaining maximum source DEV macro AUC; strict improvement rule',best=best,registered_terminal_epochs=epochs,recorded_checkpoint_sha256=checkpoint_sha,training_identity=copy.deepcopy(identity),H_registry_structural_sha256=registry_hash,roles=list(D_ROLES if route=='D' else H_ROLES),target_inference_authorized=False,tensor_and_head_validation_complete=False,equal_update_primary_claim_allowed=False,scope='Pure metadata policy only. Caller must verify original sealed files/tensors/head, actual data/split/code/environment and resource/global-target permissions; no inference authority or scientific results.')
