import copy,math,pytest
from reliability_v2 import source_best_policy as s

def hist():
 return [dict(epoch=i+1,attempts=10,success=10-(i==0),skip=int(i==0),draw_count=1280,dev={'macro_auc':auc}) for i,auc in enumerate([.5,.8,.8,.6])]

def objects(route='D'):
 task=dict(route=route,phase='formal',recipe_frozen=True,backbone='clip_vit_b16',seed=7,epochs=4,precision='AMP',selection='frozen_last')
 if route=='H':task['history']='B'
 identity={k:'a'*64 for k in ['config','train_manifest','dev_manifest','initialization','environment','code','qualification']}
 completion=dict(status='TRAINING_COMPLETE',identity=identity,epochs=4,attempts=40,success=39,skip=1,selection='frozen_last',target_used=False,source_best_epoch=2,source_best_sha256='b'*64,final_sha256='c'*64)
 checkpoint_meta=dict(epoch=2,identity=identity)
 run='D_formal_clip_vit_b16_seed7' if route=='D' else 'H_formal_clip_vit_b16_B_seed7'
 return task,run,hist(),completion,checkpoint_meta

def test_earliest_strict_source_max_and_actual_prefix_counts():
 r=s.select_source_best(hist());assert r==dict(epoch=2,source_macro_auc=.8,attempts=20,successful_updates=19,AMP_skips=1,draws=2560)

@pytest.mark.parametrize('change',['gap','nan','infinity','budget','negative','bool','draws'])
def test_bad_source_history_rejected(change):
 h=hist()
 if change=='gap':h[1]['epoch']=3
 if change=='nan':h[1]['dev']['macro_auc']=float('nan')
 if change=='infinity':h[1]['dev']['macro_auc']=float('inf')
 if change=='budget':h[1]['success']=9
 if change=='negative':h[1]['skip']=-1
 if change=='bool':h[1]['attempts']=True
 if change=='draws':h[1]['draw_count']=1279
 with pytest.raises(ValueError):s.select_source_best(h)

def test_D_metadata_binds_best_in_separate_namespace_and_no_production_claim():
 d=s.metadata_endpoint(*objects());assert d['checkpoint_selection']=='source_best' and d['best']['epoch']==2
 assert d['endpoint_id'].startswith('M_source_best_') and len(d['roles'])==6
 assert d['target_inference_authorized'] is False and d['tensor_and_head_validation_complete'] is False

@pytest.mark.parametrize('change',['development','unfrozen','seed','foreign_run','epoch','score_epoch','checkpoint_identity','checkpoint_epoch','checkpoint_sha','total','target'])
def test_foreign_incomplete_or_wrong_source_selection_rejected(change):
 task,run,h,c,cp=objects()
 if change=='development':task['phase']='development'
 if change=='unfrozen':task['recipe_frozen']=False
 if change=='seed':task['seed']=99
 if change=='foreign_run':run='D_formal_clip_vit_b16_seed42'
 if change=='epoch':c['epochs']=3
 if change=='score_epoch':c['source_best_epoch']=3
 if change=='checkpoint_identity':cp=copy.deepcopy(cp);cp['identity']['code']='d'*64
 if change=='checkpoint_epoch':cp['epoch']=3
 if change=='checkpoint_sha':c['source_best_sha256']='bad'
 if change=='total':c['attempts']=41
 if change=='target':c['target_used']=True
 with pytest.raises(ValueError):s.metadata_endpoint(task,run,h,c,cp)

def registry():
 bones=s.BACKBONES;seeds=s.SEEDS;histories=('B','AB')
 return dict(schema='H_RUN_REGISTRY_V2',status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING',formal_model_count=30,target_export_allowed=True,formal_run_ids=[f'H_formal_{b}_{h}_seed{i}' for b in bones for i in seeds for h in histories],effective_runs={b:dict(precision='AMP',run_ids=[f'H_formal_{b}_{h}_seed{i}' for i in seeds for h in histories]) for b in bones})

def test_H_requires_final_all3_effective_precision():
 with pytest.raises(ValueError):s.metadata_endpoint(*objects('H'))
 d=s.metadata_endpoint(*objects('H'),H_registry=registry());assert d['roles']==['B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST']
 assert d['equal_update_primary_claim_allowed'] is False

@pytest.mark.parametrize('change',['gate','bone','precision','ids'])
def test_H_pending_or_superseded_precision_rejected(change):
 r=registry()
 if change=='gate':r['target_export_allowed']=False
 if change=='bone':r['effective_runs'].pop('efficientnet_b0')
 if change=='precision':
  r['effective_runs']['clip_vit_b16']['precision']='FP32';r['effective_runs']['clip_vit_b16']['run_ids']=[x+'_FP32' for x in r['effective_runs']['clip_vit_b16']['run_ids']]
 if change=='ids':r['effective_runs']['clip_vit_b16']['run_ids'].reverse()
 with pytest.raises(ValueError):s.metadata_endpoint(*objects('H'),H_registry=r)
