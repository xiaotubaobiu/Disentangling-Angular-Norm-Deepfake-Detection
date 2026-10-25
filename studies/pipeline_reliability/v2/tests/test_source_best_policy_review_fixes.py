import copy,hashlib,json,pytest
from reliability_v2 import source_best_policy as s
from test_source_best_policy import objects,registry

def proper(route):
 t,run,h,c,cp=objects(route)
 if route=='H':t['draw_count']=1280
 identity=dict(c['identity']);identity['config']=hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest();c['identity']=identity;cp['identity']=copy.deepcopy(identity)
 return t,run,h,c,cp

@pytest.mark.parametrize('change',['D_seed','H_seed','H_history','D_lr','H_precision'])
def test_matching_foreign_task_receipt_metadata_is_rejected(change):
 route=change[0];t,run,h,c,cp=proper(route);reg=registry() if route=='H' else None
 if change.endswith('seed'):t['seed']=42;run=run.replace('seed7','seed42')
 if change=='H_history':t['history']='AB';run=run.replace('_B_','_AB_')
 if change=='D_lr':t['lr']=3e-5
 if change=='H_precision':
  t['precision']='FP32';run+='_FP32';reg['effective_runs']['clip_vit_b16']['precision']='FP32';reg['effective_runs']['clip_vit_b16']['run_ids']=[x+'_FP32' for x in reg['effective_runs']['clip_vit_b16']['run_ids']]
 with pytest.raises(ValueError):s.metadata_endpoint(t,run,h,c,cp,H_registry=reg)

@pytest.mark.parametrize('uniform',[False,True])
def test_frozen_H_exposure_rejects_arithmetically_consistent_changed_history(uniform):
 t,run,h,c,cp=proper('H')
 for row in h if uniform else h[:1]:row['attempts']-=1;row['success']-=1;row['draw_count']-=128
 for field in ['attempts','success','skip']:c[field]=sum(x[field] for x in h)
 with pytest.raises(ValueError):s.metadata_endpoint(t,run,h,c,cp,H_registry=registry())

@pytest.mark.parametrize('bad',[None,True,0,1279,1280.0])
def test_H_task_draw_basis_must_be_registered_integer_multiple128(bad):
 t,run,h,c,cp=proper('H');t['draw_count']=bad;c['identity']['config']=hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest();cp['identity']=copy.deepcopy(c['identity'])
 with pytest.raises(ValueError):s.metadata_endpoint(t,run,h,c,cp,H_registry=registry())
