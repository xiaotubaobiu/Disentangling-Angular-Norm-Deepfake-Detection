"""Synthetic protocol checks only; never call production main()."""
import unittest,pathlib,tempfile,importlib.util,json,random,copy
import numpy as np
import pandas as pd
import torch
HERE=pathlib.Path(__file__).parent;FILE=HERE.parent/'H_formal_checkpoint_health_20261024T2145.py'
def task(bb='clip_vit_b16'):return dict(route='H',phase='formal',backbone=bb,history='B',seed=7,epochs=45,draw_count=256,precision='AMP',recipe_frozen=True)
def fixture(bb='clip_vit_b16'):
 t=task(bb);dim={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}[bb];events=[]
 for i in range(4):
  skip=i==1;events.append(dict(epoch=i//2+1,attempt=i,success=not skip,skip=skip,lr=.001,loss=.2,gradient_finite=not skip,gradient_norm=None if skip else .3,scale_before=16. if i<2 else 8.,scale_after=8. if i>=1 else 16.))
 h=[dict(epoch=e,attempts=2,success=1 if e==1 else 2,skip=1 if e==1 else 0,draw_count=256,seconds=1.,dev={'macro_auc':.9}) for e in [1,2]]
 model={'classifier.weight':torch.zeros((2,dim)),'classifier.bias':torch.zeros(2)}
 state=dict(identity={'synthetic':True},epoch=2,attempts=4,scheduler_clock='attempt_count',history=h,optimizer={'param_groups':[{'params':[0],'lr':.001,'weight_decay':5e-4}], 'state':{0:{'step':torch.tensor(3.),'exp_avg':torch.zeros(2,dim),'exp_avg_sq':torch.zeros(2,dim)}}},model=model,best_model=copy.deepcopy(model),best_epoch=2,scaler={'scale':8.},python_rng=random.Random(1).getstate(),numpy_rng=np.random.RandomState(1).get_state(),torch_rng=torch.Generator().get_state(),cuda_rng=[torch.zeros(16,dtype=torch.uint8)],consecutive_skips=0)
 return t,state,events
class Protocol(unittest.TestCase):
 def setUp(self):
  self.assertTrue(FILE.exists(),'formal health script absent')
  spec=importlib.util.spec_from_file_location('formal_health',FILE);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
 def test_dynamic_formal_selection_ignores_development(self):
  config={'runs':{'H_formal_x':task(),'H_dev_x':dict(phase='development')}}
  q={'active':[{'run_id':'H_dev_x','pid':1},{'run_id':'H_formal_x','pid':999}]}
  self.assertEqual(self.m.select_formal_jobs(q,config),[{'run_id':'H_formal_x','pid':999}])
 def test_derived_budget_45epochs_2attempts_not_hardcoded(self):
  t,s,e=fixture();r=self.m.inspect_state(s,e,t,s['identity'],128)
  self.assertEqual(r['snapshot_attempts'],4);self.assertEqual(r['frozen_attempts_per_epoch'],2);self.assertEqual(r['registered_epochs'],45)
 def test_three_backbone_dimensions(self):
  for bb in ['clip_vit_b16','dinov2_vit_b14','efficientnet_b0']:
   t,s,e=fixture(bb);self.m.inspect_state(s,e,t,s['identity'],128)
 def test_wrong_backbone_dimension_rejected(self):
  t,s,e=fixture();t['backbone']='dinov2_vit_b14'
  with self.assertRaises(AssertionError):self.m.inspect_state(s,e,t,s['identity'],128)
 def test_skip_recovery_and_optimizer_clock(self):
  t,s,e=fixture();r=self.m.inspect_state(s,e,t,s['identity'],128)
  self.assertEqual(r['snapshot_AMP_skips'],1);self.assertEqual(r['snapshot_successful_updates'],3);self.assertEqual(r['skip_recovery_within_snapshot'][0]['next_success'],2)
 def test_bad_attempt_budget_rejected(self):
  t,s,e=fixture();s['attempts']=5
  with self.assertRaises(AssertionError):self.m.inspect_state(s,e,t,s['identity'],128)
 def test_bad_Adam_success_clock_rejected(self):
  t,s,e=fixture();s['optimizer']['state'][0]['step']=torch.tensor(4.)
  with self.assertRaises(AssertionError):self.m.inspect_state(s,e,t,s['identity'],128)
 def test_successful_nonfinite_gradient_rejected(self):
  t,s,e=fixture();e[0]['gradient_finite']=False
  with self.assertRaises(AssertionError):self.m.inspect_state(s,e,t,s['identity'],128)
 def test_scaler_chain_rejected(self):
  t,s,e=fixture();e[2]['scale_before']=32.
  with self.assertRaises(AssertionError):self.m.inspect_state(s,e,t,s['identity'],128)
 def test_bad_trailing_skip_counter_rejected(self):
  t,s,e=fixture();s['consecutive_skips']=1
  with self.assertRaises(AssertionError):self.m.inspect_state(s,e,t,s['identity'],128)
 def test_existing_report_rejected(self):
  with tempfile.TemporaryDirectory(dir=HERE) as d:
   p=pathlib.Path(d)/'report.json';p.write_text('preserve')
   with self.assertRaises(FileExistsError):self.m.ensure_new_report(p)
   self.assertEqual(p.read_text(),'preserve')
 def test_paired_stream_and_actual_mother_order(self):
  samples=np.tile([0,1],128);methods=np.array(['real','Deepfakes']*128);aug=np.tile([0.,1.,1.],(256,1));arrays={'samples':samples,'methods':methods,'augmentations':aug};tables={'B':pd.DataFrame({'mother_id':['m0','m1']}),'AB':pd.DataFrame({'mother_id':['m0','m1']})}
  self.m.verify_arrays(arrays,arrays,tables,256)
  tables['AB']=pd.DataFrame({'mother_id':['m1','m0']})
  with self.assertRaises(AssertionError):self.m.verify_arrays(arrays,arrays,tables,256)
 def test_paired_method_permutation_rejected(self):
  a={'samples':np.tile([0,1],128),'methods':np.array(['real','Deepfakes']*128),'augmentations':np.tile([0.,1.,1.],(256,1))};b={k:v.copy() for k,v in a.items()};b['methods']=b['methods'][::-1];tables={h:pd.DataFrame({'mother_id':['m0','m1']}) for h in ['B','AB']}
  with self.assertRaises(AssertionError):self.m.verify_arrays(a,b,tables,256)

 def test_formal_without_frozen_recipe_rejected(self):
  t=task();t.pop('recipe_frozen');config={'runs':{'H_formal_x':t}}
  with self.assertRaises(AssertionError):self.m.select_formal_jobs({'active':[{'run_id':'H_formal_x','pid':5}]},config)


 def recipe_fixture(self,root):
  (root/'configs').mkdir();(root/'data/H').mkdir(parents=True);freeze=root/'data/H/freeze.json';freeze.write_text('{"synthetic":true}')
  t=task();t.update(lr=1e-5,H_data_freeze_sha256=self.m.sha(freeze),h_code_bundle_sha256='hc')
  c=dict(H_data_freeze_sha256=t['H_data_freeze_sha256'],h_code_bundle_sha256='hc')
  r=dict(H_data_freeze_sha256=c['H_data_freeze_sha256'],h_code_bundle_sha256='hc',target_scores_used=False,recipes={t['backbone']:dict(lr=t['lr'],epochs=t['epochs'],shared_B_AB_recipe=True,selection_source_only=True)})
  p=root/'configs/H_frozen_recipes.json';p.write_text(json.dumps(r));t['source_recipe_sha256']=self.m.sha(p)
  return t,c,r,p
 def test_recipe_physical_SHA_and_protocol_chain(self):
  self.assertTrue(hasattr(self.m,'verify_recipe'),'physical frozen recipe chain check absent')
  with tempfile.TemporaryDirectory(dir=HERE) as d:
   t,c,r,p=self.recipe_fixture(pathlib.Path(d));self.m.verify_recipe(pathlib.Path(d),t,c)
 def test_recipe_chain_tampering_rejected(self):
  self.assertTrue(hasattr(self.m,'verify_recipe'),'physical frozen recipe chain check absent')
  for case in ['bytes','epochs','bundle','freeze']:
   with self.subTest(case=case),tempfile.TemporaryDirectory(dir=HERE) as d:
    root=pathlib.Path(d);t,c,r,p=self.recipe_fixture(root)
    if case=='bytes':p.write_text(p.read_text()+' ')
    elif case=='epochs':r['recipes'][t['backbone']]['epochs']+=1;p.write_text(json.dumps(r));t['source_recipe_sha256']=self.m.sha(p)
    elif case=='bundle':r['h_code_bundle_sha256']='changed';p.write_text(json.dumps(r));t['source_recipe_sha256']=self.m.sha(p)
    else:(root/'data/H/freeze.json').write_text('changed')
    with self.assertRaises(AssertionError):self.m.verify_recipe(root,t,c)

if __name__=='__main__':unittest.main(verbosity=2)
