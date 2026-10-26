"""Synthetic terminal protocol tests; never execute the production audit."""
from pathlib import Path
import unittest,tempfile,importlib.util,json,copy,ast
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).parent
FILE=HERE.parent/'audit_H_formal_terminal.py'
def task(h='B',seed=7):
 return dict(route='H',phase='formal',history=h,seed=seed,precision='AMP',backbone='clip_vit_b16',epochs=30,recipe_frozen=True,draw_count=256,lr=1e-5,microbatch=32,H_data_freeze_sha256='f',h_code_bundle_sha256='c',source_recipe_sha256='r')
def cfg():
 return {'runs':{f'H_formal_clip_vit_b16_{h}_seed7':task(h) for h in ('B','AB')}}
def dev_fixture():
 expected=pd.DataFrame({'video_uid':['real','d','f','s','n'],'method_id':['real','Deepfakes','Face2Face','FaceSwap','NeuralTextures'],'label':[0,1,1,1,1]})
 table=expected.copy();table['score']=[0.,1.,1.,1.,1.]
 metrics=dict(pooled_auc=1.,method_auc={m:1. for m in expected.method_id[1:]},macro_auc=1.)
 return expected,table,metrics
class Protocol(unittest.TestCase):
 def setUp(self):
  self.assertTrue(FILE.exists(),'terminal auditor entry point absent')
  spec=importlib.util.spec_from_file_location('terminal',FILE);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
 def test_repeated_run_ids_and_required_report(self):
  args=self.m.parse_args(['--run-id','a','--run-id','b','--report','x.json'])
  self.assertEqual(args.run_id,['a','b'])
  with self.assertRaises(SystemExit):self.m.parse_args(['--run-id','a'])
 def test_complete_pair_subset_accepted(self):
  c=cfg();groups=self.m.select_pairs(list(c['runs']),c)
  self.assertEqual(set(groups),{('clip_vit_b16',7)})
 def test_duplicate_unsafe_or_unregistered_run_rejected(self):
  c=cfg();runs=list(c['runs'])
  for value in [[runs[0],runs[0]],[runs[0],'../escape'],[runs[0],'H_formal_missing']]:
   with self.subTest(value=value),self.assertRaises((AssertionError,ValueError)):self.m.select_pairs(value,c)
 def test_incomplete_pair_rejected(self):
  with self.assertRaises((AssertionError,ValueError)):self.m.select_pairs(list(cfg()['runs'])[:1],cfg())
 def test_wrong_phase_precision_budget_or_recipe_rejected(self):
  for key,value in [('phase','development'),('precision','FP32'),('epochs',45),('recipe_frozen',False)]:
   c=cfg();c['runs'][list(c['runs'])[0]][key]=value
   with self.subTest(key=key),self.assertRaises((AssertionError,ValueError)):self.m.select_pairs(list(c['runs']),c)
 def test_pair_recipe_fields_must_match(self):
  c=cfg();c['runs'][list(c['runs'])[0]]['lr']=3e-5
  with self.assertRaises((AssertionError,ValueError)):self.m.select_pairs(list(c['runs']),c)
 def ready_fixture(self,root):
  run=list(cfg()['runs'])[0];folder=root/'runs'/run;folder.mkdir(parents=True)
  names=self.m.artifact_names(30)
  for name in names:(folder/name).write_text('{}')
  complete=dict(status='TRAINING_COMPLETE',epochs=30,attempts=60,success=59,skip=1,selection='frozen_last',target_used=False)
  (folder/'complete.json').write_text(json.dumps(complete))
  seal=dict(status='SEALED',run_id=run,artifact_sha256={n:self.m.health.sha(folder/n) for n in names})
  (folder/'H_completion_seal.json').write_text(json.dumps(seal))
  return run,folder,names
 def test_158_artifact_terminal_preflight(self):
  with tempfile.TemporaryDirectory(dir=HERE) as d:
   root=Path(d);run,folder,names=self.ready_fixture(root)
   self.assertEqual(len(names),158);self.m.check_terminal_ready(root,run,task(),names)
 def test_missing_seal_missing_artifact_and_E29_rejected(self):
  for case in ['seal','artifact','epoch','inventory']:
   with self.subTest(case=case),tempfile.TemporaryDirectory(dir=HERE) as d:
    root=Path(d);run,folder,names=self.ready_fixture(root)
    if case=='seal':(folder/'H_completion_seal.json').unlink()
    elif case=='artifact':(folder/'epoch_30_methods.npy').unlink()
    elif case=='epoch':
     p=folder/'complete.json';c=json.loads(p.read_text());c['epochs']=29;p.write_text(json.dumps(c))
    else:
     p=folder/'H_completion_seal.json';c=json.loads(p.read_text());c['artifact_sha256'].pop(names[-1]);p.write_text(json.dumps(c))
    with self.assertRaises((AssertionError,FileNotFoundError)):self.m.check_terminal_ready(root,run,task(),names)
 def test_report_nooverwrite_and_path_safety(self):
  with tempfile.TemporaryDirectory(dir=HERE) as d:
   root=Path(d);(root/'reports').mkdir();p=self.m.report_path(root,'new.json')
   self.assertEqual(p,root/'reports/new.json');p.write_text('preserve')
   with self.assertRaises(FileExistsError):self.m.report_path(root,'new.json')
   self.assertEqual(p.read_text(),'preserve')
   for name in ['../escape.json',str(root/'outside.json'),'bad.txt']:
    with self.assertRaises((AssertionError,ValueError)):self.m.report_path(root,name)
 def weights(self):
  identity={'x':'same'};model={'w':torch.tensor([1.,2.]),'i':torch.tensor([1])}
  state=dict(identity=identity,epoch=30,model=model,best_model=copy.deepcopy(model),best_epoch=8,best_score=.9,history=[dict(epoch=e,dev={'macro_auc':.9 if e>=8 else .8}) for e in range(1,31)])
  final=dict(identity=identity,epoch=30,model=copy.deepcopy(model));best=dict(identity=identity,epoch=8,model=copy.deepcopy(model))
  return state,final,best,identity,{'source_best_epoch':8}
 def test_final_and_best_full_tensor_identity(self):
  self.m.verify_completed_weights(*self.weights())
 def test_final_or_best_mismatch_identity_epoch_tensor_dtype_rejected(self):
  for case in ['identity','epoch','value','dtype','best_identity','best_epoch','best_score']:
   with self.subTest(case=case):
    s,f,b,i,r=self.weights()
    if case=='identity':f['identity']={}
    elif case=='epoch':f['epoch']=29
    elif case=='value':f['model']['w'][0]=0
    elif case=='dtype':f['model']['w']=f['model']['w'].double()
    elif case=='best_identity':b['identity']={}
    elif case=='best_epoch':b['epoch']=9
    else:s['best_score']=.8
    with self.assertRaises(AssertionError):self.m.verify_completed_weights(s,f,b,i,r)
 def test_independent_auc_ties_and_numerical_mismatch(self):
  self.assertEqual(self.m.independent_auc([0.,0.,1.,1.],[0,1,0,1]),.5)
  expected,table,metrics=dev_fixture();self.assertEqual(len(self.m.verify_dev_table(table,expected,metrics)),6)
  metrics['pooled_auc']=.99
  with self.assertRaises(AssertionError):self.m.verify_dev_table(table,expected,metrics)
 def test_DEV_cohort_schema_nonfinite_missing_method_rejected(self):
  for case in ['cohort','schema','nonfinite','method','label']:
   with self.subTest(case=case):
    expected,table,metrics=dev_fixture()
    if case=='cohort':table.loc[0,'video_uid']='alien'
    elif case=='schema':table['extra']=1
    elif case=='nonfinite':table.loc[0,'score']=np.nan
    elif case=='method':metrics['method_auc'].pop('FaceSwap')
    else:table.loc[0,'label']=2
    with self.assertRaises((AssertionError,ValueError)):self.m.verify_dev_table(table,expected,metrics)
 def test_pair_skip_position_asymmetry_is_explicit(self):
  r=self.m.skip_alignment([False,True,False],[True,False,False],3)
  self.assertFalse(r['same_successful_update_stream']);self.assertEqual(r['different_attempt_indices'],[0,1]);self.assertEqual(r['B_skip_attempts'],[1]);self.assertEqual(r['AB_skip_attempts'],[0])
  self.assertTrue(self.m.skip_alignment([False]*3,[False]*3,3)['same_successful_update_stream'])
 def test_pair_stream_reuses_frozen_actual_mother_check(self):
  a=dict(samples=np.tile([0,1],128),methods=np.array(['real','Deepfakes']*128),augmentations=np.tile([0.,1.,1.],(256,1)))
  tables={h:pd.DataFrame({'mother_id':['m0','m1']}) for h in ['B','AB']}
  self.m.health.verify_arrays(a,a,tables,256)
  tables['AB']=tables['AB'].iloc[::-1]
  with self.assertRaises(AssertionError):self.m.health.verify_arrays(a,a,tables,256)
 def test_pinned_health_file_and_original_verifier_order_in_main(self):
  self.assertEqual(self.m.health.sha(self.m.HEALTH_FILE),self.m.HEALTH_SHA)
  tree=ast.parse(FILE.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
  calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(main) if isinstance(n,ast.Call)]
  verifier=[n for n,f in calls if f=='worker_h.verify_run_completion'];snap=[n for n,f in calls if f=='health.snapshot']
  self.assertTrue(verifier and snap);self.assertLess(min(verifier),min(snap))
  forbidden=['finalize_completion_seal','exclusive_file','subprocess','kill','train(','target_metrics','cache_h']
  source=FILE.read_text()
  for name in forbidden:self.assertNotIn(name,source)
 def test_sealed_artifact_mutation_during_audit_rejected(self):
  with tempfile.TemporaryDirectory(dir=HERE) as d:
   root=Path(d);run,folder,names=self.ready_fixture(root)
   bound=self.m.check_terminal_ready(root,run,task(),names)
   self.m.stable_bound(folder,bound)
   (folder/'epoch_30_updates.jsonl').write_text('changed')
   with self.assertRaises(AssertionError):self.m.stable_bound(folder,bound)

if __name__=='__main__':unittest.main(verbosity=2)

