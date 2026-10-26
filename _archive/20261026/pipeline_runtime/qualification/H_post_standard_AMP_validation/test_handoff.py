import unittest,pathlib,tempfile,importlib.util,datetime,json,subprocess,sys,ast,os
from fixtures import make,sha,js,edit_budget,rebind_run,G,NAMES
HERE=pathlib.Path(__file__).parent;FILE=HERE.parent/'continue_H_formal_AMP.py'
class Admission(unittest.TestCase):
 def setUp(self):
  self.assertTrue(FILE.exists(),'finite AMP handoff wrapper not implemented')
  spec=importlib.util.spec_from_file_location('handoff',FILE);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  (HERE/'work').mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=HERE/'work');self.f=make(pathlib.Path(self.tmp.name),self.m)
 def tearDown(self):
  if hasattr(self,'tmp'):self.tmp.cleanup()
 def check(self):return self.m.preflight(**self.f)
 def test_valid_all20_and_no_DINO_extension(self):
  a=self.check();self.assertEqual(a['accepted_standard_models'],20);self.assertEqual(a['decisions']['dinov2_vit_b14']['lr'],1e-5);self.assertEqual(a['decisions']['dinov2_vit_b14']['extension'],None)
  self.assertEqual(a['capacity']['required_bytes'],111*G)
 def test_original_owner_must_exit(self):
  (self.f['proc_root']/'1993186').mkdir()
  with self.assertRaises(ValueError):self.check()
 def test_original_success_progress_required(self):
  p=self.f['root']/'reports/H_source_and_standard_development_progress_20261024T1510.json';a=json.loads(p.read_text());a['status']='RUNNING';js(p,a)
  with self.assertRaises(ValueError):self.check()
 def test_status_active_must_be_empty(self):
  js(self.f['root']/'H_queue_status.json',dict(stage='H_STANDARD_DEVELOPMENT_COMPLETE_CAPACITY_REVIEW_BEFORE_CONDITIONALS',active=[{'pid':99}]))
  with self.assertRaises(ValueError):self.check()
 def test_residual_worker_rejected_even_when_status_empty(self):
  p=self.f['proc_root']/'998';p.mkdir();(p/'cmdline').write_bytes(b'python\0-m\0reliability_v2.worker_h\0--config\0'+str(self.f['root']/'configs/H_execution.json').encode()+b'\0')
  with self.assertRaises(ValueError):self.check()
 def test_D_gpu_stage_required(self):
  js(self.f['root']/'queue_status.json',dict(stage='RUNNING',active=[]))
  with self.assertRaises(ValueError):self.check()
 def test_missing_terminal_audit_rejected(self):
  (self.f['root']/'reports'/NAMES[-1]).unlink()
  with self.assertRaises(ValueError):self.check()
 def test_changed_terminal_receipt_rejected(self):
  p=self.f['root']/'reports'/NAMES[0];p.write_text(p.read_text()+' ')
  with self.assertRaises(ValueError):self.check()
 def test_changed_complete_rejected(self):
  p=next((self.f['root']/'runs').glob('*/complete.json'));p.write_text(p.read_text()+' ')
  with self.assertRaises(ValueError):self.check()
 def test_changed_state_rejected(self):
  p=next((self.f['root']/'runs').glob('*/state.pt'));p.write_text('tampered')
  with self.assertRaises(ValueError):self.check()
 def test_changed_sealed_auxiliary_rejected(self):
  p=next((self.f['root']/'runs').glob('*/epoch_01_methods.npy'));p.write_text('tampered')
  with self.assertRaises(ValueError):self.check()
 def test_incomplete_independent_coverage_rejected(self):
  p=self.f['root']/'reports'/NAMES[0];a=json.loads(p.read_text());a['records'].pop();js(p,a);self.f['audit_pins'][NAMES[0]]=sha(p)
  with self.assertRaises(ValueError):self.check()
 def test_budget_SHA_required(self):
  self.f['budget_sha']='0'*64
  with self.assertRaises(ValueError):self.check()
 def test_stale_budget_rejected(self):
  self.f['now']+=datetime.timedelta(seconds=3601)
  with self.assertRaises(ValueError):self.check()
 def test_future_budget_rejected(self):
  self.f['now']-=datetime.timedelta(seconds=1)
  with self.assertRaises(ValueError):self.check()
 def test_negative_estimate_rejected(self):
  edit_budget(self.f,lambda a:a['remaining_components'].update(H_features=-1))
  with self.assertRaises(ValueError):self.check()
 def test_reserve_floor(self):
  edit_budget(self.f,lambda a:a.update(reserve_bytes=49*G))
  with self.assertRaises(ValueError):self.check()
 def test_temporary_floor(self):
  edit_budget(self.f,lambda a:a.update(temporary_bytes=15*G))
  with self.assertRaises(ValueError):self.check()
 def test_compact_unaccepted_savings_not_admitted(self):
  edit_budget(self.f,lambda a:a.update(compact_unvalidated_storage_savings_counted=True))
  with self.assertRaises(ValueError):self.check()
 def test_source_best_budget_not_admitted(self):
  edit_budget(self.f,lambda a:a.update(source_best_active_budget=1))
  with self.assertRaises(ValueError):self.check()
 def test_live_free_overrides_budget_snapshot(self):
  self.f['free_bytes']=111*G
  with self.assertRaises(ValueError):self.check()
 def test_budget_components_must_equal_total(self):
  edit_budget(self.f,lambda a:a.update(normal_remaining_write_bytes=1))
  with self.assertRaises(ValueError):self.check()
 def test_DINO_extension_trigger_stops(self):
  p=next((self.f['root']/'runs').glob('H_dev_dinov2*B_lr1e-05_seed1007_e30/history.json'));h=json.loads(p.read_text())
  for r in h[-5:]:r['dev']['macro_auc']=.91
  js(p,h);rebind_run(self.f,p.parent.name)
  with self.assertRaises(ValueError):self.check()
 def test_original_scientific_block_exact(self):
  block=self.m.scientific_block(self.m.SCIENCE);tree=ast.parse((self.m.SCIENCE/'queue_h.py').read_text());run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run' and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipes' for t in x.targets) for x in n.body));start=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipes' for t in n.targets));end=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='batch' and any(isinstance(a,ast.Constant) and a.value=='H_FORMAL_AMP_TRAINING' for a in n.value.args));self.assertEqual(ast.dump(block,include_attributes=False),ast.dump(ast.Module(body=run.body[start:end+1],type_ignores=[]),include_attributes=False))
 def test_concurrent_original_lock_refuses(self):
  lock=self.f['root']/'H_pipeline.lock';code="import fcntl,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); print('LOCKED',flush=True); sys.stdin.read()"
  p=subprocess.Popen([sys.executable,'-c',code,str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
  try:
   self.assertEqual(p.stdout.readline().strip(),'LOCKED')
   with self.assertRaises(BlockingIOError):
    with self.m.original_lock(self.m.SCIENCE)(lock):pass
  finally:p.communicate('release',timeout=5)
  self.assertEqual(p.returncode,0)
 def test_original_block_ends_after30AMP_and_registry_disallows_target(self):
  admission=self.check();calls=[];api=self.m.pure_api(self.m.SCIENCE);root=self.f['root'];config=json.loads((root/'configs/H_execution.json').read_text())
  class Q:
   def __init__(self):self.root=root;self.config=config
   def histories(self,ids):return [json.loads((root/'runs'/r/'history.json').read_text()) for r in ids]
   def _register(self,tasks):self.config['runs'].update(tasks)
   def batch(self,ids,stage):calls.append((ids,stage))
  q=Q();out=self.m.run_original_block(q,api,admission,check_before_batch=lambda:None)
  self.assertEqual(len(calls),1);self.assertEqual(len(calls[0][0]),30);self.assertEqual(calls[0][1],'H_FORMAL_AMP_TRAINING')
  registry=json.loads((root/'H_run_registry.json').read_text());self.assertFalse(registry['target_export_allowed']);self.assertEqual(registry['effective_runs'],{});self.assertEqual(len(registry['formal_run_ids']),30)
  self.assertFalse(any((root/'qualification').glob('H_target_export*')) if (root/'qualification').exists() else False)

 def test_admission_snapshot_rejects_postcheck_state_change(self):
  a=self.check();p=next((self.f['root']/'runs').glob('*/state.pt'));p.write_text('changed after gate')
  self.assertTrue(hasattr(self.m,'verify_snapshot'),'pre-science immutable snapshot guard missing')
  with self.assertRaises(ValueError):self.m.verify_snapshot(a)

if __name__=='__main__':unittest.main(verbosity=2)
