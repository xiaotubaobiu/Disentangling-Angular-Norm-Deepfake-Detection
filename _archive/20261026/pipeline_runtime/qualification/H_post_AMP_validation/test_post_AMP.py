from pathlib import Path
import ast,importlib.util,json,datetime,copy
import pytest
P=Path(__file__).parent/'continue_H_post_AMP.py'
spec=importlib.util.spec_from_file_location('postamp',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_exact_original_tail_and_no_training_batch_before_it():
 tree=m.scientific_tail();src=ast.unparse(tree)
 assert src.startswith('effective = {}')
 assert 'audit_paired_ledgers(self.path, b)' in src
 assert "registry.update(effective_runs=effective, target_export_allowed=True" in src
 assert 'H_FORMAL_AMP_TRAINING' not in src
 assert 'standard_development_tasks' not in src

def test_unknown_source_rejected(tmp_path):
 (tmp_path/'queue_h.py').write_text('x=1')
 with pytest.raises(ValueError):m.scientific_tail(tmp_path)

@pytest.mark.parametrize('owner,workers',[(True,[]),(False,[42])])
def test_living_original_owner_or_worker_rejected(tmp_path,owner,workers,monkeypatch):
 if owner:(tmp_path/'2222605').mkdir()
 monkeypatch.setattr(m.base,'live_workers',lambda root,proc:workers)
 with pytest.raises(ValueError):m.no_live_owner(tmp_path,tmp_path)

def test_dead_owner_no_worker_pass(tmp_path,monkeypatch):
 monkeypatch.setattr(m.base,'live_workers',lambda root,proc:[])
 m.no_live_owner(tmp_path,tmp_path)

def test_fallback_stops_before_any_training_or_registration():
 with pytest.raises(m.UniformFP32ReviewRequired):m.reject_new_training(100,additional_bytes=12)

@pytest.mark.parametrize('component',sorted(m.COMPONENTS))
def test_budget_requires_every_component(tmp_path,component):
 b=budget(tmp_path);del b['remaining_components'][component]
 with pytest.raises(ValueError):check(tmp_path,b)

def budget(root):
 p=root/'base.json';p.write_text('{}')
 return dict(schema='FULL_REMAINING_PRIMARY_CAPACITY_V2',status='REGISTERED_PRIMARY_REMAINING_COVERED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remaining_components={k:0 for k in m.COMPONENTS},normal_remaining_write_bytes=0,temporary_bytes=16*m.G,reserve_bytes=50*m.G,normal_required_bytes=66*m.G,source_best_active_budget=None,compact_unvalidated_storage_savings_counted=False,SSD_cold_release_bytes=0,D_individual_complete_count=15,D_remaining_analysis=0,H_current_persistent_run_bytes=10,H_current_temporary_run_bytes=0,H_persistent_training_budget_total_bytes=5,remaining_conditionals={'H8_DINO_extension_and_longer_formal_aux':0},base_live_path=str(p),base_live_sha256=m.base.digest(p))

def check(root,b,free=100*m.G):
 p=root/'budget.json';p.write_text(json.dumps(b));return m.budget_check(root,p,m.base.digest(p),now=datetime.datetime.now(datetime.timezone.utc),free=free)

def test_persistent_exceeding_forecast_clamped_not_negative(tmp_path):
 assert check(tmp_path,budget(tmp_path))['required_bytes']==66*m.G

@pytest.mark.parametrize('key,val',[('temporary_bytes',15*m.G),('reserve_bytes',49*m.G),('normal_required_bytes',1),('compact_unvalidated_storage_savings_counted',True),('SSD_cold_release_bytes',1),('source_best_active_budget',1),('utc','2020-01-01T00:00:00+00:00')])
def test_bad_budget_rejected(tmp_path,key,val):
 b=budget(tmp_path);b[key]=val
 with pytest.raises(ValueError):check(tmp_path,b)

def test_live_free_failure(tmp_path):
 with pytest.raises(ValueError):check(tmp_path,budget(tmp_path),free=65*m.G)

def test_whole_three_ledger_success_publishes_original_registry(tmp_path):
 q=Fake(tmp_path);m.run_original_tail(q,api(q));assert q.saved_registry['target_export_allowed'] is True
 assert set(q.saved_registry['effective_runs'])=={'a','b','c'}
 assert len(q.saved_gates)==3 and q.batch_calls==0

def test_failed_second_ledger_never_publishes_global_registry(tmp_path):
 q=Fake(tmp_path)
 with pytest.raises(RuntimeError):m.run_original_tail(q,api(q,bad='b'))
 assert q.saved_registry is None

def test_fallback_does_not_register_or_batch(tmp_path):
 q=Fake(tmp_path)
 with pytest.raises(m.UniformFP32ReviewRequired):m.run_original_tail(q,api(q,bad='fallback'))
 assert not q.registered and q.batch_calls==0 and q.saved_registry is None

class Fake:
 def __init__(self,root):
  self.root=root;self.path=root/'config.json';self.config={'H_data_freeze_sha256':'f','h_code_bundle_sha256':'c'};self.saved_registry=None;self.saved_gates=[];self.batch_calls=0;self.registered=[]
  (root/'H_run_registry.json').write_text('{}');(root/'configs').mkdir();(root/'configs/H_frozen_recipes.json').write_text('{}')
 def _register(self,tasks):self.registered.append(tasks)
 def batch(self,*args):self.batch_calls+=1
 def status(self,*args):pass

def api(q,bad=None):
 def audit(path,b):return {'status':'REQUIRES_UNIFORM_FP32_FALLBACK' if bad=='fallback' else 'BAD' if b==bad else 'MATCHED_SUCCESSFUL_UPDATE_STREAMS'}
 def save(path,value):
  if path.name=='H_run_registry.json':q.saved_registry=value
  if path.parent.name=='qualification':q.saved_gates.append(value)
 return dict(BACKBONES=('a','b','c'),FORMAL_SEEDS=(7,42,123,2026,3407),HISTORIES=('B','AB'),audit_paired_ledgers=audit,save_json=save,digest=lambda p:'h',formal_id=lambda b,h,s,p:f'{b}_{h}_{s}_{p}',formal_tasks=lambda *a,**k:{})

@pytest.fixture
def full(tmp_path):
 root=tmp_path/'study';(root/'configs').mkdir(parents=True);(root/'data/H').mkdir(parents=True);(root/'qualification').mkdir();(root/'reports').mkdir();(root/'runs').mkdir();proc=tmp_path/'proc';proc.mkdir()
 api=m.base.pure_api();config=json.loads((m.ROOT/'configs/H_execution.json').read_text());config['artifact_root']=str(root)
 for name in ['data/H/freeze.json','qualification/H_gate.json']:(root/name).write_text('{}')
 config['H_data_freeze_sha256']=m.base.digest(root/'data/H/freeze.json');config['H_qualification_sha256']=m.base.digest(root/'qualification/H_gate.json')
 recipe=json.loads((m.ROOT/'configs/H_frozen_recipes.json').read_text());recipe['H_data_freeze_sha256']=config['H_data_freeze_sha256'];rp=root/'configs/H_frozen_recipes.json';rp.write_text(json.dumps(recipe))
 formal=api['formal_tasks'](config,recipe['recipes'],recipe_sha256=m.base.digest(rp));config['runs']=dict(api['standard_development_tasks'](config),**formal);cp=root/'configs/H_execution.json';cp.write_text(json.dumps(config))
 records=[];names=api['_completion_artifacts'](30)
 for run in formal:
  folder=root/'runs'/run;folder.mkdir()
  for name in names:(folder/name).write_text('{}')
  complete=dict(status='TRAINING_COMPLETE',epochs=30,attempts=5730,success=5730,skip=0,target_used=False,selection='frozen_last');(folder/'complete.json').write_text(json.dumps(complete))
  seal=dict(status='SEALED',run_id=run,artifact_root=str(root),config_path=str(cp),target_used=False,H_data_freeze_sha256=config['H_data_freeze_sha256'],h_code_bundle_sha256=config['h_code_bundle_sha256'],artifact_sha256={name:m.base.digest(folder/name) for name in names});sp=folder/'H_completion_seal.json';sp.write_text(json.dumps(seal))
  records.append(dict(run_id=run,terminal_seal_sha256=m.base.digest(sp),terminal_complete_sha256=m.base.digest(folder/'complete.json'),state_sha256=m.base.digest(folder/'state.pt'),seal_current_artifacts_verified=158,snapshot_epoch=30,snapshot_attempts=5730,snapshot_successful_updates=5730,snapshot_AMP_skips=0,source_DEV_tables_independently_recomputed=30,source_DEV_metric_checks=180,source_DEV_max_auc_abs_error=1e-16,target_used=False))
 (root/'H_run_registry.json').write_text(json.dumps(dict(formal_run_ids=list(formal),effective_runs={},target_export_allowed=False)))
 (root/'H_queue_status.json').write_text(json.dumps(dict(owner_pid=2222605,active=[],stage='H_FORMAL_AMP_COMPLETE_INDEPENDENT_TERMINAL_AND_PAIRED_LEDGER_REVIEW_REQUIRED')))
 (root/'queue_status.json').write_text(json.dumps(dict(stage='P3_D_CACHE_COMPLETE_ANALYSIS_PENDING',active=[])))
 audit=root/'reports/audit.json';payload=dict(status='PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS',source_only=True,CUDA_initialized=False,production_processes_restarted=False,selected_run_ids=list(formal),records=records,inputs={'H_execution_sha256':m.base.digest(cp)});audit.write_text(json.dumps(payload))
 listing=root/'audit_list.json';listing.write_text(json.dumps({str(audit):m.base.digest(audit)}));b=root/'budget.json';b.write_text(json.dumps(budget(root)))
 def run():return m.preflight(root,b,m.base.digest(b),listing,m.base.digest(listing),proc_root=proc)
 return root,audit,listing,run

def test_full30_synthetic_current_seals_preflight_pass(full):
 root,audit,listing,run=full;assert run()['formal_models']==30

@pytest.mark.parametrize('change',['missing','duplicate','wrong_config','target','metric'])
def test_independent_coverage_and_proof_drift_rejected(full,change):
 root,audit,listing,run=full;a=json.loads(audit.read_text())
 if change=='missing':a['records'].pop();a['selected_run_ids'].pop()
 if change=='duplicate':a['records'].append(a['records'][0])
 if change=='wrong_config':a['inputs']['H_execution_sha256']='0'*64
 if change=='target':a['records'][0]['target_used']=True
 if change=='metric':a['records'][0]['source_DEV_metric_checks']=179
 audit.write_text(json.dumps(a));listing.write_text(json.dumps({str(audit):m.base.digest(audit)}))
 with pytest.raises(ValueError):run()

def test_sealed_bytes_drift_rejected(full):
 root,audit,listing,run=full;next((root/'runs').glob('*/final.pt')).write_text('tampered')
 with pytest.raises(ValueError):run()

def test_prior_global_promotion_rejected(full):
 root,audit,listing,run=full;p=root/'H_run_registry.json';a=json.loads(p.read_text());a['target_export_allowed']=True;p.write_text(json.dumps(a))
 with pytest.raises(ValueError):run()

def test_original_nonblocking_lock_exclusion(tmp_path):
 lock=m.base.original_lock()
 with lock(tmp_path/'H_pipeline.lock'):
  with pytest.raises((BlockingIOError,RuntimeError)):
   with lock(tmp_path/'H_pipeline.lock'):pass

def test_fallback_exception_retains_exact_capacity_observation():
 with pytest.raises(m.UniformFP32ReviewRequired) as info:m.reject_new_training(123,additional_bytes=456)
 assert info.value.capacity_observation=={'live_free_bytes':123,'projected_uniform_branch_bytes':456,'full_remaining_branch_admission_complete':False}
