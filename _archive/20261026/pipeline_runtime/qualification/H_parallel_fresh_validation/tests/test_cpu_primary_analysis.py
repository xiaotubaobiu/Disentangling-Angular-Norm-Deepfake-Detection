from pathlib import Path
import importlib.util,copy,json
import pytest

def api():
 p=Path(__file__).parents[1]/'cpu_primary_analysis.py';s=importlib.util.spec_from_file_location('cpu_primary_analysis_entry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def valid():return dict(schema='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1',status='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE',allowed_backbones=['clip_vit_b16'],allowed_use=['independent_role_admission','primary_pair_analysis'],original_role_checks=14,fresh_default_source_checks=140,all_original_ledger_checks_exact_except_utc=True,independent_review_pass=True,production_GPU_allowed=False)

def test_plan_has_no_scientific_execution():
 m=api();r=m.main([]);assert r['execute'] is False and r['GPU_launch_attempted'] is False

@pytest.mark.parametrize('field,value',[('original_role_checks',13),('fresh_default_source_checks',14),('independent_review_pass',False),('production_GPU_allowed',True),('all_original_ledger_checks_exact_except_utc',False)])
def test_actual_qualification_requires_complete_exact_checks(field,value):
 m=api();q=valid();m.validate_qualification(q,'clip_vit_b16');q[field]=value
 with pytest.raises(ValueError):m.validate_qualification(q,'clip_vit_b16')

def test_qualification_backbone_and_typed_counts_not_coerced():
 m=api();q=valid()
 with pytest.raises(ValueError):m.validate_qualification(q,'dinov2_vit_b14')
 q['original_role_checks']=14.0
 with pytest.raises(ValueError):m.validate_qualification(q,'clip_vit_b16')

def test_missing_actual_gate_refuses_before_science_import(tmp_path,monkeypatch):
 m=api();monkeypatch.setattr(m,'ROOT',tmp_path)
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(name,'2')
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
 with pytest.raises(FileNotFoundError):m.main(['--execute','--backbone','clip_vit_b16','--seed','42','--slot','1','--qualification',str(tmp_path/'reports/absent.json'),'--qualification-sha','0'*64,'--admission',str(tmp_path/'reports/absent_pair.json'),'--admission-sha','0'*64])

@pytest.mark.parametrize('models,roles,seed',[(1,14,42),(2,7,42),(2,14,7)])
def test_pair_admission_requires_exact_pair_scope(models,roles,seed):
 m=api();d=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',backbone='clip_vit_b16',seed=42,models=2,roles=14,identity={'run_ids':{'B':'b','AB':'ab'},'cache_completion_sha256':{h+'/'+r:'0'*64 for h in ('B','AB') for r in m.ROLES}})
 m.validate_pair(d,'clip_vit_b16',42,{'B':'b','AB':'ab'});d.update(models=models,roles=roles,seed=seed)
 with pytest.raises(ValueError):m.validate_pair(d,'clip_vit_b16',42,{'B':'b','AB':'ab'})

def test_implementation_hash_checked_before_admission_executes(tmp_path,monkeypatch):
 import hashlib,sys
 m=api();monkeypatch.setattr(m,'ROOT',tmp_path);folder=tmp_path/'qualification/H_parallel_fresh_validation';folder.mkdir(parents=True);selfpath=folder/'cpu_primary_analysis.py';selfpath.write_text('pass');monkeypatch.setattr(m,'__file__',str(selfpath));sentinel=tmp_path/'foreign_executed'
 for name in ('parallel_source.py','cpu_ledger_adapter.py'):(folder/name).write_text('pass')
 (folder/'cpu_ledger_admission.py').write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("EXECUTED")\nraise RuntimeError("foreign admission executed")\n')
 q=valid();q['implementation_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('parallel_source.py','cpu_ledger_adapter.py')};q['implementation_sha256']['cpu_ledger_admission.py']='0'*64
 reports=tmp_path/'reports';reports.mkdir();qp=reports/'q.json';dp=reports/'d.json';qp.write_text(json.dumps(q));dp.write_text('{}')
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(name,'2')
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.delitem(sys.modules,'cpu_ledger_admission',raising=False)
 with pytest.raises(ValueError):m.main(['--execute','--qualification',str(qp),'--qualification-sha',hashlib.sha256(qp.read_bytes()).hexdigest(),'--admission',str(dp),'--admission-sha',hashlib.sha256(dp.read_bytes()).hexdigest()])
 assert not sentinel.exists()

def test_preloaded_local_dependency_cannot_execute(monkeypatch):
 import sys,types
 m=api();seen=[]
 def bad(*a):seen.append(True);raise AssertionError('preloaded dependency executed')
 monkeypatch.setitem(sys.modules,'cpu_ledger_admission',types.SimpleNamespace(input_pins=bad))
 monkeypatch.setattr(m,'read_proof',lambda p,pin:valid() if 'qualification' in str(p) else {})
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(name,'2')
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
 with pytest.raises(ValueError,match='isolated'):m.main(['--execute','--qualification','qualification','--qualification-sha','0'*64,'--admission','admission','--admission-sha','0'*64])
 assert not seen

def test_slot_survives_python_scope_and_releases_on_process_exit(tmp_path):
 import subprocess,sys,fcntl,os
 p=Path(__file__).parents[1]/'cpu_primary_analysis.py'
 script='import importlib.util,pathlib,sys;sp=importlib.util.spec_from_file_location("lease_fixture",'+repr(str(p))+');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m.ROOT=pathlib.Path('+repr(str(tmp_path))+');m.legacy_slot_alive=lambda slot:False\nwith m.cpu_slot(0):pass\nprint("READY",flush=True)\nsys.stdin.readline()\n'
 child=subprocess.Popen([sys.executable,'-B','-c',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 lock=None
 try:
  assert child.stdout.readline().strip()=='READY';lock=(tmp_path/'analysis/H/.cpu_slots/slot0.lock').open('a+b')
  with pytest.raises(BlockingIOError):fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 finally:
  if child.poll() is None:child.communicate('exit\n',timeout=5)
  if lock is not None:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
 assert child.returncode==0
