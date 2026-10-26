from pathlib import Path
import importlib.util,os
import pytest

def module():
 p=Path(__file__).parents[1]/'cpu_ledger_admission.py';s=importlib.util.spec_from_file_location('candidate_cpu_admission_entry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_plan_is_explicitly_nonexecuting():
 m=module();r=m.main([]);assert r['execute'] is False and r['GPU_launch_attempted'] is False

def test_missing_execute_does_not_read_root(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m,'ROOT',tmp_path/'absent');assert m.main([])['execute'] is False

def test_existing_report_is_preserved_before_cpu_or_input_checks(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'reports').mkdir();out=m.report_path('clip_vit_b16',42);out.write_text('preserve');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','visible')
 with pytest.raises(FileExistsError):m.main(['--execute','--backbone','clip_vit_b16','--seed','42'])
 assert out.read_text()=='preserve'

def test_gpu_visible_or_unbounded_threads_reject_before_scientific_import(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(ValueError,match='CPU'):m.main(['--execute','--backbone','clip_vit_b16','--seed','42'])
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(key,'2')
 monkeypatch.setenv('OMP_NUM_THREADS','8')
 with pytest.raises(ValueError,match='CPU'):m.main(['--execute','--backbone','clip_vit_b16','--seed','42'])

def test_invalid_backbone_and_seed_reject_before_launch():
 m=module()
 with pytest.raises(SystemExit):m.main(['--execute','--backbone','foreign','--seed','42'])
 with pytest.raises(SystemExit):m.main(['--execute','--backbone','clip_vit_b16','--seed','1'])

def test_current_gate_hash_must_match_pinned_accepted_snapshot(monkeypatch):
 m=module()
 if not m.ROOT.exists():pytest.skip('actual canonical metadata fixture unavailable')
 original=m.digest;gate=m.ROOT/'qualification/H_target_export_dinov2_vit_b14.json'
 monkeypatch.setattr(m,'digest',lambda p:'0'*64 if Path(p)==gate else original(p))
 with pytest.raises(ValueError):m.input_pins('clip_vit_b16')

def test_all3_snapshot_missing_seal_is_rejected(monkeypatch):
 import types,json
 m=module()
 if not m.ROOT.exists():pytest.skip('actual canonical metadata fixture unavailable')
 original=json.loads
 def altered(s):
  value=original(s)
  if isinstance(value,dict) and 'current_canonical_proof_sha256' in value:
   keys=[k for k in value['current_canonical_proof_sha256'] if k.endswith('H_completion_seal.json')]
   del value['current_canonical_proof_sha256'][keys[-1]]
  return value
 monkeypatch.setattr(m,'json',types.SimpleNamespace(loads=altered))
 with pytest.raises(ValueError):m.input_pins('clip_vit_b16')
