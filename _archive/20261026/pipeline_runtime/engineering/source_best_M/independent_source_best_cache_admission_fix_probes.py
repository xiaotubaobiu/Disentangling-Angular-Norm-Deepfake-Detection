"""Independent cache-admission fix verification; bounded /tmp only; prior evidence intact."""
from pathlib import Path
from types import SimpleNamespace
import json,pytest
import test_source_best_cache_admission as original
from test_source_best_cache import project
import independent_source_best_cache_admission_probes as old
m=original.m;cache=m.cache

def build_D(tmp_path,monkeypatch):
 p=project.__wrapped__(tmp_path,monkeypatch,SimpleNamespace(param='D'))
 return original.cached.__wrapped__(p)

@pytest.mark.parametrize('mode',['source_qualification','cache'])
def test_fixed_collectors_positive_inventory(tmp_path,monkeypatch,mode):
 old.test_collectors_tiny_positive_exact_inventory(tmp_path,monkeypatch,mode)

@pytest.mark.parametrize('mutation',['earlier_head','same_verified_marker'])
def test_fixed_real_D_rejects_verified_or_earlier_derived_drift(tmp_path,monkeypatch,mutation):
 p,c=build_D(tmp_path,monkeypatch);first=cache.role_path(c,c.roles[0]);verify=cache.verify_role
 def late(context,role):
  answer=verify(context,role)
  if mutation=='earlier_head' and role==c.roles[-1]:old.mutate_head(first/'head_weight.npy')
  elif mutation=='same_verified_marker' and role==c.roles[0]:(first/'complete.json').write_text('{}')
  return answer
 monkeypatch.setattr(cache,'verify_role',late)
 with pytest.raises(ValueError,match='derived role changed before aggregate admission publication'):
  m.admit_contexts(m.analysis.job('D','clip_vit_b16',7),{'D':c})
 print('FIX_REJECT',mutation)

@pytest.mark.parametrize('mode',['source_qualification','cache'])
def test_fixed_collectors_reject_late_head(tmp_path,monkeypatch,mode):
 runs=m.qualification.qualification_runs() if mode=='source_qualification' else m.analysis.expected_feature_runs()
 root=tmp_path/'artifacts';stage,plan,ref,first=old.make_corpus(root,runs,mode);old.bind_trusted_owner(monkeypatch,stage,plan)
 owned=m.owned_role;calls=[];expected=9 if mode=='source_qualification' else 180
 def late(*args,**kwargs):
  result=owned(*args,**kwargs);calls.append((args[2],args[3]))
  if len(calls)==expected:old.mutate_head(first[2]/'head_weight.npy')
  return result
 monkeypatch.setattr(m,'owned_role',late)
 with pytest.raises(ValueError,match='derived role changed before aggregate admission publication'):
  m.collect_source_qualification(root,ref,0) if mode=='source_qualification' else m.collect_feature_capacity(root,ref)
 assert not (root/'qualification/M_source_only_GPU_bridge'/old.INPUT/'accepted.json').exists()
 assert not (root/'reports/M_feature_capacity_closure'/old.INPUT/'complete.json').exists()
 print('FIX_REJECT',mode)

def test_fixed_capacity_consumer_rejects_own_sweep_drift(tmp_path,monkeypatch):
 root=tmp_path/'artifacts';stage,plan,ref,first=old.make_corpus(root,m.analysis.expected_feature_runs(),'cache');old.bind_trusted_owner(monkeypatch,stage,plan)
 out=m.collect_feature_capacity(root,ref);owned=m.owned_role;calls=[]
 def late(*args,**kwargs):
  answer=owned(*args,**kwargs);calls.append((args[2],args[3]))
  if len(calls)==180:old.mutate_head(first[2]/'head_weight.npy')
  return answer
 monkeypatch.setattr(m,'owned_role',late)
 with pytest.raises(ValueError,match='derived role changed before aggregate admission publication'):m.validate_feature_capacity(root,out)
 print('FIX_REJECT capacity_consumer_sweep')

@pytest.mark.parametrize('mutation',['foreign_file','oversized_head'])
def test_bounded_preflight_must_reject_before_snapshot_whole_file_read(tmp_path,monkeypatch,mutation):
 p,c=build_D(tmp_path,monkeypatch);folder=cache.role_path(c,c.roles[0])
 if mutation=='foreign_file':
  target=folder/'foreign.bin';target.write_bytes(b'keep')
 else:
  target=folder/'head_weight.npy'
  # Sparse logical file only: spy blocks any attempted full read. No huge RAM allocation.
  with target.open('r+b') as f:f.truncate(128*1024**2)
 digest=m.digest;early=[]
 def guarded(path):
  if Path(path)==target:
   early.append(str(path));raise AssertionError('snapshot attempted a whole-file read before bounded inventory/NPY rejection')
  return digest(path)
 monkeypatch.setattr(m,'digest',guarded)
 with pytest.raises(ValueError,match='unregistered candidate/orphan artifact|array header/file byte budget mismatch'):
  m.admit_contexts(m.analysis.job('D','clip_vit_b16',7),{'D':c})
 assert early==[]

def test_final_derived_namespace_includes_model_parent_inventory(tmp_path,monkeypatch):
 p,c=build_D(tmp_path,monkeypatch);first=cache.role_path(c,c.roles[0]);parent=first.parent;verify=cache.verify_role
 def late(context,role):
  answer=verify(context,role)
  if role==c.roles[-1]:(parent/'foreign_parent.bin').write_bytes(b'keep')
  return answer
 monkeypatch.setattr(cache,'verify_role',late)
 result=m.admit_contexts(m.analysis.job('D','clip_vit_b16',7),{'D':c})
 with pytest.raises(ValueError,match='unregistered model namespace artifact'):verify(c,c.roles[0])
 print('FIX_RESIDUAL_PARENT',json.dumps({'status':result['status'],'foreign_parent_preserved':(parent/'foreign_parent.bin').read_bytes()==b'keep','real_verify_now_refuses':True}))
 assert result['status']!='PASS','STALE PASS: final rebind checked leaves but omitted strict model-parent namespace inventory'
