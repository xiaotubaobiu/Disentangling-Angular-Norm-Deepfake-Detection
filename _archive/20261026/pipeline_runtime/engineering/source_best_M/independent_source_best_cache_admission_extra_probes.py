"""Additional immutable evidence: verified-byte/receipt binding and closure consumer sweep."""
from pathlib import Path
from types import SimpleNamespace
import json,pytest
import test_source_best_cache_admission as original
from test_source_best_cache import project
from independent_source_best_cache_admission_probes import make_corpus,bind_trusted_owner,mutate_head,INPUT
m=original.m;cache=m.cache

def test_real_D_verified_marker_and_recorded_SHA_must_be_same_bytes(tmp_path,monkeypatch):
 p=project.__wrapped__(tmp_path,monkeypatch,SimpleNamespace(param='D'));p,c=original.cached.__wrapped__(p)
 marker=cache.role_path(c,c.roles[0])/'complete.json';verified_sha=m.digest(marker);verify=cache.verify_role
 def replace_after_verify(context,role):
  answer=verify(context,role)
  if role==c.roles[0]:marker.write_text('{}')
  return answer
 monkeypatch.setattr(cache,'verify_role',replace_after_verify)
 result=m.admit_contexts(m.analysis.job('D','clip_vit_b16',7),{'D':c})
 published_sha=result['cache_completion_sha256']['D/'+c.roles[0]]
 assert published_sha==m.digest(marker) and published_sha!=verified_sha
 with pytest.raises(ValueError,match='unexpected cache completion fields'):verify(c,c.roles[0])
 print('PROBE verified_marker_vs_recorded',json.dumps({'status':result['status'],'actually_verified_sha256':verified_sha,'recorded_corrupt_marker_sha256':published_sha}))
 assert result['status']!='PASS','BINDING GAP: recorded marker SHA refers to bytes not validated by verify_role'

def test_capacity_consumer_rejects_drift_during_its_own_revalidation(tmp_path,monkeypatch):
 root=tmp_path/'artifacts';stage,plan,ref,first=make_corpus(root,m.analysis.expected_feature_runs(),'cache');bind_trusted_owner(monkeypatch,stage,plan)
 out=m.collect_feature_capacity(root,ref);owned=m.owned_role;calls=[]
 def late(*args,**kwargs):
  answer=owned(*args,**kwargs);calls.append((args[2],args[3]))
  if len(calls)==180:mutate_head(first[2]/'head_weight.npy')
  return answer
 monkeypatch.setattr(m,'owned_role',late)
 credit=m.validate_feature_capacity(root,out)
 with pytest.raises(ValueError,match='head bytes changed'):owned(root,INPUT,first[0],first[1],first[3])
 print('PROBE capacity_consumer_sweep',json.dumps({'returned_remaining_bytes':credit,'current_first_head_invalid':True,'revalidated_roles':len(calls)}))
 assert credit!=0,'STALE ZERO CREDIT: fresh closure regeneration itself lacked a final aggregate rebind'
