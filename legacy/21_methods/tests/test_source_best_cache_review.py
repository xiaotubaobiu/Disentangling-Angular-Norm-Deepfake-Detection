import json,hashlib
from pathlib import Path
import numpy as np,pytest,torch
from test_source_best_cache import project,sha,save
from reliability_v2 import source_best_cache as m

def cached(e):
 c=e['context'];role=c.roles[0];h=np.ones((len(e['frames'][role]),512),np.float32)
 m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[h[start:]])
 return c,role,m.role_path(c,role)

def test_review_dangling_role_lock_symlink_no_external_creation(project,tmp_path):
 e=project;c=e['context'];role=c.roles[0];folder=m.role_path(c,role);folder.mkdir(parents=True);outside=tmp_path/'outside-lock';(folder/'write.lock').symlink_to(outside)
 with pytest.raises((ValueError,OSError)):m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[np.ones((len(e['frames'][role]),512),np.float32)])
 assert not outside.exists() and not (folder/'complete.json').exists()

def test_review_extract_ancestor_symlink_no_external_directories(project,tmp_path,monkeypatch):
 e=project;outside=tmp_path/'outside';outside.mkdir();(e['root']/'features').symlink_to(outside);monkeypatch.setattr(m,'_permit',lambda *a:{})
 with pytest.raises((ValueError,OSError)):m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path='synthetic-only',permit_sha256='0'*64,device='cpu')
 assert list(outside.iterdir())==[]

def test_review_cached_manifest_leaf_symlink_refused(project,tmp_path):
 c,role,folder=cached(project);p=folder/'manifest.parquet';outside=tmp_path/'outside.parquet';outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside);before=sha(outside)
 with pytest.raises(ValueError):m.verify_role(c,role)
 assert sha(outside)==before

def test_review_metadata_extra_fields_exceed_bound_refused(project):
 c,role,folder=cached(project);p=folder/'complete.json';meta=json.loads(p.read_text());meta['unregistered_padding']='x'*(2*1024**2);save(p,meta)
 with pytest.raises(ValueError):m.verify_role(c,role)

def test_review_metadata_unknown_field_refused_even_if_small(project):
 c,role,folder=cached(project);p=folder/'complete.json';meta=json.loads(p.read_text());meta['extra']='x';save(p,meta)
 with pytest.raises(ValueError):m.verify_role(c,role)

@pytest.mark.parametrize('filename',['chunk_999999.npy','.foreign.tmp'])
def test_review_unregistered_or_out_of_range_file_refused_preserved(project,filename):
 c,role,folder=cached(project);p=folder/filename;p.write_bytes(b'preserve failed/foreign evidence');before=sha(p)
 with pytest.raises(ValueError):m.verify_role(c,role)
 assert sha(p)==before

def test_review_changed_production_permit_during_inference_refuses(project,monkeypatch):
 e=project;c=e['context'];root=e['root'];proofs={};code=sha(m.__file__)
 for name in ('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity'):
  p=root/(name+'.json');save(p,dict(M_production_cache_admission=True,cache_code_sha256=code,input_admission_sha256=e['admission_sha'],required_remaining_bytes=1));proofs[name]=dict(path=str(p),sha256=sha(p))
 permit=root/'synthetic_unit_permit.json';save(permit,dict(schema=m.PERMIT_SCHEMA,status='PASS_PRODUCTION_CACHE_ADMISSION',artifact_root=str(root),input_admission_sha256=e['admission_sha'],cache_code_sha256=code,target_inference_authorized=True,four_core_comparisons_only=True,parent_run_ids=[e['run']],**proofs));pin=sha(permit)
 # Patch the returned model class, which will receive the selected source state.
 factory=m._build_model
 def build(*a):
  model=factory(*a);original=model.encoder.forward
  def changed(x):permit.write_bytes(b'changed after initial authority check');return original(x)
  model.encoder.forward=changed;return model
 monkeypatch.setattr(m,'_build_model',build)
 with pytest.raises(ValueError):m.extract(root,e['admission'],e['admission_sha'],e['run'],permit_path=permit,permit_sha256=pin,device='cpu')
