import copy,hashlib,json,os,zipfile
from pathlib import Path
import numpy as np
import pytest
from reliability_v2 import multiplicity_bundle as m
LIMITS=dict(max_cells=20000,max_json_bytes=4000000,max_array_bytes=1000000)
IDENTITY={'endpoint':'M_source_best_D_formal_clip_vit_b16_seed7','scope':'engineering_only','draws':2000}
def value():
 return {'NA':'NA','negative':-.025,'precision':'0.12345678901234567890','draws':[{'repeat':i,'group_draw_counts':{'g2':i%3,'g1':2},'effect':None} for i in range(2000)]}
def read(path,r,**kw):return m.read_bundle(path,expected_complete_sha256=r['complete_sha256'],expected_identity=IDENTITY,**(LIMITS|kw))
def manifest(path):return json.loads((path/'complete.json').read_text())
def rewrite_manifest(path,data):
 (path/'complete.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');return hashlib.sha256((path/'complete.json').read_bytes()).hexdigest()

def test_real2000_draws_all_fields_and_order_roundtrip(tmp_path):
 p=tmp_path/'bundle';v=value();r=m.write_bundle(p,v,identity=IDENTITY,**LIMITS);restored=read(p,r)
 assert restored==v and list(restored['draws'][0]['group_draw_counts'])==['g2','g1']
 assert r['raw_json_sha256']==hashlib.sha256((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
 assert r['production_admitted'] is False

def test_present_zero_absent_and_empty_count_maps_are_exact(tmp_path):
 v={'rows':[{'group_draw_counts':{'a':0}},{'group_draw_counts':{}},{'closure_draw_counts':{'z':1}},{'x':[{'group_draw_counts':{'x':3}}]}]}
 p=tmp_path/'bundle';r=m.write_bundle(p,v,identity=IDENTITY,**LIMITS);assert read(p,r)==v

def test_no_overwrite_and_complete_resume_same_input(tmp_path):
 p=tmp_path/'bundle';v=value();r=m.write_bundle(p,v,identity=IDENTITY,**LIMITS);files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.name!='writer.lock'}
 with pytest.raises(FileExistsError):m.write_bundle(p,v,identity=IDENTITY,**LIMITS)
 rr=m.write_bundle(p,v,identity=IDENTITY,resume=True,**LIMITS);assert rr['complete_sha256']==r['complete_sha256']
 assert files=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.name!='writer.lock'}

@pytest.mark.parametrize('kind',['identity','value'])
def test_resume_foreign_context_or_input_rejects_and_preserves(tmp_path,kind):
 p=tmp_path/'bundle';v=value();r=m.write_bundle(p,v,identity=IDENTITY,**LIMITS);ident=copy.deepcopy(IDENTITY)
 if kind=='identity':ident['endpoint']='foreign'
 else:v['negative']=.2
 with pytest.raises(ValueError):m.write_bundle(p,v,identity=ident,resume=True,**LIMITS)
 assert (p/'complete.json').is_file() and read(p,r)==value()

def test_read_requires_external_sha_identity_and_positive_budgets(tmp_path):
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
 with pytest.raises(ValueError):m.read_bundle(p,expected_complete_sha256='0'*64,expected_identity=IDENTITY,**LIMITS)
 with pytest.raises(ValueError):m.read_bundle(p,expected_complete_sha256=r['complete_sha256'],expected_identity={'foreign':True},**LIMITS)
 for key in LIMITS:
  with pytest.raises(ValueError):read(p,r,**{key:True})

@pytest.mark.parametrize('budget,value_limit',[('max_cells',1),('max_json_bytes',10),('max_array_bytes',1)])
def test_writer_and_reader_enforce_resource_budgets(tmp_path,budget,value_limit):
 p=tmp_path/'bundle';v=value()
 with pytest.raises(ValueError):m.write_bundle(p,v,identity=IDENTITY,**(LIMITS|{budget:value_limit}))
 p=tmp_path/'valid';r=m.write_bundle(p,v,identity=IDENTITY,**LIMITS)
 with pytest.raises(ValueError):read(p,r,**{budget:value_limit})

@pytest.mark.parametrize('filename',['metadata.json.gz','multiplicities.npz','complete.json'])
def test_corrupt_components_reject(tmp_path,filename):
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS);f=p/filename;f.write_bytes(f.read_bytes()[:-3]+b'bad')
 with pytest.raises(ValueError):read(p,r)

def test_partial_after_metadata_can_resume_without_rewriting(tmp_path,monkeypatch):
 p=tmp_path/'bundle';original=m._publish_npz
 def fail(*a,**kw):raise OSError('injected interruption')
 monkeypatch.setattr(m,'_publish_npz',fail)
 with pytest.raises(OSError):m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
 assert not (p/'complete.json').exists();meta=(p/'metadata.json.gz').read_bytes();monkeypatch.setattr(m,'_publish_npz',original)
 r=m.write_bundle(p,value(),identity=IDENTITY,resume=True,**LIMITS);assert read(p,r)==value() and (p/'metadata.json.gz').read_bytes()==meta

def test_final_publish_interruption_is_not_complete_and_resumes(tmp_path,monkeypatch):
 p=tmp_path/'bundle';original=os.link
 def fail(source,target,*a,**kw):
  if Path(target).name=='complete.json':raise OSError('publish interruption')
  return original(source,target,*a,**kw)
 monkeypatch.setattr(os,'link',fail)
 with pytest.raises(OSError):m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
 assert not (p/'complete.json').exists();monkeypatch.setattr(os,'link',original)
 r=m.write_bundle(p,value(),identity=IDENTITY,resume=True,**LIMITS);assert read(p,r)==value()

def test_unknown_existing_directory_is_not_modified(tmp_path):
 p=tmp_path/'foreign';p.mkdir();(p/'evidence').write_text('preserve')
 with pytest.raises(ValueError):m.write_bundle(p,value(),identity=IDENTITY,resume=True,**LIMITS)
 assert {x.name for x in p.iterdir()}=={'evidence'}

def test_symlink_output_and_component_rejected(tmp_path):
 foreign=tmp_path/'foreign';foreign.mkdir();p=tmp_path/'link';p.symlink_to(foreign,target_is_directory=True)
 with pytest.raises(ValueError):m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS);f=p/'multiplicities.npz';b=f.read_bytes();f.unlink();outside=tmp_path/'outside.npz';outside.write_bytes(b);f.symlink_to(outside)
 with pytest.raises(ValueError):read(p,r)

@pytest.mark.parametrize('change',['huge_shape','object_dtype','extra_member','duplicate_member'])
def test_npz_header_rejected_before_array_load(tmp_path,monkeypatch,change):
 p=tmp_path/'bundle';r=m.write_bundle(p,{'draws':[{'group_draw_counts':{'a':1}}]},identity=IDENTITY,**LIMITS);meta=manifest(p)
 import io
 stream=io.BytesIO()
 shape=(10**12,1) if change=='huge_shape' else (1,1);dtype='|O' if change=='object_dtype' else '<u4'
 np.lib.format.write_array_header_1_0(stream,{'descr':dtype,'fortran_order':False,'shape':shape})
 with zipfile.ZipFile(p/'multiplicities.npz','w',compression=zipfile.ZIP_DEFLATED) as z:
  z.writestr('counts_0.npy',stream.getvalue())
  array=io.BytesIO();np.save(array,np.array([[0]],dtype=np.int32),allow_pickle=False);z.writestr('order_0.npy',array.getvalue())
  if change=='extra_member':z.writestr('foreign.npy',b'bad')
  if change=='duplicate_member':z.writestr('counts_0.npy',stream.getvalue())
 meta['npz']['sha256']=hashlib.sha256((p/'multiplicities.npz').read_bytes()).hexdigest();meta['npz']['bytes']=(p/'multiplicities.npz').stat().st_size
 expected=rewrite_manifest(p,meta)
 def forbidden(*a,**kw):raise AssertionError('allocation attempted before validating header')
 monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(np,'frombuffer',forbidden)
 with pytest.raises(ValueError):m.read_bundle(p,expected_complete_sha256=expected,expected_identity=IDENTITY,**LIMITS)

def test_nonfinite_json_and_invalid_identity_reject(tmp_path):
 for i,v in enumerate([{'x':float('inf')},{'x':float('nan')} ]):
  with pytest.raises(ValueError):m.write_bundle(tmp_path/str(i),v,identity=IDENTITY,**LIMITS)
 with pytest.raises(ValueError):m.write_bundle(tmp_path/'identity',{},identity={},**LIMITS)

def test_writer_code_drift_cannot_publish_complete(tmp_path,monkeypatch):
 p=tmp_path/'bundle';actual=m._implementations();changed=dict(actual);changed['multiplicity_bundle.py']='0'*64;state=[False];publish=m._publish_npz
 monkeypatch.setattr(m,'_implementations',lambda:changed if state[0] else actual)
 def drift(*a,**kw):publish(*a,**kw);state[0]=True
 monkeypatch.setattr(m,'_publish_npz',drift)
 with pytest.raises(ValueError):m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
 assert not (p/'complete.json').exists()

def test_reader_rebinds_code_after_full_decode(tmp_path,monkeypatch):
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS);actual=m._implementations();changed=dict(actual);changed['multiplicity_bundle.py']='0'*64;state=[False];original=m._json_identity
 monkeypatch.setattr(m,'_implementations',lambda:changed if state[0] else actual)
 def drift(*a,**kw):result=original(*a,**kw);state[0]=True;return result
 monkeypatch.setattr(m,'_json_identity',drift)
 with pytest.raises(ValueError):read(p,r)

def test_active_writer_excludes_resume_and_does_not_leak_lock(tmp_path,monkeypatch):
 import threading
 p=tmp_path/'bundle';started=threading.Event();release=threading.Event();errors=[];publish=m._publish_npz
 def slow(*a,**kw):
  started.set()
  if not release.wait(4):raise RuntimeError('probe timeout')
  return publish(*a,**kw)
 def first():
  try:m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
  except BaseException as error:errors.append(error)
 monkeypatch.setattr(m,'_publish_npz',slow);thread=threading.Thread(target=first);thread.start();assert started.wait(2)
 try:
  with pytest.raises(RuntimeError,match='active'):m.write_bundle(p,value(),identity=IDENTITY,resume=True,**LIMITS)
 finally:release.set();thread.join(4)
 assert not thread.is_alive() and not errors
 monkeypatch.setattr(m,'_publish_npz',publish);r=m.write_bundle(p,value(),identity=IDENTITY,resume=True,**LIMITS);assert read(p,r)==value()

def test_authenticated_but_wrong_numeric_payload_rejected(tmp_path):
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS);doc=manifest(p)
 with np.load(p/'multiplicities.npz',allow_pickle=False) as z:arrays={name:z[name] for name in z.files}
 arrays['counts_0'][0,0]+=1;np.savez_compressed(p/'multiplicities.npz',**arrays)
 doc['npz']={'sha256':hashlib.sha256((p/'multiplicities.npz').read_bytes()).hexdigest(),'bytes':(p/'multiplicities.npz').stat().st_size};expected=rewrite_manifest(p,doc)
 with pytest.raises(ValueError,match='roundtrip'):m.read_bundle(p,expected_complete_sha256=expected,expected_identity=IDENTITY,**LIMITS)

def test_writer_context_drift_cannot_publish_complete(tmp_path,monkeypatch):
 p=tmp_path/'bundle';identity=copy.deepcopy(IDENTITY);publish=m._publish_npz
 def drift(*a,**kw):publish(*a,**kw);identity['endpoint']='changed'
 monkeypatch.setattr(m,'_publish_npz',drift)
 with pytest.raises(ValueError):m.write_bundle(p,value(),identity=identity,**LIMITS)
 assert not (p/'complete.json').exists()

def test_reader_context_rebound_after_decode(tmp_path,monkeypatch):
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS);expected=copy.deepcopy(IDENTITY);original=m._json_identity
 def drift(*a,**kw):out=original(*a,**kw);expected['endpoint']='changed';return out
 monkeypatch.setattr(m,'_json_identity',drift)
 with pytest.raises(ValueError):m.read_bundle(p,expected_complete_sha256=r['complete_sha256'],expected_identity=expected,**LIMITS)

@pytest.mark.parametrize('action',['read','resume'])
@pytest.mark.parametrize('foreign',[{'flag':1,'draws':1},{'flag':True,'draws':1.0}])
def test_typed_context_identity_is_not_python_numeric_equality(tmp_path,action,foreign):
 p=tmp_path/'bundle';stored={'flag':True,'draws':1};r=m.write_bundle(p,{},identity=stored,**LIMITS)
 with pytest.raises(ValueError):
  if action=='read':m.read_bundle(p,expected_complete_sha256=r['complete_sha256'],expected_identity=foreign,**LIMITS)
  else:m.write_bundle(p,{},identity=foreign,resume=True,**LIMITS)

@pytest.mark.parametrize('json_budget,padding',[(4000000,1048530),(4000000,1047854),(4096,3800)])
def test_full_evidence_wrapper_budget_checked_before_claiming_directory(tmp_path,json_budget,padding):
 p=tmp_path/'bundle';identity={'padding':'x'*padding}
 with pytest.raises(ValueError):m.write_bundle(p,{},identity=identity,**(LIMITS|{'max_json_bytes':json_budget}))
 assert not p.exists()

def test_canonical_identity_key_order_can_resume(tmp_path):
 p=tmp_path/'bundle';r=m.write_bundle(p,value(),identity=IDENTITY,**LIMITS);reverse=dict(reversed(list(IDENTITY.items())))
 rr=m.write_bundle(p,value(),identity=reverse,resume=True,**LIMITS);assert rr['complete_sha256']==r['complete_sha256']
 assert m.read_bundle(p,expected_complete_sha256=r['complete_sha256'],expected_identity=reverse,**LIMITS)==value()

def test_nonstring_identity_keys_rejected(tmp_path):
 with pytest.raises(ValueError):m.write_bundle(tmp_path/'bundle',{},identity={'nested':{1:'ambiguous'}},**LIMITS)

def test_before_first_transaction_publication_is_fail_closed_boundary(tmp_path,monkeypatch):
 p=tmp_path/'bundle';publish=m._atomic_json
 def fail(path,*a,**kw):
  if path.name=='transaction.json':raise OSError('first-claim interruption')
  return publish(path,*a,**kw)
 monkeypatch.setattr(m,'_atomic_json',fail)
 with pytest.raises(OSError):m.write_bundle(p,value(),identity=IDENTITY,**LIMITS)
 before={x.name:x.read_bytes() for x in p.iterdir()};assert 'transaction.json' not in before and 'complete.json' not in before
 monkeypatch.setattr(m,'_atomic_json',publish)
 with pytest.raises(ValueError):m.write_bundle(p,value(),identity=IDENTITY,resume=True,**LIMITS)
 assert before=={x.name:x.read_bytes() for x in p.iterdir()}
 fresh=tmp_path/'unclaimed-attempt';r=m.write_bundle(fresh,value(),identity=IDENTITY,**LIMITS);assert read(fresh,r)==value()
