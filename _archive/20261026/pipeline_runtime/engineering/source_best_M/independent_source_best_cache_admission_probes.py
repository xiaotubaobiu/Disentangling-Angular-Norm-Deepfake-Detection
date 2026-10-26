"""Independent synthetic-only cache-admission probes; owner boundary assumed validated.
No production images/checkpoints, GPU, model inference, or scientific jobs are used.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,json,shutil
import numpy as np,pandas as pd,pytest
import test_source_best_cache_admission as original
from test_source_best_cache import project
m=original.m
cache=m.cache
INPUT='a'*64

def mutate_head(path):
 a=np.load(path);a[0,0]+=1;np.save(path,a,allow_pickle=False)

def test_real_D_all_role_snapshot_rejects_late_earlier_head_drift(tmp_path,monkeypatch):
 p=project.__wrapped__(tmp_path,monkeypatch,SimpleNamespace(param='D'))
 p,c=original.cached.__wrapped__(p)
 first=cache.role_path(c,c.roles[0])/'head_weight.npy'
 verify=cache.verify_role
 def late(context,role):
  answer=verify(context,role)
  if role==c.roles[-1]:mutate_head(first)
  return answer
 monkeypatch.setattr(cache,'verify_role',late)
 result=m.admit_contexts(m.analysis.job('D','clip_vit_b16',7),{'D':c})
 with pytest.raises(ValueError,match='head bits changed'):verify(c,c.roles[0])
 print('PROBE real_D_late_drift',json.dumps({'status':result['status'],'fresh_all_roles':result['fresh_original_source_context_and_all_selected_role_checks'],'current_first_head_fails_real_verify':True}))
 assert result['status']!='PASS','STALE PASS: first role original-head bytes changed before aggregate admission returned'

def make_corpus(root,runs,mode):
 root.mkdir();receipts={};records=m.known_runs();roles_by_run={};first=None
 # Tiny table footer/hash fixtures, deliberately no real image or model data.
 for route in ('D','H'):
  for role in cache.ROLES[route]:
   source=root/f'data/{route}'/(role+'.parquet');source.parent.mkdir(parents=True,exist_ok=True)
   pd.DataFrame({'synthetic_row':[1]}).to_parquet(source,index=False)
 for run in runs:
  spec=records[run]['job'];route=spec['route'];dimension=cache.cache_d.DIMENSIONS[spec['backbone']]
  selected=cache.ROLES[route] if mode=='cache' else ('source_cal',) if route=='D' else ('B/SOURCE_CAL',)
  role_records={}
  for role in selected:
   source=root/f'data/{route}'/(role+'.parquet');source_sha=m.digest(source)
   folder=root/'features/source_best_M'/INPUT/run/role;folder.mkdir(parents=True)
   w=np.ones((2,dimension),np.float32);b=np.zeros((2,),np.float32);h=np.ones((1,dimension),np.float32)
   np.save(folder/'head_weight.npy',w,allow_pickle=False);np.save(folder/'head_bias.npy',b,allow_pickle=False);np.save(folder/'chunk_000000.npy',h,allow_pickle=False)
   head=cache._head_hash(w,b,dimension)
   ident=dict(checkpoint_selection='source_best',artifact_root=str(root),run_id=run,parent_run_id=run,role=role,route=route,dimension=dimension,input_admission_sha256=INPUT,cache_code_sha256=m.digest(cache.__file__),checkpoint_path=str(root/'runs'/run/'source_best.pt'),source_manifest_path=str(source),source_manifest_sha256=source_sha,accepted_table_rows=1,source_best_epoch=1,original_head_sha256=head,checkpoint_sha256='b'*64)
   item=dict(file='chunk_000000.npy',start=0,rows=1,dimension=dimension,dtype='float32',identity_sha256='c'*64,row_sha256='d'*64,zero_norm_count=0,sha256=m.digest(folder/'chunk_000000.npy'))
   meta=dict(schema=cache.SCHEMA,status='CACHE_COMPLETE',identity=ident,rows=1,dimension=dimension,precision='FP32',autocast=False,epsilon=cache.cache_d.EPSILON,zero_norm_count=0,chunk_rows=4096,cached_manifest_sha256=source_sha,head=dict(weight_file='head_weight.npy',bias_file='head_bias.npy',weight_sha256=m.digest(folder/'head_weight.npy'),bias_sha256=m.digest(folder/'head_bias.npy')),chunks=[item])
   if route=='H':meta['input_validation']='synthetic-only format receipt'
   for name,value in [('identity.json',ident),('chunk_000000.json',item),('complete.json',meta)]:
    (folder/name).write_text(json.dumps(value,sort_keys=True,separators=(',',':')))
   shutil.copyfile(source,folder/'manifest.parquet')
   marker_sha=m.digest(folder/'complete.json')
   role_records[role]={'complete_sha256':marker_sha}
   if first is None:first=(run,role,folder,marker_sha)
  if mode=='cache':rec={'roles':role_records}
  else:rec=dict(source_only_cache_complete_sha256=marker_sha,rows=1,original_head_sha256=head,source_best_epoch=1,checkpoint_sha256='b'*64)
  receipt=root/'synthetic_receipts'/(run+'.json');receipt.parent.mkdir(exist_ok=True);receipt.write_text(json.dumps(rec))
  receipts[run]={'path':str(receipt),'sha256':m.digest(receipt)};roles_by_run[run]=role_records
 stage={'completed':receipts};plan={'input':{'sha256':INPUT}}
 source=root/'synthetic_owner_stage.json';source.write_text(json.dumps(stage))
 return stage,plan,{'path':str(source),'sha256':m.digest(source)},first

def bind_trusted_owner(monkeypatch,stage,plan):
 # Upstream finite-owner verification is independently reviewed elsewhere.
 # Stub only that trust boundary; downstream role/header/head/hash checking and
 # shared CPU slot/report writer are the real code under review.
 monkeypatch.setattr(m,'read_owner_stage',lambda root,ref,mode:(copy.deepcopy(stage),copy.deepcopy(plan)))
 def proof(root,ref):
  path=Path(ref['path']);assert path.is_relative_to(root) and m.digest(path)==ref['sha256']
  return json.loads(path.read_text())
 monkeypatch.setattr(m.owner,'proof',proof)

@pytest.mark.parametrize('mode',['source_qualification','cache'])
def test_collectors_tiny_positive_exact_inventory(tmp_path,monkeypatch,mode):
 runs=m.qualification.qualification_runs() if mode=='source_qualification' else m.analysis.expected_feature_runs()
 root=tmp_path/'artifacts';stage,plan,ref,first=make_corpus(root,runs,mode);bind_trusted_owner(monkeypatch,stage,plan)
 out=m.collect_source_qualification(root,ref,0) if mode=='source_qualification' else m.collect_feature_capacity(root,ref)
 report=json.loads(Path(out['path']).read_text())
 assert len(report['results'] if mode=='source_qualification' else report['roles'])==(9 if mode=='source_qualification' else 180)
 if mode=='cache':assert m.validate_feature_capacity(root,out)==0
 print('PROBE collector_positive',mode,out['sha256'])

@pytest.mark.parametrize('mode',['source_qualification','cache'])
def test_collectors_reject_late_earlier_head_drift(tmp_path,monkeypatch,mode):
 runs=m.qualification.qualification_runs() if mode=='source_qualification' else m.analysis.expected_feature_runs()
 root=tmp_path/'artifacts';stage,plan,ref,first=make_corpus(root,runs,mode);bind_trusted_owner(monkeypatch,stage,plan)
 owned=m.owned_role;calls=[];expected=9 if mode=='source_qualification' else 180
 def late(*args,**kwargs):
  result=owned(*args,**kwargs);calls.append((args[2],args[3]))
  if len(calls)==expected:mutate_head(first[2]/'head_weight.npy')
  return result
 monkeypatch.setattr(m,'owned_role',late)
 out=m.collect_source_qualification(root,ref,0) if mode=='source_qualification' else m.collect_feature_capacity(root,ref)
 report=json.loads(Path(out['path']).read_text())
 with pytest.raises(ValueError,match='head bytes changed'):owned(root,INPUT,first[0],first[1],first[3],content_checksums=mode=='source_qualification')
 print('PROBE collector_late_drift',mode,json.dumps({'status':report['status'],'entries_checked':len(calls),'current_first_head_invalid':True,'zero_feature_writes':report.get('future_scheduled_feature_write_bytes')}))
 assert not report['status'].startswith('PASS'),'STALE PASS: earlier original-head bytes changed before aggregate collector publication'
