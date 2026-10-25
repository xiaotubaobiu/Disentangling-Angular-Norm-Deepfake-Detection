from pathlib import Path
import contextlib,hashlib,os
import numpy as np,pytest,torch
from test_source_best_cache import project
from reliability_v2 import source_best_cache as cache,source_best_gpu as m

def test_owned_cpu_kernel_matches_selected_encoder_and_old_component(project):
 e=project;c=e['context'];calls=[];result=m._extract_owned(c,'cpu',lambda full=False:calls.append(full))
 assert set(result['roles'])==set(c.roles) and result['independent_scientific_admission'] is False
 for role in c.roles:
  meta,frame,h,w,b=cache.load_role(c,role);expected=[]
  from reliability_v2 import trainer
  with torch.inference_mode():
   for x,_ in trainer.loader(frame,'clip_vit_b16',8,workers=0):expected.append(e['model'].eval().encoder(x).numpy())
  assert np.array_equal(h,np.concatenate(expected)) and meta['identity']['source_best_epoch']==1
 assert calls.count(True)>=len(c.roles)+2 and False in calls and not torch.cuda.is_initialized()

def test_owned_resume_never_builds_encoder(project,monkeypatch):
 e=project;c=e['context'];m._extract_owned(c,'cpu',lambda full=False:None)
 def forbidden(*a):raise AssertionError('completed candidate recomputed')
 monkeypatch.setattr(cache,'_build_model',forbidden);result=m._extract_owned(c,'cpu',lambda full=False:None);assert len(result['roles'])==len(c.roles)

def test_guard_revocation_after_encoder_before_commit_refuses(project,monkeypatch):
 e=project;c=e['context'];revoked=False;factory=cache._build_model
 def build(*args):
  model=factory(*args);original=model.encoder.forward
  def forward(x):
   nonlocal revoked
   result=original(x);revoked=True;return result
  model.encoder.forward=forward;return model
 monkeypatch.setattr(cache,'_build_model',build)
 def guard(full=False):
  if revoked:raise ValueError('revoked after actual encoder call')
 with pytest.raises(ValueError,match='revoked'):m._extract_owned(c,'cpu',guard)
 assert revoked and not (cache.role_path(c,c.roles[0])/'complete.json').exists()

def test_missing_production_permit_refuses_before_native_gpu(project,monkeypatch):
 e=project;monkeypatch.setattr(m.lease,'verify_worker',lambda *a,**kw:None)
 def forbidden(*a,**kw):raise AssertionError('GPU ownership attempted without production permit')
 monkeypatch.setattr(m.lease,'native_gpu',forbidden);fd=os.open(e['root']/'input.json',os.O_RDONLY)
 with pytest.raises((ValueError,FileNotFoundError)):m.extract_worker(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path=e['root']/'absent.json',permit_sha256='0'*64,phase_proof={},phase_fd=fd)
 assert not (e['root']/'features').exists()

def test_incomplete_primary_gpu_stage_refused_even_with_selected_scope(project):
 with pytest.raises((ValueError,FileNotFoundError)):m._bridge_gate(project['context'],{})


def gate_case(e):
 from reliability_v2 import trainer
 root=e['root'];source=root/'source.json';records={}
 for backbone in ('clip_vit_b16','dinov2_vit_b14','efficientnet_b0'):
  for seed in (7,42,123,2026,3407):
   records[f'D_formal_{backbone}_seed{seed}']={'route':'D'}
   for history in ('B','AB'):records[f'H_formal_{backbone}_{history}_seed{seed}']={'route':'H'}
 trainer.save_json(source,dict(records=records));a=cache._json(e['admission']);a['source_file_admission']['sha256']=m.sf.digest(source);trainer.save_json(e['admission'],a);context=cache.prepare_context(root,e['admission'],m.sf.digest(e['admission']),e['run'])
 pins={str(Path(m.__file__)):m.sf.digest(m.__file__),str(Path(m.lease.__file__)):m.sf.digest(m.lease.__file__),str(Path(cache.__file__)):m.sf.digest(cache.__file__)}
 qual=root/'synthetic_bridge_qualification.json';trainer.save_json(qual,dict(schema='M_GPU_BRIDGE_QUALIFICATION_V1',status='PASS',real_source_only_GPU_qualification_passed=True,phase_native_mutual_exclusion_verified=True,source_sha256=pins))
 ids=sorted(k for k,v in records.items() if v['route']=='H');stage=root/'reports/H_primary_cache_stage_complete.json';trainer.save_json(stage,dict(status='PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION',role_count=210,run_ids=ids,bindings={str(e['admission']):m.sf.digest(e['admission'])}))
 queue=root/'H_queue_status.json';q=dict(owner_pid=12345,active=[],stage='H_PRIMARY_CACHE_EXPORT_REQUIRES_INDEPENDENT_ROLE_ADMISSION_COMPLETE',detail={'run_ids':ids});trainer.save_json(queue,q)
 permit=dict(gpu_bridge=dict(source_sha256=pins,qualification={'path':str(qual),'sha256':m.sf.digest(qual)},primary_cache_stage={'path':str(stage),'sha256':m.sf.digest(stage)}))
 return context,permit,queue,q

def test_bridge_requires_exact_original_queue_terminal_positive_case(project):
 c,p,queue,q=gate_case(project);assert m._bridge_gate(c,p)==p['gpu_bridge']

@pytest.mark.parametrize('bad',['missing','null','false','unrelated','wrong_matrix'])
def test_review_missing_or_foreign_queue_terminal_refused(project,bad):
 from reliability_v2 import trainer
 c,p,queue,q=gate_case(project)
 if bad=='missing':del q['active']
 elif bad=='null':q['active']=None
 elif bad=='false':q['active']=False
 elif bad=='unrelated':q['stage']='UNRELATED_COMPLETE'
 else:q['detail']['run_ids']=q['detail']['run_ids'][:-1]
 trainer.save_json(queue,q)
 with pytest.raises(ValueError):m._bridge_gate(c,p)
