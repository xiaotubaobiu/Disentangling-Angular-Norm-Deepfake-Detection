"""Independent public cache permit probes. Entire corpus is /tmp synthetic evidence.
Real accepted bridge, owner-plan, complete-budget and cache._permit functions run;
no checkpoint/image/GPU/real phase/PID/process or scientific job is used.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,pytest
import test_source_best_cache_permit as original
m=original.m
GiB=1024**3

def corpus(root,monkeypatch,source_problem=None):
 root.mkdir();home=root/'proofs';home.mkdir()
 def write(name,value,path=None):
  p=path or home/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False));return {'path':str(p),'sha256':m.owner.digest(p)}
 records={run:{'route':'D' if run.startswith('D_') else 'H','synthetic_only':True} for run in m.analysis.expected_feature_runs()}
 if source_problem=='missing_D_record':records.pop('D_formal_clip_vit_b16_seed42')
 source=write('source',dict(records=records,synthetic_only=True))
 if source_problem=='wrong_source_SHA':source['sha256']='f'*64
 inp=write('input',dict(schema=m.cache.INPUT_SCHEMA,artifact_root=str(root),cache_code_sha256=m.owner.digest(m.cache.__file__),source_input_scope='SELECTED_M_D6_H3_FIVE_SEEDS',target_inference_authorized=False,source_file_admission=source,synthetic_only=True))
 pins={str(Path(x.__file__)):m.owner.digest(x.__file__) for x in (m.gpu,m.lease,m.cache)}
 ids=sorted(k for k,v in records.items() if v['route']=='H')
 stage=write('primary',dict(status='PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION',role_count=210,run_ids=ids,bindings={inp['path']:inp['sha256']},synthetic_only=True),path=root/'reports/H_primary_cache_stage_complete.json')
 queue=write('queue',dict(owner_pid=999999999,active=[],stage='H_PRIMARY_CACHE_EXPORT_REQUIRES_INDEPENDENT_ROLE_ADMISSION_COMPLETE',detail={'run_ids':ids},synthetic_only=True),path=root/'H_queue_status.json')
 qual=write('qualification',dict(schema='M_GPU_BRIDGE_QUALIFICATION_V1',status='PASS',real_source_only_GPU_qualification_passed=True,phase_native_mutual_exclusion_verified=True,source_sha256=pins,input_admission_sha256=inp['sha256'],synthetic_only=True))
 bridge=dict(qualification=qual,primary_cache_stage=stage,source_sha256=pins)
 budget_dict=dict(schema=m.analysis.PLAN_SCHEMA,input_admission_sha256=inp['sha256'],base_future_and_reserve_bytes=100*GiB,runtime_floor_bytes=50*GiB,global_M_temporary_bytes=6*GiB,feature_cap_bytes=18*GiB,feature_runs=list(m.analysis.expected_feature_runs()),jobs={j:dict(cap_bytes=GiB,output=None) for j in m.analysis.expected_jobs()})
 budget=write('budget',budget_dict);gates={}
 for name in m.GATES:
  proof=dict(gate=name,M_production_cache_admission=True,independent_acceptance=True,evidence_kind='actual',input_admission_sha256=inp['sha256'],cache_code_sha256=m.owner.digest(m.cache.__file__),synthetic_only=True)
  if name=='cache_qualification':proof.update(real_source_only_GPU_qualification_passed=True,qualification=qual,run_ids=list(m.qualification.qualification_runs()))
  elif name in ('actual_D_storage','actual_H_storage'):
   route=name[7];proof.update(route=route,run_ids=[r for r in m.qualification.qualification_runs() if r.startswith(route+'_')],qualification=qual)
  else:proof.update(initial_remaining_plan=budget,required_remaining_bytes=50*GiB,initial_required_remaining_bytes=m.analysis.remaining_required(root,budget_dict))
  gates[name]=write(name,proof)
 request=write('request',dict(schema='M_CACHE_PERMIT_REQUEST_V1',artifact_root=str(root),input=inp,gates=gates,gpu_bridge=bridge,initial_remaining_plan=budget,synthetic_only=True))
 monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=500*GiB))
 def forbidden(*args,**kwargs):raise AssertionError('no GPU/dispatcher may run in permit issuer')
 monkeypatch.setattr(m.lease,'native_gpu',forbidden);monkeypatch.setattr(m.owner,'schedule',forbidden)
 return dict(root=root,input=inp,source=source,stage=stage,queue=queue,qual=qual,gates=gates,bridge=bridge,budget=budget,request=request,write=write)

def public(p):
 return m.main(['--execute','--root',str(p['root']),'--request',p['request']['path'],'--request-sha',p['request']['sha256']])

def scope_preflight(p,result):
 plan,inp,stage=m.owner.validate_plan(p['root'],result['owner_plan']['path'],result['owner_plan']['sha256'])
 assert plan['run_ids']==list(m.analysis.expected_feature_runs()) and len(plan['run_ids'])==45 and stage==p['stage']['sha256']
 for run in plan['run_ids']:
  context=SimpleNamespace(root=p['root'],admission_sha=p['input']['sha256'],base={'parent_run_id':run})
  permit=m.cache._permit(context,result['cache_permit']['path'],result['cache_permit']['sha256'])
  assert permit['parent_run_ids']==plan['run_ids']
 with pytest.raises(ValueError,match='endpoint not authorized'):
  m.cache._permit(SimpleNamespace(root=p['root'],admission_sha=p['input']['sha256'],base={'parent_run_id':'F_unregistered'}),result['cache_permit']['path'],result['cache_permit']['sha256'])
 return plan

def test_public_issue_resume_real_owner_bridge_and_all45_cache_preflights(tmp_path,monkeypatch):
 p=corpus(tmp_path/'artifacts',monkeypatch);result=public(p);plan=scope_preflight(p,result)
 assert result['GPU_or_dispatcher_started'] is False and result['statistics_permit_issued'] is False
 before={Path(ref['path']):(m.owner.digest(ref['path']),Path(ref['path']).stat().st_mtime_ns) for ref in (result['cache_permit'],result['owner_plan'])}
 again=public(p);assert again==result;scope_preflight(p,again)
 assert all((m.owner.digest(path),path.stat().st_mtime_ns)==bound for path,bound in before.items())
 print('PUBLIC_POSITIVE',json.dumps({'fixed45':len(plan['run_ids']),'remaining_bytes':m.analysis.remaining_required(p['root'],m.owner.proof(p['root'],p['budget'])),'result':result,'resume_exact':True}))

@pytest.mark.parametrize('source_problem',['missing_D_record','wrong_source_SHA'])
def test_owner_preflight_rejection_must_precede_success_permit_publication(tmp_path,monkeypatch,source_problem):
 p=corpus(tmp_path/'artifacts',monkeypatch,source_problem)
 pattern='exact45 staged formal checkpoints required' if source_problem=='missing_D_record' else 'input/code hash drift'
 with pytest.raises(ValueError,match=pattern):public(p)
 successful=[];accepted=[]
 for path in p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json'):
  value=json.loads(path.read_text())
  if value.get('status')=='PASS_PRODUCTION_CACHE_ADMISSION':
   successful.append(str(path));ref=dict(path=str(path),sha256=m.owner.digest(path))
   c=SimpleNamespace(root=p['root'],admission_sha=p['input']['sha256'],base={'parent_run_id':m.analysis.expected_feature_runs()[0]})
   assert m.cache._permit(c,ref['path'],ref['sha256'])['target_inference_authorized'] is True;accepted.append(str(path))
 print('FAILED_OWNER_PUBLICATION',json.dumps({'problem':source_problem,'success_permits_left':successful,'real_cache_permit_accepts':accepted}))
 assert successful==[],'BLOCKER: public issuer raised owner preflight rejection but left a successful cache permit accepted by real cache._permit'

@pytest.mark.parametrize('problem',['active_queue','wrong_phase','bridge_code_drift'])
def test_public_phase_and_bridge_rejection_before_outputs(tmp_path,monkeypatch,problem):
 p=corpus(tmp_path/'artifacts',monkeypatch)
 if problem=='bridge_code_drift':
  request=m.owner.proof(p['root'],p['request']);request['gpu_bridge']['source_sha256'][str(Path(m.gpu.__file__))]='d'*64;p['request']=p['write']('changed_request',request);pattern='GPU bridge/source implementation admission changed'
 else:
  value=json.loads(Path(p['queue']['path']).read_text())
  if problem=='active_queue':value['active']=[{'synthetic':True}]
  else:value['stage']='UNRELATED_DONE'
  Path(p['queue']['path']).write_text(json.dumps(value));pattern='exact original primary queue terminal/empty active list required'
 with pytest.raises(ValueError,match=pattern):public(p)
 assert not (p['root']/'qualification/M_cache_production').exists()


def test_public_gate_drift_after_first_assemble_before_locked_check(tmp_path,monkeypatch):
 p=corpus(tmp_path/'artifacts',monkeypatch);dirs=m.runtime.sink._dirs;changed=[]
 def mutate(root,path):
  answer=dirs(root,path)
  if not changed:
   q=Path(p['gates']['actual_H_storage']['path']);q.write_text('{}');changed.append(True)
  return answer
 monkeypatch.setattr(m.runtime.sink,'_dirs',mutate)
 with pytest.raises(ValueError,match='input/code hash drift'):public(p)
 assert not list(p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json'))


def test_public_append_only_collision_preserves_foreign_proof(tmp_path,monkeypatch):
 p=corpus(tmp_path/'artifacts',monkeypatch);before=m.analysis.implementations()|{str(Path(m.__file__)):m.owner.digest(m.__file__),str(Path(m.owner.__file__)):m.owner.digest(m.owner.__file__)}
 key=hashlib.sha256(m.runtime.sink.typed(dict(input=p['input'],gates=p['gates'],bridge=p['bridge'],budget=p['budget'],source=before))).hexdigest();folder=p['root']/'qualification/M_cache_production'/p['input']['sha256']/key;m.runtime.sink._dirs(p['root'],folder);foreign=folder/'cache_permit.json';foreign.write_text('{"foreign":true}')
 with pytest.raises(ValueError,match='append-only permit namespace collision'):public(p)
 assert foreign.read_text()=='{"foreign":true}' and not (folder/'cache_owner_plan.json').exists()


def test_public_resume_after_plan_publish_interruption_preserves_permit(tmp_path,monkeypatch):
 p=corpus(tmp_path/'artifacts',monkeypatch);publish=m.runtime.publish
 def interrupt(path,value):
  if Path(path).name=='cache_owner_plan.json':raise OSError('synthetic plan publish interruption')
  return publish(path,value)
 monkeypatch.setattr(m.runtime,'publish',interrupt)
 with pytest.raises(OSError,match='synthetic plan publish interruption'):public(p)
 permits=list(p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json'));assert len(permits)==1
 before=m.owner.digest(permits[0]),permits[0].stat().st_mtime_ns
 monkeypatch.setattr(m.runtime,'publish',publish);result=public(p);scope_preflight(p,result)
 assert before==(m.owner.digest(permits[0]),permits[0].stat().st_mtime_ns)
