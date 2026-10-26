
import copy,hashlib,json,types
from pathlib import Path
import numpy as np,pytest
from reliability_v2 import source_best_analysis as m,source_best_cache as cache,source_best_runtime as runtime,source_best_sink as sink,cache_d,cache_h
import test_source_best_cache as cache_tests
import test_source_best_stats as stats_tests

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(sink.typed(value)+b'\n');return m.sf.digest(path)
def files(root):return {str(p.relative_to(root)):(m.sf.digest(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
class Context:
 def check(self,*,full=False):self.check_calls.append(full);return self

def scenario(tmp_path,monkeypatch,route):
 assert m.sf.digest(m.__file__)=='0ec974eec85a0a5204b8804707d627fa908a80eb1a4e76cfa81729ce0a875a32'
 assert m.sf.digest(runtime.__file__)=='69b3d590d95fd3004c7c64617fbad642526b02862a97f61546d3cd375fc18742'
 fixture=tmp_path/'fixture';fixture.mkdir()
 if route=='D':
  e=cache_tests.project.__wrapped__(fixture,monkeypatch,types.SimpleNamespace(param='D'));data,selected,freeze=stats_tests.d_data(e);root=e['root'];datasets={'D':data};selections={'D':selected};write(root/'data/D/freeze.json',freeze)
 else:
  e=cache_tests.project.__wrapped__(fixture,monkeypatch,types.SimpleNamespace(param='H'));root=e['root'];datasets={};selections={}
  for history in ('B','AB'):
   datasets[history]={}
   for role,frame in e['frames'].items():
    h=np.zeros((len(frame),512),dtype=np.float32);h[:,0]=np.arange(len(frame),dtype=np.float32)/100+(0.01 if history=='AB' else 0.);w=e['model'].classifier.weight.detach().numpy().copy();b=e['model'].classifier.bias.detach().numpy().copy();copy_frame=frame.copy(deep=True);copy_frame['video_id']=copy_frame.video_uid
    for i in range(len(copy_frame)):
     path=root/'synthetic_H_kernel_inputs'/role/(str(i)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
     if not path.exists():cache_tests.Image.new('RGB',(7,9),(int(copy_frame.iloc[i].label)*130+i%8,30+i//8,40)).save(path)
     copy_frame.loc[i,'image_path']=str(path);copy_frame.loc[i,'content_sha256']=cache_tests.sha(path);copy_frame.loc[i,'rgb_sha256']=cache_tests.rgb(path);copy_frame.loc[i,'bytes']=path.stat().st_size;copy_frame.loc[i,'mtime_ns']=path.stat().st_mtime_ns
    datasets[history][role]=(copy_frame,h,w,b)
   selections[history]=stats_tests.select('H',history=history,head=cache._head_hash(w,b,512))
 spec=m.job(route,'clip_vit_b16',7);code=m.implementations();inp=root/'synthetic_public_entry_input.json'
 input_sha=write(inp,dict(schema=cache.INPUT_SCHEMA,artifact_root=str(root),cache_code_sha256=m.sf.digest(cache.__file__),source_input_scope='SELECTED_M_D6_H3_FIVE_SEEDS',target_inference_authorized=False,synthetic_fixture_only=True))
 contexts={};markers={};slot_wanted=[0]
 for label,roles in datasets.items():
  c=Context();c.root=root;c.admission_sha=input_sha;c.roles=tuple(cache.ROLES[route]);c.legacy=cache_d if route=='D' else cache_h;c.check_calls=[];c.task=dict(route=route,seed=7,backbone='clip_vit_b16',history=None if route=='D' else label,epochs=30);c.source=dict(selections[label],parent_run_id=spec['runs'][label],endpoint_id='SYNTHETIC_ONLY_'+spec['runs'][label]);dimension=roles[c.roles[0]][1].shape[1]
  c.base=dict(schema=cache.SCHEMA,artifact_root=str(root),run_id=spec['runs'][label],parent_run_id=spec['runs'][label],config_path=str(root/'synthetic_context_config.json'),route=route,backbone='clip_vit_b16',seed=7,dimension=dimension,original_head_sha256=c.source['head_sha256'],checkpoint_selection='source_best',source_best_epoch=c.source['best']['epoch'],source_best_prefix=c.source['best'],input_admission_sha256=input_sha,target_inference_authorized=False,synthetic_fixture_only=True)
  write(root/'synthetic_context_config.json',{'scope':'SYNTHETIC_ONLY'});c.tables={}
  for role,(frame,h,w,b) in roles.items():
   path=root/'synthetic_cache_inputs'/label/(role.replace('/','__')+'.parquet');path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False);c.tables[role]=dict(path=str(path),sha256=m.sf.digest(path),rows=len(frame),videos=int(frame.video_uid.nunique()))
  for role,(frame,h,w,b) in roles.items():cache.write_role(c,role,w,b,lambda start,h=h:[h[start:]])
  contexts[label]=c
  for role in c.roles:markers[label+'/'+role]=m.sf.digest(cache.role_path(c,role)/'complete.json')
 bases={label:c.base for label,c in contexts.items()};ident=m.identity(spec,input_sha,bases,markers);output=root/'analysis/source_best_M'/hashlib.sha256(sink.typed(ident)).hexdigest()/spec['job_id']
 admission=root/'synthetic_independent_cache_job.json';adsha=write(admission,dict(schema=m.ADMISSION_SCHEMA,artifact_root=str(root),status='PASS',job=spec,independent_role_and_source_best_head_validation_passed=True,input_admission_sha256=input_sha,cache_code_sha256=m.sf.digest(cache.__file__),source_contexts=bases,cache_completion_sha256=markers,scope='SYNTHETIC_PUBLIC_ENTRY_REVIEW_ONLY'))
 limits=dict(max_total_bytes=256*1024**2,max_temporary_bytes=256*1024**2,max_artifact_bytes=8*1024**2,max_json_bytes=64*1024**2,max_json_array_bytes=32*1024**2,max_json_cells=1000000,max_bundle_members=64,max_table_rows=1000000,max_decoded_bytes=64*1024**2)
 plan=root/'synthetic_fixed30_capacity.json';planvalue=dict(schema=m.PLAN_SCHEMA,input_admission_sha256=input_sha,base_future_and_reserve_bytes=50*1024**3+512*1024**2+16*1024**2,runtime_floor_bytes=50*1024**3,global_M_temporary_bytes=512*1024**2,feature_cap_bytes=64*1024**2,feature_runs=list(m.expected_feature_runs()),jobs={name:dict(cap_bytes=256*1024**2,output=str(output) if name==spec['job_id'] else None) for name in m.expected_jobs()});plansha=write(plan,planvalue)
 q=root/'synthetic_engineering_qualification.json';qvalue=dict(schema='M_STATISTIC_RESOURCE_AND_FORMAT_ACCEPTANCE_V1',status='PASS',code_sha256=code,input_admission_sha256=input_sha,actual_D_transport_accepted=True,actual_H_terminal_bound_transport_accepted=True,independent_entry_review_passed=True,actual_entry_resource_qualified=False,bounded_canary_resource_plan_independently_accepted=True,scope='SYNTHETIC_PUBLIC_ENTRY_REVIEW_ONLY');qsha=write(q,qvalue)
 gates={}
 for gate in ('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity'):
  path=root/('synthetic_cache_gate_'+gate+'.json');pin=write(path,dict(M_production_cache_admission=True,cache_code_sha256=m.sf.digest(cache.__file__),input_admission_sha256=input_sha,required_remaining_bytes=1,scope='SYNTHETIC_PUBLIC_ENTRY_REVIEW_ONLY'));gates[gate]=dict(path=str(path),sha256=pin)
 cp=root/'synthetic_cache_execution_permit.json';cpsha=write(cp,dict(schema=cache.PERMIT_SCHEMA,status='PASS_PRODUCTION_CACHE_ADMISSION',artifact_root=str(root),input_admission_sha256=input_sha,cache_code_sha256=m.sf.digest(cache.__file__),target_inference_authorized=True,four_core_comparisons_only=True,parent_run_ids=list(spec['runs'].values()),**gates,scope='SYNTHETIC_PUBLIC_ENTRY_REVIEW_ONLY'))
 permit=root/'synthetic_canary_permit.json';permitvalue=dict(schema=m.CANARY_SCHEMA,status='PASS',artifact_root=str(root),input_admission_sha256=input_sha,four_core_comparisons_only=True,all_five_seeds_required=True,code_sha256=code,job_ids=list(m.CANARY_JOBS),qualification=dict(path=str(q),sha256=qsha),capacity=dict(path=str(plan),sha256=plansha),cache_permit=dict(path=str(cp),sha256=cpsha),limits={'D':limits,'H':limits},scope='SYNTHETIC_PUBLIC_ENTRY_REVIEW_ONLY');psha=write(permit,permitvalue)
 counts={'source_context_calls':0,'cache_permit_calls':0,'load_role_calls':0,'kernel_calls':0,'slot_exclusion_checks':0,'live_pool_checks':0}
 by_run={c.base['parent_run_id']:c for c in contexts.values()}
 def prepare(r,ip,ish,run):
  assert Path(r)==root and Path(ip)==inp and ish==input_sha
  with pytest.raises(BlockingIOError):
   with runtime.cpu_slot(root,slot_wanted[0]):pass
  counts['slot_exclusion_checks']+=1;counts['source_context_calls']+=1;return by_run[run]
 original_cache_permit=cache._permit
 def cache_permit(c,path,pin):
  counts['cache_permit_calls']+=1;return original_cache_permit(c,path,pin)
 original_load=cache.load_role
 def loaded(c,role):counts['load_role_calls']+=1;return original_load(c,role)
 monkeypatch.setattr(cache,'prepare_context',prepare);monkeypatch.setattr(cache,'_permit',cache_permit);monkeypatch.setattr(cache,'load_role',loaded)
 name='d_driver' if route=='D' else 'h_driver';original_driver=getattr(m.stats,name)
 def driver(*args):
  counts['kernel_calls']+=1
  observer=runtime.TemporaryPool(root,output,limits['max_temporary_bytes'],planvalue['global_M_temporary_bytes'])
  with observer._lock():
   claims,stages,charged=observer._state();assert any(x['output']==str(output) for x in claims) and charged<=observer.total
  counts['live_pool_checks']+=1;return original_driver(*args)
 monkeypatch.setattr(m.stats,name,driver)
 return dict(root=root,route=route,spec=spec,contexts=contexts,output=output,input=inp,input_sha=input_sha,admission=admission,admission_sha=adsha,permit=permit,permit_sha=psha,qualification=q,qualification_sha=qsha,plan=plan,plan_sha=plansha,planvalue=planvalue,cache_permit=cp,cache_permit_sha=cpsha,args=(root,inp,input_sha,admission,adsha,permit,psha),counts=counts,slot_wanted=slot_wanted,driver_name=name,driver=driver)

@pytest.mark.parametrize('route',['D','H'])
def test_public_execute_resume_real_slots_pool_sink_kernel(tmp_path,monkeypatch,route):
 c=scenario(tmp_path,monkeypatch,route);before=m.remaining_required(c['root'],c['planvalue']);first=m.execute(*c['args'],slot=0);output=c['output'];assert first['scientific_final_acceptance'] is False and first['bounded_resource_canary'] is True and first['source_best_only'] is True
 assert first['output']==str(output) and len(json.loads((output/'complete.json').read_text())['receipts'])==(60 if route=='D' else 37)
 saved=files(output);counts=dict(c['counts']);assert counts['kernel_calls']==1 and counts['load_role_calls']==6 and counts['live_pool_checks']==1
 c['slot_wanted'][0]=1;second=m.execute(*c['args'],slot=1);assert second==first and saved==files(output);assert c['counts']['kernel_calls']==1 and c['counts']['load_role_calls']==6
 assert m.remaining_required(c['root'],c['planvalue'])<before
 claims=list((c['root']/'analysis/source_best_M/.resource_claims').glob('*.json'));assert sum(not p.name.endswith('.released.json') for p in claims)==2 and sum(p.name.endswith('.released.json') for p in claims)==2

@pytest.mark.parametrize('reference',['qualification','plan','cache_permit'])
def test_running_external_reference_drift_must_prevent_complete(tmp_path,monkeypatch,reference):
 c=scenario(tmp_path,monkeypatch,'H');driver=c['driver'];changed=[False]
 def mutate_before_kernel(*args):
  if not changed[0]:c[reference].write_bytes(b'{"revoked_during_synthetic_run":true}');changed[0]=True
  return driver(*args)
 monkeypatch.setattr(m.stats,c['driver_name'],mutate_before_kernel)
 with pytest.raises(ValueError,match='drift|hash|qualification|capacity|permit'):m.execute(*c['args'],slot=0)
 assert changed[0] and not (c['output']/'complete.json').exists()

def test_foreign_feature_bytes_must_not_reduce_unwritten_feature_bound(tmp_path):
 root=tmp_path;p=dict(schema=m.PLAN_SCHEMA,input_admission_sha256='a'*64,base_future_and_reserve_bytes=2*1024**2,runtime_floor_bytes=100000,global_M_temporary_bytes=1024**2,feature_cap_bytes=1024**2,feature_runs=list(m.expected_feature_runs()),jobs={name:dict(cap_bytes=1024**2,output=None) for name in m.expected_jobs()});before=m.remaining_required(root,p)
 foreign=root/'features/source_best_M'/('a'*64)/m.expected_feature_runs()[0]/'foreign_unclaimed.bin';foreign.parent.mkdir(parents=True);foreign.write_bytes(b'x'*4096)
 after=m.remaining_required(root,p)
 assert after==before,'unqualified foreign bytes incorrectly reduced remaining valid-feature writes by '+str(before-after)


def test_foreign_statistic_sink_claim_must_not_reduce_current_input_writes(tmp_path):
 root=tmp_path;name=m.expected_jobs()[0];p=dict(schema=m.PLAN_SCHEMA,input_admission_sha256='a'*64,base_future_and_reserve_bytes=2*1024**2,runtime_floor_bytes=100000,global_M_temporary_bytes=1024**2,feature_cap_bytes=1024**2,feature_runs=list(m.expected_feature_runs()),jobs={key:dict(cap_bytes=2*1024**2,output=None) for key in m.expected_jobs()});ident={'scope':'SYNTHETIC_FOREIGN_UNRELATED_SINK','input_admission_sha256':'b'*64};output=root/'analysis/source_best_M'/hashlib.sha256(sink.typed(ident)).hexdigest()/name;p['jobs'][name]['output']=str(output)
 limits=dict(max_total_bytes=2*1024**2,max_temporary_bytes=8*1024**2,max_artifact_bytes=1024**2,max_json_bytes=1024**2,max_json_array_bytes=1024**2,max_json_cells=1000,max_bundle_members=64,max_table_rows=1000,max_decoded_bytes=1024**2);before=m.remaining_required(root,p)
 with sink.CompactSink(root,output,ident,inventory={'unrelated.json':'json'},limits=limits) as emit:emit('unrelated.json','json',{'not_current_job':True});emit.finish()
 try:after=m.remaining_required(root,p)
 except ValueError as error:
  assert 'foreign statistic identity' in str(error) or 'canonical full output inventory' in str(error);return
 assert after==before,'foreign-input/wrong-inventory sink incorrectly received write-budget credit '+str(before-after)


@pytest.mark.parametrize('gate',['cache_qualification','actual_D_storage','actual_H_storage','joint_capacity'])
def test_running_nested_cache_gate_drift_must_prevent_complete(tmp_path,monkeypatch,gate):
 c=scenario(tmp_path,monkeypatch,'H');driver=c['driver'];permit=json.loads(c['cache_permit'].read_text());target=Path(permit[gate]['path']);changed=[False]
 def mutate_before_kernel(*args):
  if not changed[0]:target.write_bytes(b'{"revoked_nested_synthetic_gate":true}');changed[0]=True
  return driver(*args)
 monkeypatch.setattr(m.stats,c['driver_name'],mutate_before_kernel)
 with pytest.raises(ValueError,match='hash drift'):m.execute(*c['args'],slot=0)
 assert changed[0] and not (c['output']/'complete.json').exists()
