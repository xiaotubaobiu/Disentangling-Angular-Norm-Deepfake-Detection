import importlib.util,sys,copy,json
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/'tools/source_best_cache_permit.py';spec=importlib.util.spec_from_file_location('selected_m_permit_under_test',P);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

@pytest.fixture
def packet(tmp_path,monkeypatch):
 root=tmp_path;folder=root/'proofs';folder.mkdir();input_sha='a'*64
 def write(name,value):
  p=folder/(name+'.json');p.write_text(json.dumps(value));return {'path':str(p),'sha256':m.owner.digest(p)}
 source=write('source',{'records':{r:{'route':r[0]} for r in m.analysis.expected_feature_runs()}})
 inp=write('input',{'source_file_admission':source,'schema':m.cache.INPUT_SCHEMA,'artifact_root':str(root),'cache_code_sha256':m.owner.digest(m.cache.__file__),'target_inference_authorized':False,'source_input_scope':'SELECTED_M_D6_H3_FIVE_SEEDS'})
 input_sha=inp['sha256'];plan={'schema':m.analysis.PLAN_SCHEMA,'input_admission_sha256':input_sha,'base_future_and_reserve_bytes':100*1024**3,'runtime_floor_bytes':50*1024**3,'global_M_temporary_bytes':6*1024**3,'feature_cap_bytes':18*1024**3,'feature_runs':list(m.analysis.expected_feature_runs()),'jobs':{j:{'cap_bytes':1024**3,'output':None} for j in m.analysis.expected_jobs()}}
 budget=write('budget',plan);bridge=write('bridge',{'synthetic_test_only':True});primary=write('primary',{'synthetic_test_only':True})
 gates={}
 for name in m.GATES:
  proof={'M_production_cache_admission':True,'independent_acceptance':True,'input_admission_sha256':input_sha,'cache_code_sha256':m.owner.digest(m.cache.__file__),'evidence_kind':'actual','gate':name}
  if name=='cache_qualification':proof.update(real_source_only_GPU_qualification_passed=True,qualification=bridge,run_ids=list(m.qualification.qualification_runs()))
  elif name in ('actual_D_storage','actual_H_storage'):
   route=name[7];proof.update(route=route,run_ids=[r for r in m.qualification.qualification_runs() if r.startswith(route+'_')],qualification=bridge)
  else:proof.update(initial_remaining_plan=budget,required_remaining_bytes=50*1024**3,initial_required_remaining_bytes=m.analysis.remaining_required(root,plan))
  gates[name]=write(name,proof)
 seen=[];monkeypatch.setattr(m.qualification,'closed_primary_stage',lambda root,inp:'e'*64);monkeypatch.setattr(m.gpu,'_bridge_gate',lambda context,permit,full=True:seen.append((context.root,permit['gpu_bridge'],full)))
 monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Space',(),{'free':500*1024**3})())
 return root,inp,gates,bridge,primary,budget,seen,write

def call(p):
 root,inp,gates,bridge,primary,budget,seen,write=p;return m.assemble(root,inp,gates,{'qualification':bridge,'primary_cache_stage':primary,'source_sha256':{str(Path(x.__file__)):m.owner.digest(x.__file__) for x in (m.gpu,m.lease,m.cache)}},budget)

def test_exact_four_independent_gates_generate_only_fixed45_cache_and_owner(packet):
 permit,plan=call(packet);assert permit['target_inference_authorized'] is True and len(permit['parent_run_ids'])==45 and permit['four_core_comparisons_only'] is True
 assert plan['mode']=='cache' and plan['run_ids']==list(m.analysis.expected_feature_runs()) and not plan['resume_receipts'] and len(packet[6])==1
 assert not list(packet[0].glob('qualification/*'))

@pytest.mark.parametrize('mutation',['missing_gate','not_independent','candidate','wrong_input','wrong_code','wrong_D_group','missing_qualification','low_floor','low_initial','wrong_budget','budget_input','budget_missing_job','low_free'])
def test_unaccepted_gate_or_budget_never_creates_outputs(packet,mutation,monkeypatch):
 root,inp,gates,bridge,primary,budget,seen,write=packet
 if mutation=='missing_gate':gates.pop('actual_H_storage')
 elif mutation=='low_free':monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Space',(),{'free':60*1024**3})())
 elif mutation.startswith('budget_'):
  d=json.loads(Path(budget['path']).read_text())
  if mutation=='budget_input':d['input_admission_sha256']='f'*64
  else:d['jobs'].pop(next(iter(d['jobs'])))
  new=write('changedbudget',d);budget.clear();budget.update(new)
 else:
  name='actual_D_storage' if mutation=='wrong_D_group' else 'joint_capacity' if mutation in ('low_floor','low_initial','wrong_budget') else 'cache_qualification';d=json.loads(Path(gates[name]['path']).read_text())
  if mutation=='not_independent':d['independent_acceptance']=False
  elif mutation=='candidate':d['evidence_kind']='candidate'
  elif mutation=='wrong_input':d['input_admission_sha256']='b'*64
  elif mutation=='wrong_code':d['cache_code_sha256']='c'*64
  elif mutation=='wrong_D_group':d['run_ids']=d['run_ids'][:-1]
  elif mutation=='missing_qualification':d.pop('qualification')
  elif mutation=='low_floor':d['required_remaining_bytes']=1
  elif mutation=='low_initial':d['initial_required_remaining_bytes']=1
  elif mutation=='wrong_budget':d['initial_remaining_plan']={'path':'other','sha256':'d'*64}
  gates[name]=write('changed_'+name,d)
 with pytest.raises((ValueError,KeyError)):call(packet)
 assert not (root/'qualification').exists()

def test_bridge_rejection_propagates_before_any_permit_write(packet,monkeypatch):
 def refused(*a,**k):raise ValueError('original GPU phase incomplete')
 monkeypatch.setattr(m.gpu,'_bridge_gate',refused)
 with pytest.raises(ValueError,match='original GPU phase incomplete'):call(packet)
 assert not (packet[0]/'qualification').exists()


@pytest.mark.parametrize('mutation',['missing_D','source_SHA','source_route'])
def test_source_admission_is_checked_before_any_success_documents(packet,mutation):
 root,inp,gates,bridge,primary,budget,seen,write=packet;data=json.loads(Path(inp['path']).read_text());source_ref=data['source_file_admission'];source=json.loads(Path(source_ref['path']).read_text())
 if mutation=='missing_D':source['records'].pop('D_formal_clip_vit_b16_seed42')
 elif mutation=='source_route':source['records']['D_formal_clip_vit_b16_seed42']['route']='H'
 else:source_ref['sha256']='0'*64
 if mutation!='source_SHA':
  source_ref=write('changed_source',source);data['source_file_admission']=source_ref
 new_input=write('changed_input',data);new_sha=new_input['sha256'];inp.clear();inp.update(new_input)
 # All supplied downstream evidence deliberately matches the new capsule, so
 # only validating its embedded original-source admission can reject this.
 for name in gates:
  d=json.loads(Path(gates[name]['path']).read_text());d['input_admission_sha256']=new_sha;gates[name]=write('source_case_'+name,d)
 bp=json.loads(Path(budget['path']).read_text());bp['input_admission_sha256']=new_sha;new_budget=write('source_case_budget',bp);budget.clear();budget.update(new_budget);d=json.loads(Path(gates['joint_capacity']['path']).read_text());d['initial_remaining_plan']=budget;gates['joint_capacity']=write('source_case_joint',d)
 with pytest.raises(ValueError):call(packet)
 assert not (root/'qualification').exists()

def test_strict_original_owner_phase_gate_runs_before_publication(packet,monkeypatch):
 def refused(*a):raise ValueError('strict original phase refuses')
 monkeypatch.setattr(m.qualification,'closed_primary_stage',refused)
 with pytest.raises(ValueError,match='strict original phase refuses'):call(packet)
 assert not (packet[0]/'qualification').exists()
