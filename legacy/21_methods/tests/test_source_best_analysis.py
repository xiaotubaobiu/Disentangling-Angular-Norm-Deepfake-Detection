import copy,json,hashlib
from pathlib import Path
import pytest
from reliability_v2 import source_best_analysis as m,source_best_sink as sink

def test_exact_thirty_jobs_five_seeds_three_bones():
 jobs=m.expected_jobs();assert len(jobs)==30
 assert sum(x.startswith('D_') for x in jobs)==15 and sum(x.startswith('H_') for x in jobs)==15
 for route in ('D','H'):
  for b in m.BACKBONES:assert [m.job(route,b,s)['seed'] for s in m.SEEDS]==list(m.SEEDS)

@pytest.mark.parametrize('route,bone,seed',[('DEV','clip_vit_b16',7),('H','other',7),('D','clip_vit_b16',True),('D','clip_vit_b16',8)])
def test_unregistered_job_refused(route,bone,seed):
 with pytest.raises(ValueError):m.job(route,bone,seed)

def test_static_identity_excludes_volatile_resource_proofs():
 spec=m.job('D','clip_vit_b16',7);base={'D':{'source_best_epoch':1,'original_head_sha256':'a'*64}};markers={'D/train':'b'*64}
 one=m.identity(spec,'c'*64,base,markers);assert 'budget' not in json.dumps(one).lower() and 'slot' not in json.dumps(one).lower()
 assert one['TEST_repeats']==2000 and one['CAL_repeats']==200
 assert one!=m.identity(spec,'d'*64,base,markers)

@pytest.fixture
def plan_case(tmp_path):
 plan={'schema':m.PLAN_SCHEMA,'input_admission_sha256':'a'*64,'base_future_and_reserve_bytes':2*1024**2,'runtime_floor_bytes':100000,'global_M_temporary_bytes':1024**2,'feature_cap_bytes':1024**2,'feature_runs':list(m.expected_feature_runs()),'jobs':{k:{'cap_bytes':2*1024**2,'output':None} for k in m.expected_jobs()}}
 return tmp_path,plan

def test_full_remaining_budget_counts_every_unwritten_cap(plan_case):
 root,p=plan_case
 assert m.remaining_required(root,p)==p['base_future_and_reserve_bytes']+p['feature_cap_bytes']+30*2*1024**2

@pytest.mark.parametrize('mutation',['missingjob','extrajob','boolcap','foreignfeature','foreignoutput'])
def test_invalid_capacity_plan_refused(plan_case,mutation):
 root,p=plan_case;k=next(iter(p['jobs']))
 if mutation=='missingjob':p['jobs'].pop(k)
 elif mutation=='extrajob':p['jobs']['DEV_extra']={'cap_bytes':1,'output':None}
 elif mutation=='boolcap':p['jobs'][k]['cap_bytes']=True
 elif mutation=='foreignfeature':p['feature_runs'].append('JEV_something')
 else:p['jobs'][k]['output']=str(root/'outside')
 with pytest.raises(ValueError):m.remaining_required(root,p)

def test_consumed_bounded_output_bytes_are_not_scientific_admission(plan_case):
 root,p=plan_case;k=next(iter(p['jobs']));spec=m.job('D','clip_vit_b16',7);ident=m.identity(spec,p['input_admission_sha256'],{'D':{'parent_run_id':spec['runs']['D'],'checkpoint_selection':'source_best','input_admission_sha256':p['input_admission_sha256']}},{'D/'+r:'b'*64 for r in m.cache.ROLES['D']});out=root/'analysis/source_best_M'/hashlib.sha256(sink.typed(ident)).hexdigest()/k;p['jobs'][k]['output']=str(out)
 limits=dict(max_total_bytes=2*1024**2,max_temporary_bytes=8*1024**2,max_artifact_bytes=1024**2,max_json_bytes=1024**2,max_json_array_bytes=1024**2,max_json_cells=1000,max_bundle_members=64,max_table_rows=1000,max_decoded_bytes=1024**2)
 before=m.remaining_required(root,p)
 with sink.CompactSink(root,out,ident,inventory=sink.declared_inventory('D'),limits=limits) as s:s('training_reference.json','json',{'negative':-.2})
 assert not (out/'complete.json').exists()
 consumed=sink._usage(out)-sink._usage(out/'.staging')
 assert m.remaining_required(root,p)==before-consumed-sink._usage(out/'.staging')
 assert 'scientific' not in json.dumps(p).lower()

def test_unqualified_public_entry_refuses_before_slot_or_cache(tmp_path,monkeypatch):
 called=[]
 monkeypatch.setattr(m.cache,'prepare_context',lambda *a,**k:called.append('cache'))
 monkeypatch.setattr(m.runtime,'cpu_slot',lambda *a,**k:called.append('slot'))
 with pytest.raises((ValueError,FileNotFoundError)):
  m.execute(tmp_path,'none','0'*64,'none','0'*64,'none','0'*64,slot=0)
 assert called==[]


def test_canary_closes_resource_qualification_bootstrap_without_new_experiments():
 normal={'actual_D_transport_accepted':True,'actual_H_terminal_bound_transport_accepted':True,'independent_entry_review_passed':True,'actual_entry_resource_qualified':False,'bounded_canary_resource_plan_independently_accepted':True}
 permit={'schema':m.CANARY_SCHEMA,'job_ids':list(m.CANARY_JOBS)}
 for name in m.CANARY_JOBS:m.qualified_resources(permit,normal,name)
 with pytest.raises(ValueError):m.qualified_resources(permit,normal,'D_clip_vit_b16_seed42')
 with pytest.raises(ValueError):m.qualified_resources(dict(schema=m.PERMIT_SCHEMA,job_ids=list(m.expected_jobs())),normal,m.CANARY_JOBS[0])
 accepted=dict(normal,actual_entry_resource_qualified=True)
 m.qualified_resources(dict(schema=m.PERMIT_SCHEMA,job_ids=list(m.expected_jobs())),accepted,'D_clip_vit_b16_seed42')
 for field in ('actual_D_transport_accepted','actual_H_terminal_bound_transport_accepted','independent_entry_review_passed'):
  bad=dict(normal);bad[field]=False
  with pytest.raises(ValueError):m.qualified_resources(permit,bad,m.CANARY_JOBS[0])


def test_foreign_feature_files_never_reduce_unwritten_feature_bound(plan_case):
 root,p=plan_case;before=m.remaining_required(root,p);run=p['feature_runs'][0];d=root/'features/source_best_M'/p['input_admission_sha256']/run;d.mkdir(parents=True);(d/'foreign.bin').write_bytes(b'garbage')
 assert m.remaining_required(root,p)==before

@pytest.mark.parametrize('mutation',['wrong_input','wrong_inventory'])
def test_unrelated_sink_contract_never_credits_fixed_plan(plan_case,mutation):
 root,p=plan_case;spec=m.job('D','clip_vit_b16',7);ident=m.identity(spec,'z'*64 if mutation=='wrong_input' else p['input_admission_sha256'],{},{});out=root/'analysis/source_best_M'/hashlib.sha256(sink.typed(ident)).hexdigest()/spec['job_id'];p['jobs'][spec['job_id']]['output']=str(out)
 limits=dict(max_total_bytes=2*1024**2,max_temporary_bytes=8*1024**2,max_artifact_bytes=1024**2,max_json_bytes=1024**2,max_json_array_bytes=1024**2,max_json_cells=1000,max_bundle_members=64,max_table_rows=1000,max_decoded_bytes=1024**2)
 with sink.CompactSink(root,out,ident,inventory={'a.json':'json'},limits=limits) as out:out('a.json','json',{'x':1})
 with pytest.raises(ValueError):m.remaining_required(root,p)

def test_proof_reference_drift_always_refused(tmp_path):
 refs={}
 for name in ('qualification','capacity','cache_permit'):
  path=tmp_path/(name+'.json');path.write_text('{"scope":"SYNTHETIC"}');refs[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 m.rebind_references(tmp_path,refs)
 for name in refs:
  path=Path(refs[name]['path']);old=path.read_bytes();path.write_text('{"changed":true}')
  with pytest.raises(ValueError):m.rebind_references(tmp_path,refs)
  path.write_bytes(old)

def test_whole_package_sources_include_every_scientific_dependency():
 root=Path(m.__file__).parent;assert set(m.implementations())=={str(p) for p in root.glob('*.py')}
 for name in ('stats_h.py','scoring_d.py','bootstrap_fast.py','analysis_h.py','source_best_lease.py'):
  assert str(root/name) in m.implementations()
