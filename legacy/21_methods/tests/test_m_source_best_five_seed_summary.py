"""Synthetic-only M five-seed summary tests; no real slot/result/source loads."""
import copy,importlib.util,math,statistics
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

MODULE=Path(__file__).parents[1]/'tools/m_source_best_five_seed_summary.py'
CACHE=None

def api():
 global CACHE
 if CACHE is None:
  if not MODULE.exists():pytest.fail('Missing selected-M five-seed summary tool')
  spec=importlib.util.spec_from_file_location('m_summary_under_test',MODULE);CACHE=importlib.util.module_from_spec(spec);spec.loader.exec_module(CACHE)
 return CACHE

def job(route='D',seed=7,bone='clip_vit_b16'):
 runs={'D':f'D_formal_{bone}_seed{seed}'} if route=='D' else {h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')}
 return dict(job_id=f'{route}_{bone}_seed{seed}',route=route,backbone=bone,seed=seed,runs=runs)

def prefix(spec):
 return {h:dict(seed=spec['seed'],epoch=i+1,attempts=(i+1)*10,successful_updates=(i+1)*10-2,AMP_skips=2) for i,h in enumerate(spec['runs'])}

def accepted(spec):
 m=api();numeric=dict(accepted_for_selected_core=True,blockers_for_selected_core=[])
 if spec['route']=='D':numeric.update(schema='M_SOURCE_BEST_D_INDEPENDENT_NUMERIC_CORE_V1',status='PASS_SELECTED_D_CORE',scientific_final_acceptance=False)
 else:numeric.update(status='NUMERIC_SELECTED_NATIVE_B_CORE_RECOMPUTED')
 return dict(schema='M_INDEPENDENT_SAVED_RESULT_AUDIT_V1',status='SELECTED_CORE_METADATA_AND_NUMERIC_ACCEPTED',audited_scope='SELECTED_FOUR_CORE_ONLY',accepted_for_selected_core=True,scientific_final_acceptance=False,all_five_seed_conclusions_available=False,job=spec,core_endpoints=m.audit.CORE_ENDPOINTS,layers=m.audit.LAYERS,selected_prefix=prefix(spec),external_complete_sha256='c'*64,artifact_count=60 if spec['route']=='D' else 37,all_declared_artifacts_SHA_and_codec_verified=True,metadata_status='ACTUAL_BOUND_SIX_ROLE_PRODUCER_EXIT0_AND_RESOURCE_EVIDENCE_RECHECKED',numeric=numeric,fixed_SHA_rechecked_before_return=True,model_or_GPU_inference_started=False)

class Artifacts(dict):
 def __init__(self,values):
  super().__init__(values);self.output=Path('/synthetic/M');self.complete_sha256='c'*64
  self.records={name:dict(files={name+'.encoded':'d'*64}) for name in values}

def normal(raw=True):
 groups=[f'g{i}' for i in range(20)];draws=[]
 for i in range(2000):
  value=.25 if raw else (.75 if i<1000 else 0.)
  draws.append(dict(repeat=i,pooled_valid=True,pooled_auc=value,group_draw_counts={groups[i%20]:20}))
 return dict(source_groups=groups,n_source_groups=20,repeats=2000,seed=20261023,min_groups=20,max_group_fraction=.5,largest_group_fraction=.05,status='ok',draws=draws,point_estimate={'pooled_auc':.6 if raw else .4})

def d_case(seed=7):
 spec=job('D',seed);r=accepted(spec);values={}
 cells=[]
 for a in ('mean_logit','mean_probability'):
  for g,raw,l2 in [('correct',.6,.4),('collision',.5,.8)]:
   for rep,value in [('raw',raw),('l2',l2)]:cells.append(dict(grouping=g,representation=rep,aggregation=a,pooled_auc=value))
 values['target_full/metrics.csv']=pd.DataFrame(cells)
 values['target_full/effects.csv']=pd.DataFrame([dict(seed=seed,aggregation=a,delta_correct=-.2,delta_collision=.3,interaction=.5,sign_correct=-1,sign_collision=1,reversal=True,valid=True,tolerance=1e-10) for a in ('mean_logit','mean_probability')])
 values['target_full/normal_bootstrap_raw_mean_logit.json']=normal(True);values['target_full/normal_bootstrap_l2_mean_logit.json']=normal(False)
 values['target_full/closure_bootstrap.json']={'status':'insufficient_closure_groups','intervals':None}
 r['numeric']['bootstrap_states']=[{'artifact':'target_full/closure_bootstrap.json','status':'insufficient_closure_groups'}]
 for rep in ('raw','l2'):
  for agg in ('mean_logit','mean_probability'):
   folder=f'rq4/{rep}_{agg}'
   cal=[dict(calibration_kind='default',budget=0,repeat=-1,status='ok'),dict(calibration_kind='source_real',budget=20,repeat=-1,status='ok')]
   cal += [dict(calibration_kind='target_real',budget=b,repeat=i,status='ok') for i in range(200) for b in (10,20,40,80)]
   values[folder+'/calibration.csv']=pd.DataFrame(cal)
   values[folder+'/primary_n20_vs_source_bootstrap.json']=dict(status='insufficient_source_groups',calibration_budget=20,calibration_repeats=200,point_estimates={'delta_fpr':-.1,'delta_tpr':-.2,'delta_pooled_ba':-.05},intervals=None)
 return spec,r,Artifacts(values)

def h_case(seed=7):
 spec=job('H',seed);r=accepted(spec);values={};cohorts=[]
 metrics=('pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba')
 for agg in ('mean_logit','mean_probability'):
  for domain in ('SOURCE_TEST','TARGET_TEST'):
   points={f'{kind}:comparison:H10-H00:{metric}':(-.1 if domain=='SOURCE_TEST' else .2) for kind in ('default','source_real') for metric in metrics}
   cohorts.append(dict(domain=domain,aggregation=agg,cohort='all_B',status='ok',draws_checked=2000,independent_point_estimates=points))
   values[f'{agg}/{domain}/all_B/conditional_bootstrap.json']={}
 r['numeric']['cohorts']=cohorts
 return spec,r,Artifacts(values)

def test_default_prepares_without_execution(monkeypatch):
 m=api();monkeypatch.setattr(m,'execute_summary',lambda *a,**k:pytest.fail('No execution from default'))
 result=m.main([]);assert result['status']=='PREPARE_ONLY' and result['summary_executed'] is False

def test_five_registered_models_sample_sd_is_ddof1_and_negative_is_retained():
 m=api();result=m.five_seed_statistics(dict(zip((7,42,123,2026,3407),(-2.,-1.,0.,1.,2.))))
 assert result['mean']==0 and result['sample_sd']==pytest.approx(math.sqrt(2.5)) and result['sample_sd']!=pytest.approx(math.sqrt(2.))
 assert result['registered_n']==5 and result['finite_n']==5 and result['status']=='AVAILABLE'

@pytest.mark.parametrize('values',[(None,-1.,0.,1.,2.),(None,)*5,(float('nan'),-1.,0.,1.,2.)])
def test_any_NA_makes_five_seed_mean_and_sd_unavailable(values):
 result=api().five_seed_statistics(dict(zip((7,42,123,2026,3407),values)))
 assert result['mean'] is None and result['sample_sd'] is None and result['status']=='UNAVAILABLE_NA_REGISTERED_SEED'

@pytest.mark.parametrize('keys',[(7,42,123,2026),(7,42,123,2026,99)])
def test_missing_or_wrong_seed_never_becomes_four_model_mean(keys):
 with pytest.raises(ValueError):api().five_seed_statistics({k:0. for k in keys})

@pytest.mark.parametrize('value',[True,float('inf'),float('-inf')])
def test_boolean_or_nonfinite_non_NA_effect_rejected(value):
 values=dict.fromkeys((7,42,123,2026,3407),0.);values[7]=value
 with pytest.raises(ValueError):api().five_seed_statistics(values)

def test_RQ2_closed_form_aligned_frozen_draw_delta_CI_and_independent_quantile():
 raw,l2=normal(True),normal(False);result=api().normal_delta_interval(raw,l2,-.2)
 assert result['lower']==-.25 and result['upper']==.5 and result['valid_repeats']==2000
 assert result['derivation']=='downstream arithmetic from aligned frozen normal draws'
 assert result['lower']==pd.Series([.5]*1000+[-.25]*1000).quantile(.025)
 assert result['upper']==pd.Series([.5]*1000+[-.25]*1000).quantile(.975)

@pytest.mark.parametrize('mutation',['groups','repeat','counts'])
def test_RQ2_unaligned_draws_are_explicit_unavailable_not_closure_replacement(mutation):
 raw,l2=normal(True),normal(False)
 if mutation=='groups':l2['source_groups'].reverse()
 elif mutation=='repeat':l2['draws'][4]['repeat']=5
 else:l2['draws'][4]['group_draw_counts']={'g0':20}
 result=api().normal_delta_interval(raw,l2,-.2)
 assert result['status']=='UNAVAILABLE_UNALIGNED_FROZEN_NORMAL_DRAWS' and result['lower'] is None and result['upper'] is None

@pytest.mark.parametrize('mutation',['seed','repeats','bool_count','float_count'])
def test_RQ2_wrong_protocol_or_invalid_integer_multiplicity_rejected(mutation):
 raw,l2=normal(True),normal(False)
 if mutation=='seed':l2['seed']=7
 elif mutation=='repeats':l2['repeats']=1999
 elif mutation=='bool_count':l2['draws'][0]['group_draw_counts']={'g0':True}
 else:l2['draws'][0]['group_draw_counts']={'g0':20.}
 with pytest.raises(ValueError):api().normal_delta_interval(raw,l2,-.2)

def test_RQ2_invalid_draws_preserve_valid_count_and_NA_rule():
 raw,l2=normal(True),normal(False)
 for x in (raw,l2):
  for row in x['draws'][:20]:row['pooled_valid']=False;row['pooled_auc']=None
 result=api().normal_delta_interval(raw,l2,-.2)
 assert result['valid_repeats']==1980 and result['invalid_repeats']==20 and result['lower']==-.25 and result['upper']==.5

def test_frozen_D_core_extraction_sign_reversal_and_layers():
 spec,report,reader=d_case();rows=api().extract_core(spec,report,reader)
 assert len(rows)==15 and sum(r['core']=='RQ2' for r in rows)==1 and sum(r['core']=='RQ3' for r in rows)==2 and sum(r['core']=='RQ4' for r in rows)==12
 rq2=next(r for r in rows if r['core']=='RQ2');assert rq2['estimate']==pytest.approx(-.2) and rq2['TEST_ci']['valid_repeats']==2000
 rq3=next(r for r in rows if r['core']=='RQ3');assert rq3['estimate']==pytest.approx(.5) and rq3['reversal'] is True and rq3['TEST_ci']['status']=='insufficient_closure_groups'
 rq4=next(r for r in rows if r['core']=='RQ4');assert rq4['CAL_reference']['requested_subsets']==200 and rq4['layer']=='fixed_training_seed_point'
 assert rq2['TEST_reference']['derivation']=='downstream arithmetic from aligned frozen normal draws'

def test_frozen_H_core_has_native_primary_FF_validation_and_sensitivity():
 spec,report,reader=h_case();rows=api().extract_core(spec,report,reader)
 assert len(rows)==56 and {r['domain'] for r in rows}=={'DFD_TARGET_TEST','FF_SOURCE_TEST'}
 assert all(r['comparison']=='H10-H00' and r['representation']=='raw' for r in rows)
 assert any(r['estimate']<0 for r in rows) and all(r['TEST_reference']['layer']=='conditional_TEST_2000' for r in rows)

def test_RQ4_unavailable_n20_is_not_replaced_with_another_budget():
 spec,report,reader=d_case();folder='rq4/raw_mean_logit'
 reader[folder+'/primary_n20_vs_source_bootstrap.json']={'status':'UNAVAILABLE_CALIBRATION_BUDGET','calibration_budget':20,'available_target_real_videos':12,'intervals':None}
 reader[folder+'/calibration.csv'].loc[reader[folder+'/calibration.csv'].budget.eq(20)&reader[folder+'/calibration.csv'].calibration_kind.eq('target_real'),'status']='unavailable'
 rows=api().extract_core(spec,report,reader)
 chosen=[r for r in rows if r['core']=='RQ4' and r['aggregation']=='mean_logit' and r['representation']=='raw']
 assert len(chosen)==3 and all(r['estimate'] is None for r in chosen)

@pytest.mark.parametrize('mutation',['accepted','complete','LAST','prefix','bone'])
def test_reject_unaccepted_wrong_source_complete_prefix_or_backbone(mutation):
 spec=job();report=accepted(spec);wanted=prefix(spec)
 if mutation=='accepted':report['accepted_for_selected_core']=False
 elif mutation=='complete':report['all_declared_artifacts_SHA_and_codec_verified']=False
 elif mutation=='LAST':report['selected_prefix']['D']['checkpoint_selection']='last'
 elif mutation=='prefix':wanted['D']['epoch']+=1
 else:report['job']=job(bone='dinov2_vit_b14')
 with pytest.raises(ValueError):api().validate_audit_report(report,spec,wanted)

def test_D_inconsistent_saved_effect_not_silently_selected():
 spec,report,reader=d_case();reader['target_full/effects.csv'].loc[0,'interaction']=.99
 with pytest.raises(ValueError):api().extract_core(spec,report,reader)

def test_summary_all5_rows_per_endpoint_and_no_CAL_TEST_IID_pooling():
 m=api();rows=[]
 for seed in (7,42,123,2026,3407):
  for make in (d_case,h_case):
   spec,report,reader=make(seed);rows.extend(m.extract_core(spec,report,reader))
 summary=m.summarize_rows(rows,'clip_vit_b16')
 assert len(rows)==355 and len(summary)==71 and all(r['registered_n']==5 for r in summary)
 bad=copy.deepcopy(rows);bad[0]['layer']='CAL_200_as_models'
 with pytest.raises(ValueError):m.summarize_rows(bad,'clip_vit_b16')
 with pytest.raises(ValueError):m.summarize_rows(rows+rows[:1],'clip_vit_b16')
 with pytest.raises(ValueError):m.summarize_rows([r for r in rows if r['seed']!=7],'clip_vit_b16')


def tiny_ref(m,path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(m.audit.typed(value));return dict(path=str(path),sha256=m.audit.sha(path))

def summary_request(tmp_path,monkeypatch):
 m=api();root=tmp_path/'root';root.mkdir();monkeypatch.setattr(m,'ROOT',root)
 deps={};monkeypatch.setattr(m,'dependencies',lambda:deps)
 refs=[]
 for route in ('D','H'):
  for seed in m.SEEDS:
   spec=job(route,seed);refs.append(dict(job=spec,audit={'path':str(root/f'{spec["job_id"]}.audit.json'),'sha256':'a'*64},audit_request={'path':str(root/f'{spec["job_id"]}.request.json'),'sha256':'b'*64},selected_prefix=prefix(spec)))
 base=dict(schema=m.REQUEST_SCHEMA,root=str(root),backbone='clip_vit_b16',audits=refs,dependencies_sha256=deps)
 identity=m.inputs_identity(base)
 capacity=tiny_ref(m,root/'capacity.json',dict(input_admission_sha256='e'*64,runtime_floor_bytes=50*1024**3))
 budget=tiny_ref(m,root/'budget.json',dict(schema=m.BUDGET_SCHEMA,root=str(root),inputs_sha256=identity,capacity_plan=capacity,write_accounting='ADDITIONAL_TO_PLAN',max_output_bytes=4*1024**2,temporary_peak_bytes=1024**2,max_peak_RSS_bytes=1024**3,address_space_limit_bytes=32*1024**3,runtime_floor_bytes=50*1024**3))
 base['budget']=budget;base['output']=str(m.canonical_output(base));request=tiny_ref(m,root/'summary_request.json',base)
 return m,root,base,request

def test_SHA_bound_ten_job_request_and_canonical_identity(tmp_path,monkeypatch):
 m,root,q,ref=summary_request(tmp_path,monkeypatch);loaded,pins,budget=m.load_summary_request(ref['path'],ref['sha256'])
 assert loaded==q and budget['inputs_sha256']==m.inputs_identity(q)
 assert str(root/'budget.json') in pins.expected
 assert m.canonical_output(q)==Path(q['output'])

@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_seed','LAST','DEV','bone','prefix','dependency','escape','budget_identity','capacity_sha'])
def test_request_rejects_wrong_jobs_or_external_binding(tmp_path,monkeypatch,mutation):
 m,root,q,ref=summary_request(tmp_path,monkeypatch)
 if mutation=='missing':q['audits'].pop()
 elif mutation=='duplicate':q['audits'][-1]=q['audits'][0]
 elif mutation=='wrong_seed':q['audits'][0]['job']['seed']=99
 elif mutation=='LAST':q['audits'][0]['job']['runs']['D']+='LAST'
 elif mutation=='DEV':q['audits'][0]['job']['job_id']='DEV'
 elif mutation=='bone':q['audits'][0]['job']=job(bone='dinov2_vit_b14')
 elif mutation=='prefix':q['audits'][0]['selected_prefix']['D']['AMP_skips']=99
 elif mutation=='dependency':q['dependencies_sha256']={'wrong':'0'*64}
 elif mutation=='escape':q['output']=str(root.parent/'escaped')
 else:
  b=m.audit.json_read(q['budget']['path'])
  if mutation=='budget_identity':b['inputs_sha256']='0'*64
  else:b['capacity_plan']['sha256']='0'*64
  q['budget']=tiny_ref(m,root/'budget.json',b)
 ref=tiny_ref(m,root/'summary_request.json',q)
 with pytest.raises(ValueError):m.load_summary_request(ref['path'],ref['sha256'])

def test_request_external_SHA_and_symlink_escape_rejected(tmp_path,monkeypatch):
 m,root,q,ref=summary_request(tmp_path,monkeypatch)
 with pytest.raises(ValueError):m.load_summary_request(ref['path'],'f'*64)
 link=root/'link';link.symlink_to(root/'summary_request.json')
 with pytest.raises(ValueError):m.load_summary_request(link,ref['sha256'])

def test_fresh_joint_capacity_adds_summary_caps_and_keeps_fiftyGiB(tmp_path,monkeypatch):
 m,root,q,ref=summary_request(tmp_path,monkeypatch);_,pins,b=m.load_summary_request(ref['path'],ref['sha256'])
 monkeypatch.setattr(m,'_remaining_required',lambda root,plan:100*1024**3)
 monkeypatch.setattr(m,'_free_bytes',lambda root:100*1024**3+b['max_output_bytes']+b['temporary_peak_bytes'])
 receipt=m.resource_gate(q,b,pins,'e'*64);assert receipt['remaining_required_including_summary_bytes']==100*1024**3+5*1024**2
 monkeypatch.setattr(m,'_free_bytes',lambda root:100*1024**3+5*1024**2-1)
 with pytest.raises(ValueError):m.resource_gate(q,b,pins,'e'*64)
 with pytest.raises(ValueError):m.resource_gate(q,b,pins,'f'*64)

def execution_fixture(tmp_path,monkeypatch):
 m,root,q,ref=summary_request(tmp_path,monkeypatch);events=[]
 from contextlib import contextmanager
 @contextmanager
 def slot(root,index):
  assert index in (0,1);events.append('slot_enter');yield;events.append('slot_exit')
 monkeypatch.setattr(m,'_actual_slot',slot);monkeypatch.setattr(m.resource,'getrlimit',lambda kind:(32*1024**3,32*1024**3))
 monkeypatch.setattr(m,'_remaining_required',lambda root,plan:100*1024**3);monkeypatch.setattr(m,'_free_bytes',lambda root:200*1024**3)
 for k,v in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items():monkeypatch.setenv(k,v)
 def load(item,pins):
  assert events[-1]=='slot_enter' and events.count('slot_enter')==events.count('slot_exit')+1;events.append('read');events.pop()
  spec,r,reader=(d_case if item['job']['route']=='D' else h_case)(item['job']['seed'])
  reader.check=lambda:None
  return r,reader,'e'*64
 monkeypatch.setattr(m,'load_accepted_item',load)
 return m,root,q,ref,events

def test_append_only_full_synthetic_summary_and_csv_readback(tmp_path,monkeypatch):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch)
 out=m.execute_summary(ref['path'],ref['sha256'],0)
 assert events==['slot_enter','slot_exit'] and out['status']=='COMPLETE_DESCRIPTIVE_SOURCE_BEST_FIVE_SEED_SUMMARY'
 assert out['scientific_final_acceptance'] is False and out['training_seed_statistics']['ddof']==1
 output=Path(q['output']);assert {p.name for p in output.iterdir()}=={'per_seed_core.csv','five_seed_core.csv','manifest.json','complete.json'}
 table=pd.read_csv(output/'per_seed_core.csv');summary=pd.read_csv(output/'five_seed_core.csv')
 assert len(table)==355 and len(summary)==71 and (table.estimate<0).any()
 assert all(summary.registered_n==5) and all(summary.ddof==1)
 original={p.name:p.read_bytes() for p in output.iterdir()}
 again=m.execute_summary(ref['path'],ref['sha256'],1)
 assert again['complete_sha256']==out['complete_sha256'] and {p.name:p.read_bytes() for p in output.iterdir()}==original

@pytest.mark.parametrize('mutation',['CUDA','thread','AS','capacity','partial','foreign'])
def test_actual_execution_resource_or_namespace_refusal(tmp_path,monkeypatch,mutation):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch)
 if mutation=='CUDA':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 elif mutation=='thread':monkeypatch.setenv('OMP_NUM_THREADS','8')
 elif mutation=='AS':monkeypatch.setattr(m.resource,'getrlimit',lambda k:(-1,-1))
 elif mutation=='capacity':monkeypatch.setattr(m,'_free_bytes',lambda root:0)
 else:
  out=Path(q['output']);out.mkdir(parents=True)
  if mutation=='foreign':(out/'foreign').write_text('preserve')
 with pytest.raises(ValueError):m.execute_summary(ref['path'],ref['sha256'],0)
 assert not (Path(q['output'])/'complete.json').exists()
 if mutation=='foreign':assert (Path(q['output'])/'foreign').read_text()=='preserve'

def test_pin_drift_before_publication_prevents_completion(tmp_path,monkeypatch):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch);original=m._csv_bytes
 def drift(rows):
  result=original(rows);Path(ref['path']).write_text('{}');return result
 monkeypatch.setattr(m,'_csv_bytes',drift)
 with pytest.raises(ValueError):m.execute_summary(ref['path'],ref['sha256'],0)
 assert not (Path(q['output'])/'complete.json').exists()

def test_frozen_RQ2_gate_cannot_be_marked_ok_and_pair_NA_is_retained():
 raw,l2=normal(True),normal(False)
 for obj in (raw,l2):obj['largest_group_fraction']=.6
 with pytest.raises(ValueError):api().normal_delta_interval(raw,l2,-.2)
 raw,l2=normal(True),normal(False)
 for obj in (raw,l2):
  for d in obj['draws']:d['pooled_valid']=False;d['pooled_auc']=None
 result=api().normal_delta_interval(raw,l2,-.2);assert result['status']=='UNAVAILABLE_NUMERIC_NA' and result['valid_repeats']==0

def test_count_order_alignment_and_native_validity_must_not_silently_shift():
 raw,l2=normal(True),normal(False)
 raw['draws'][0]['group_draw_counts']={'g0':10,'g1':10};l2['draws'][0]['group_draw_counts']={'g1':10,'g0':10}
 assert api().normal_delta_interval(raw,l2,-.2)['status']=='UNAVAILABLE_UNALIGNED_FROZEN_NORMAL_DRAWS'



def test_summary_inside_existing_primary_audit_pool_does_not_double_count(tmp_path,monkeypatch):
 m,root,q,ref=summary_request(tmp_path,monkeypatch)
 b=m.audit.json_read(q['budget']['path']);b['write_accounting']='INCLUDED_IN_PRIMARY_LOGS_FIGURES_AUDITS_AND_M_EXTRA';q['budget']=tiny_ref(m,root/'budget.json',b);q['output']=str(m.canonical_output(q));ref=tiny_ref(m,root/'summary_request.json',q)
 _,pins,b=m.load_summary_request(ref['path'],ref['sha256'])
 monkeypatch.setattr(m,'_remaining_required',lambda root,plan:100*1024**3);monkeypatch.setattr(m,'_free_bytes',lambda root:100*1024**3)
 receipt=m.resource_gate(q,b,pins,'e'*64)
 assert receipt['remaining_required_including_summary_bytes']==100*1024**3 and receipt['additional_summary_reservation_bytes']==0


def accepted_item_fixture(tmp_path,monkeypatch):
 """Literal accepted metadata; existing authenticated reader is a tested boundary.

 No producer,source images/weights,feature math or bootstrap oracle runs here.
 """
 m=api();root=tmp_path/'root';root.mkdir();monkeypatch.setattr(m,'ROOT',root);auditor=m._load_audit() if hasattr(m,'_load_audit') else m.audit;monkeypatch.setattr(auditor,'ROOT',root)
 spec=job('H');report=accepted(spec);refs={};initial={}
 for name in m.audit.PROOF_KEYS:
  ref=tiny_ref(m,root/'proofs'/f'{name}.json',{'name':name});refs[name]=ref;initial[ref['path']]=ref['sha256']
 contexts={}
 for h in spec['runs']:
  contexts[h]=dict(schema='M_SOURCE_BEST_RAW_FP32_CACHE_V1',run_id=spec['runs'][h],checkpoint_selection='source_best',route='H',seed=7,backbone=spec['backbone'],parent_run_id=spec['runs'][h],epochs=30,source_best_epoch=prefix(spec)[h]['epoch'],source_best_prefix={k:v for k,v in prefix(spec)[h].items() if k!='seed'})
 ident=dict(schema='M_STATISTIC_STATIC_IDENTITY_V1',scope='SELECTED_FOUR_CORE_SOURCE_BEST_DESCRIPTIVE_SENSITIVITY',job=spec,checkpoint_selection='source_best',CAL_repeats=200,TEST_repeats=2000,resampling_seed=20261023,update_matched_primary_claim=False,code_sha256=m.audit.producer_package_codes(),input_admission_sha256=refs['input_admission']['sha256'],source_contexts=contexts,cache_completion_sha256={})
 for h,run in spec['runs'].items():
  for role in ('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST'):
   folder=root/'features/source_best_M'/ident['input_admission_sha256']/run/role;folder.mkdir(parents=True)
   manifest=root/'data/H'/(role+'.parquet');manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_bytes(b'literal manifest metadata')
   (folder/'manifest.parquet').write_bytes(manifest.read_bytes())
   manifest_sha=m.audit.sha(manifest)
   for file in ('head_weight.npy','head_bias.npy','chunk_000000.npy'):(folder/file).write_bytes(b'literal accepted bytes')
   chunk=dict(file='chunk_000000.npy',start=0,rows=2,dimension=512,dtype='float32',sha256=m.audit.sha(folder/'chunk_000000.npy'))
   tiny_ref(m,folder/'chunk_000000.json',chunk)
   identity=dict(contexts[h],role=role,source_manifest_path=str(manifest),source_manifest_sha256=manifest_sha)
   marker=tiny_ref(m,folder/'complete.json',dict(status='CACHE_COMPLETE',precision='FP32',autocast=False,identity=identity,head=dict(weight_file='head_weight.npy',bias_file='head_bias.npy',weight_sha256=m.audit.sha(folder/'head_weight.npy'),bias_sha256=m.audit.sha(folder/'head_bias.npy')),chunks=[chunk]))
   ident['cache_completion_sha256'][h+'/'+role]=marker['sha256']
   for f in folder.iterdir():initial[str(f)]=m.audit.sha(f)
   initial[str(manifest)]=manifest_sha
 output=root/'analysis/source_best_M'/__import__('hashlib').sha256(m.audit.typed(ident)).hexdigest()/spec['job_id'];output.mkdir(parents=True)
 payload=tiny_ref(m,output/'literal.json',{'accepted':True});result_pins=m.audit.Pins();result_pins.add(payload['path'],payload['sha256'])
 limitations=dict(checkpoint_selection='source_best',update_matched_primary_claim=False,source_only_selected_prefix=prefix(spec),source_best_epochs={h:v['epoch'] for h,v in prefix(spec).items()})
 reader=Artifacts({'limitations.json':limitations});reader.output=output;reader.identity=ident;reader.pins=result_pins;reader.check=result_pins.check
 producer=dict(schema='M_STATISTIC_EXECUTION_RETURN_V1',status='COMPLETE_DECLARED_TRANSPORT_PENDING_INDEPENDENT_SAVED_RESULT_AUDIT',job=spec,output=str(output),complete_sha256='c'*64,identity_sha256=output.parent.name,independent_cache_admission_sha256=refs['cache_admission']['sha256'],production_permit_sha256=refs['production_permit']['sha256'])
 refs['producer_execution']=tiny_ref(m,root/'proofs/producer_execution.json',producer);initial[refs['producer_execution']['path']]=refs['producer_execution']['sha256']
 request=dict(schema=m.audit.REQUEST_SCHEMA,root=str(root),output=str(output),complete_sha256='c'*64,job=spec,proofs=refs,consumer_dependencies_sha256=m.audit.dependency_pins())
 requestref=tiny_ref(m,root/'audit_request.json',request);initial[requestref['path']]=requestref['sha256'];initial.update(request['consumer_dependencies_sha256']);initial.update(ident['code_sha256'])
 history=tiny_ref(m,root/'source_proofs/original_independent_acceptance.json',dict(schema='literal historical independent proof',accepted=True));initial[history['path']]=history['sha256']
 # Upstream original weight pin is a historical accepted fact. Do not re-open it.
 initial[str(root/'original_source_best.pt')]='9'*64
 report.update(initial_fixed_sha256=initial,result_initial_fixed_sha256=dict(result_pins.expected),consumer_dependencies_sha256=request['consumer_dependencies_sha256'])
 reportref=tiny_ref(m,root/'audit_report.json',report)
 monkeypatch.setattr(auditor,'ReadOnlyArtifacts',lambda *a,**k:reader)
 return m,dict(job=spec,audit=reportref,audit_request=requestref,selected_prefix=prefix(spec)),report,reader


def test_accepted_report_rebinds_six_roles_without_original_weight_reopen(tmp_path,monkeypatch):
 m,item,report,reader=accepted_item_fixture(tmp_path,monkeypatch);pins=m.audit.Pins()
 actual,loaded,input_sha=m.load_accepted_item(item,pins)
 assert actual==report and loaded is reader and input_sha==reader.identity['input_admission_sha256']
 assert not any(p.endswith('.pt') for p in pins.expected)
 assert len([p for p in pins.expected if '/features/' in p and p.endswith('/complete.json')])==6

@pytest.mark.parametrize('mutation',['head_pin','result_pin','role_marker','producer','package','source_role_bytes','historical_proof','manifest_pin'])
def test_accepted_boundary_rejects_missing_or_changed_fixed_evidence(tmp_path,monkeypatch,mutation):
 m,item,report,reader=accepted_item_fixture(tmp_path,monkeypatch)
 if mutation=='head_pin':
  path=next(p for p in report['initial_fixed_sha256'] if p.endswith('head_weight.npy'));del report['initial_fixed_sha256'][path]
 elif mutation=='result_pin':report['result_initial_fixed_sha256']={str(Path(reader.output)/'fake'): '0'*64}
 elif mutation=='role_marker':reader.identity['cache_completion_sha256'].pop(next(iter(reader.identity['cache_completion_sha256'])))
 elif mutation=='producer':
  path=next(p for p in report['initial_fixed_sha256'] if p.endswith('producer_execution.json'));d=m.audit.json_read(path);d['job']=job('H',42);tiny_ref(m,Path(path),d)
 elif mutation=='package':reader.identity['code_sha256']={}
 elif mutation=='historical_proof':
  path=next(p for p in report['initial_fixed_sha256'] if p.endswith('original_independent_acceptance.json'));Path(path).write_text('{}')
 elif mutation=='manifest_pin':
  path=next(p for p in report['initial_fixed_sha256'] if '/features/' in p and p.endswith('manifest.parquet'));del report['initial_fixed_sha256'][path]
 else:
  path=next(p for p in report['initial_fixed_sha256'] if p.endswith('chunk_000000.npy'));Path(path).write_bytes(b'changed')
 item['audit']=tiny_ref(m,Path(item['audit']['path']),report)
 with pytest.raises(ValueError):m.load_accepted_item(item,m.audit.Pins())



def test_insufficient_fresh_budget_blocks_before_any_result_or_cache_read(tmp_path,monkeypatch):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch);monkeypatch.setattr(m,'_free_bytes',lambda root:0)
 monkeypatch.setattr(m,'load_accepted_item',lambda *a:pytest.fail('No accepted result/cache reads before fresh resource gate'))
 with pytest.raises(ValueError):m.execute_summary(ref['path'],ref['sha256'],0)
 assert not Path(q['output']).exists()

@pytest.mark.parametrize('mutation',['n_float','min_float','fraction_bool','AUC_range'])
def test_normal_metadata_and_finite_draw_auc_types_are_strict(mutation):
 raw,l2=normal(True),normal(False)
 if mutation=='n_float':l2['n_source_groups']=20.
 elif mutation=='min_float':l2['min_groups']=20.
 elif mutation=='fraction_bool':l2['largest_group_fraction']=True
 else:l2['draws'][1]['pooled_auc']=2.
 with pytest.raises(ValueError):api().normal_delta_interval(raw,l2,-.2)


def test_RQ2_percentile_interpolates_frozen_draws_and_never_RNG(monkeypatch):
 m=api();raw,l2=normal(True),normal(False)
 for i,d in enumerate(l2['draws']):d['pooled_auc']=.25+i/4000
 monkeypatch.setattr(np.random,'default_rng',lambda *a,**k:pytest.fail('No new resampling'))
 result=m.normal_delta_interval(raw,l2,-.2)
 values=[i/4000 for i in range(2000)]
 assert result['lower']==pytest.approx(49.975/4000) and result['upper']==pytest.approx(1949.025/4000)
 assert result['lower']==pytest.approx(np.quantile(values,.025,method='linear'))
 assert result['upper']==pytest.approx(np.quantile(values,.975,method='linear'))


def test_entire_H_numeric_NA_layer_keeps_all_five_seed_slots():
 m=api();rows=[]
 for seed in m.SEEDS:
  for make in (d_case,h_case):
   spec,report,reader=make(seed)
   if make is h_case:
    for c in report['numeric']['cohorts']:
     c['independent_point_estimates']={k:None for k in c['independent_point_estimates']}
   rows.extend(m.extract_core(spec,report,reader))
 result=m.summarize_rows(rows,'clip_vit_b16');unavailable=[r for r in result if r['core']=='RQ1']
 assert len(unavailable)==56 and all(r['registered_n']==5 and r['finite_n']==0 and r['mean'] is None and r['sample_sd'] is None for r in unavailable)


def test_same_row_count_but_changed_comparison_family_rejected():
 m=api();rows=[]
 for seed in m.SEEDS:
  for make in (d_case,h_case):
   spec,report,reader=make(seed);rows.extend(m.extract_core(spec,report,reader))
 for r in rows:
  if r['core']=='RQ2':r['threshold_kind']='DEV_selected_budget80'
 with pytest.raises(ValueError):m.summarize_rows(rows,'clip_vit_b16')


def test_mixed_source_input_capsules_cannot_publish(tmp_path,monkeypatch):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch);original=m.load_accepted_item
 def mixed(item,pins):
  r,reader,source=original(item,pins);return r,reader,source if item['job']['route']=='D' else 'f'*64
 monkeypatch.setattr(m,'load_accepted_item',mixed)
 with pytest.raises(ValueError):m.execute_summary(ref['path'],ref['sha256'],0)
 assert not Path(q['output']).exists()


# Independent-review fixes: no real M33 source bytes are ever modified.
def _fake_capacity_import(monkeypatch,module,phase=None):
 import builtins,types,importlib
 original_import=builtins.__import__;original_import_module=importlib.import_module
 state={'imports':0,'called':0,'drift':False};path=module.CODE/'reliability_v2/source_best_analysis.py'
 fake=types.SimpleNamespace(__file__=str(path))
 def remaining(root,plan):
  state['called']+=1
  if phase=='return':state['drift']=True
  return 123
 fake.remaining_required=remaining
 def selected():
  state['imports']+=1
  if phase=='import':state['drift']=True
  return fake
 def fake_import(name,globals=None,locals=None,fromlist=(),level=0):
  if name=='reliability_v2' and 'source_best_analysis' in (fromlist or ()):return types.SimpleNamespace(source_best_analysis=selected())
  return original_import(name,globals,locals,fromlist,level)
 def fake_module(name,package=None):
  if name=='reliability_v2.source_best_analysis':return selected()
  return original_import_module(name,package)
 monkeypatch.setattr(builtins,'__import__',fake_import);monkeypatch.setattr(importlib,'import_module',fake_module)
 return state


def test_resource_dependency_refresh_cannot_self_seal_or_enter_loader(monkeypatch):
 import hashlib
 original_read=Path.read_bytes;path=api().CODE/'reliability_v2/source_best_analysis.py';bad=b'private observed unreviewed source; never written'
 monkeypatch.setattr(Path,'read_bytes',lambda self:bad if self==path else original_read(self))
 spec=importlib.util.spec_from_file_location('m_summary_private_dependency_refresh',MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if len(m.RESOURCE_CODES)==2:
  original_sha=m.audit.sha;monkeypatch.setattr(m.audit,'sha',lambda p:hashlib.sha256(bad).hexdigest() if Path(p)==path else original_sha(p))
 state=_fake_capacity_import(monkeypatch,m)
 with pytest.raises(ValueError):m._remaining_required(Path('/tmp/private'),{})
 assert state['imports']==state['called']==0


def test_prepare_only_does_not_import_M33_source(monkeypatch):
 import builtins
 original=builtins.__import__;seen=[]
 def observe(name,globals=None,locals=None,fromlist=(),level=0):
  if name=='reliability_v2' or name.startswith('reliability_v2.'):seen.append((name,fromlist))
  return original(name,globals,locals,fromlist,level)
 monkeypatch.setattr(builtins,'__import__',observe)
 spec=importlib.util.spec_from_file_location('m_summary_private_prepare_import',MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.main([])['status']=='PREPARE_ONLY' and seen==[]


def test_resource_expected_M33_is_fixed_qualification_inventory():
 m=api();d=__import__('json').loads((m.ROOT/'engineering/source_best_M/M_statistic_canary_engineering_acceptance_20261026T0430.json').read_text())
 assert m.RESOURCE_CODES==d['code_sha256'] and len(m.RESOURCE_CODES)==33
 assert m.RESOURCE_CODES[str(m.CODE/'reliability_v2/source_best_analysis.py')]=='0ec974eec85a0a5204b8804707d627fa908a80eb1a4e76cfa81729ce0a875a32'
 assert m.RESOURCE_CODES[str(m.CODE/'reliability_v2/source_best_runtime.py')]=='69b3d590d95fd3004c7c64617fbad642526b02862a97f61546d3cd375fc18742'

@pytest.mark.parametrize('phase',['before','import','return'])
def test_transitive_cache_drift_rejected_before_import_after_import_and_return(monkeypatch,phase):
 m=api();state=_fake_capacity_import(monkeypatch,m,phase);state['drift']=phase=='before';old_sha=m._sha
 target=m.CODE/'reliability_v2/source_best_cache.py'
 monkeypatch.setattr(m,'_sha',lambda path:'0'*64 if Path(path)==target and state['drift'] else old_sha(path))
 with pytest.raises(ValueError):m._remaining_required(Path('/tmp/private'),{})
 assert state['imports']==(0 if phase=='before' else 1)
 assert state['called']==(1 if phase=='return' else 0)


def test_fixed_qualification_metadata_drift_rejected_before_loader(monkeypatch):
 m=api();state=_fake_capacity_import(monkeypatch,m);original=m._sha
 monkeypatch.setattr(m,'_sha',lambda p:'f'*64 if Path(p).name=='M_statistic_canary_engineering_acceptance_20261026T0430.json' else original(p))
 with pytest.raises(ValueError):m._remaining_required(Path('/tmp/private'),{})
 assert state['imports']==state['called']==0


def test_five_seed_sign_counts_are_descriptive_tolerance_and_NA_preserving():
 m=api();values=dict(zip(m.SEEDS,(2e-10,-2e-10,1e-10,-1e-10,None)));r=m.five_seed_statistics(values)
 assert (r['positive_count'],r['negative_count'],r['zero_count'],r['na_count'])==(1,1,2,1)
 assert r['sign_valid_count']==4 and r['sign_unknown_count']==1 and r['sign_tolerance']==1e-10
 assert r['mean'] is None and r['sample_sd'] is None


def test_sign_counts_keep_strict_near_tolerance_cases_and_all_NA():
 m=api();r=m.five_seed_statistics(dict(zip(m.SEEDS,(np.nextafter(1e-10,np.inf),np.nextafter(-1e-10,-np.inf),np.nextafter(1e-10,0.),np.nextafter(-1e-10,0.),0.))))
 assert (r['positive_count'],r['negative_count'],r['zero_count'],r['na_count'])==(1,1,3,0)
 r=m.five_seed_statistics(dict.fromkeys(m.SEEDS,None));assert r['na_count']==r['sign_unknown_count']==5 and r['sign_valid_count']==0


def test_RQ3_reversal_counts_use_two_deltas_not_interaction_sign():
 m=api();pairs=dict(zip(m.SEEDS,((.1,.2),(-.1,.1),(.1,-.1),(0.,-.2),(None,.2))))
 # interaction is positive for first two,negative for third/fourth; only2/3 reverse.
 r=m.five_seed_reversal_counts(pairs)
 assert r==dict(reversal_count=2,non_reversal_count=2,reversal_valid_count=4,reversal_unknown_count=1,reversal_tolerance=1e-10)
 assert sum((r['reversal_count'],r['non_reversal_count'],r['reversal_unknown_count']))==5


def test_RQ3_reversal_counts_zero_tolerance_and_entire_unknown():
 m=api();pairs=dict(zip(m.SEEDS,((1e-10,-.2),(-1e-10,.2),(1.01e-10,-1.01e-10),(None,None),(.1,None))))
 r=m.five_seed_reversal_counts(pairs);assert r['reversal_count']==1 and r['non_reversal_count']==2 and r['reversal_valid_count']==3 and r['reversal_unknown_count']==2
 r=m.five_seed_reversal_counts(dict.fromkeys(m.SEEDS,(None,None)));assert r['reversal_unknown_count']==5 and r['reversal_count']==r['non_reversal_count']==0


def test_summary_and_saved_CSV_reversal_counts_have_unchanged_endpoint_inventory():
 m=api();rows=[]
 pairs=dict(zip(m.SEEDS,((.1,.2),(-.1,.1),(.1,-.1),(0.,-.2),(None,.2))))
 for seed in m.SEEDS:
  for make in (d_case,h_case):
   spec,report,reader=make(seed);new=m.extract_core(spec,report,reader)
   for row in new:
    if row['core']=='RQ3':
     a,b=pairs[seed];row['delta_correct']=a;row['delta_collision']=b;row['estimate']=None if a is None or b is None else b-a
     row['reversal_valid']=a is not None and b is not None;row['reversal']=bool(row['reversal_valid'] and a*b<0)
   rows.extend(new)
 summary=m.summarize_rows(rows,'clip_vit_b16');rq3=[r for r in summary if r['core']=='RQ3']
 assert len(rows)==355 and len(summary)==71 and len(rq3)==2
 assert all((r['reversal_count'],r['non_reversal_count'],r['reversal_unknown_count'])==(2,2,1) for r in rq3)
 assert all(r['mean'] is None and r['sample_sd'] is None for r in rq3)
 m._independent_csv_check(m._csv_bytes(rows),m._csv_bytes(summary),rows,summary,'clip_vit_b16')
 wrong=copy.deepcopy(summary);next(r for r in wrong if r['core']=='RQ3')['reversal_count']=4
 with pytest.raises(ValueError):m._independent_csv_check(m._csv_bytes(rows),m._csv_bytes(wrong),rows,wrong,'clip_vit_b16')



def _keep_size_allocate_private_csv(path,bytes_count):
 import subprocess
 # Only this test's tmp namespace is eligible; never a source or real result.
 assert '/tmp/pytest-' in str(path) and path.name in ('per_seed_core.csv','five_seed_core.csv')
 subprocess.run(['/usr/bin/fallocate','--keep-size','--length',str(bytes_count),str(path)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def test_existing_reuse_rejects_allocated_overcap_with_all_SHA_bytes_unchanged(tmp_path,monkeypatch):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch);result=m.execute_summary(ref['path'],ref['sha256'],0)
 output=Path(q['output']);before={p.name:p.read_bytes() for p in output.iterdir()};cap=m.audit.json_read(q['budget']['path'])['max_output_bytes']
 initial=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in (output,*output.iterdir()));assert initial<cap
 _keep_size_allocate_private_csv(output/'per_seed_core.csv',cap)
 after=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in (output,*output.iterdir()));assert after>cap
 assert {p.name:p.read_bytes() for p in output.iterdir()}==before
 with pytest.raises(ValueError,match='allocation'):m.execute_summary(ref['path'],ref['sha256'],1)
 assert {p.name:p.read_bytes() for p in output.iterdir()}==before and (output/'complete.json').exists()
 assert result['complete_sha256']==__import__('hashlib').sha256(before['complete.json']).hexdigest()


def test_allocation_measure_includes_directories_blocks_and_logical_roundup(tmp_path):
 m=api();out=tmp_path/'allocation';(out/'sub').mkdir(parents=True);(out/'small').write_bytes(b'x')
 with (out/'sub/sparse').open('wb') as stream:stream.truncate(4097)
 fs=__import__('os').statvfs(out);unit=fs.f_frsize or fs.f_bsize
 members=(out,*out.rglob('*'));expected=sum(max(((p.stat().st_size+unit-1)//unit)*unit,p.stat().st_blocks*512) for p in members)
 assert m._allocation_bytes(out)==expected
 assert expected>=sum(p.stat().st_blocks*512 for p in (out,*out.rglob('*')))


def test_fresh_publication_late_allocated_overcap_refuses_completion_and_preserves_files(tmp_path,monkeypatch):
 m,root,q,ref,events=execution_fixture(tmp_path,monkeypatch);original=m._write_new;observed={};cap=m.audit.json_read(q['budget']['path'])['max_output_bytes']
 def allocate(path,data):
  original(path,data)
  if path.name=='five_seed_core.csv':
   observed['bytes']=path.read_bytes();_keep_size_allocate_private_csv(path,cap);assert path.read_bytes()==observed['bytes']
 monkeypatch.setattr(m,'_write_new',allocate)
 with pytest.raises(ValueError,match='allocation'):m.execute_summary(ref['path'],ref['sha256'],0)
 output=Path(q['output']);assert not (output/'complete.json').exists() and (output/'five_seed_core.csv').read_bytes()==observed['bytes']
