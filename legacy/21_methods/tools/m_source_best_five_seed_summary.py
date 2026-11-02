"""Frozen four-core source-best descriptive five-seed summary. Default PREPARE_ONLY.

No models,source revalidation,random draws,training,p-values or permit issuer.
An actual execution consumes ten independently accepted immutable job audits.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,io,json,math,numbers,os,resource,shutil,stat,statistics,sys
from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import pandas as pd

CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
AUDIT=Path(__file__).parent/'m_source_best_result_audit.py'
AUDIT_SHA='7836d63269669aea0c7ef73e3c8bc88309dd4b17645d06d40b30b8f11c0e9e98'
PLAN=Path('/home/z/obsidian/paper/deepfake-detector/新实验设计/20261023/新实验计划_本机适配版.md')
PLAN_SHA='b018b01915a62c645fc3718134970cd49dc0651ecfd02792220f3a0086c6594d'
def require(ok,why):
 if not ok:raise ValueError(why)
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
ENTRY_SHA=_sha(__file__)
QUALIFICATION=ROOT/'engineering/source_best_M/M_statistic_canary_engineering_acceptance_20261026T0430.json'
QUALIFICATION_SHA='9bec22803f8e7c15918fdb1b4938e028e09a3115a4d23aea89dff1cb89d25571'
M33_NAMES=('__init__.py','analysis_d.py','analysis_h.py','bootstrap_fast.py','cache_d.py','cache_h.py','coverage_h.py','data_d.py','data_h.py','geometry_h.py','models.py','multiplicity_bundle.py','multiplicity_codec.py','qualification_h.py','qualify.py','queue_d.py','queue_h.py','scoring_d.py','sensitivity_io.py','source_best_analysis.py','source_best_cache.py','source_best_files.py','source_best_gpu.py','source_best_lease.py','source_best_policy.py','source_best_runtime.py','source_best_sink.py','source_best_source_qualification.py','source_best_stats.py','stats_h.py','summary_d.py','trainer.py','worker_h.py')

def _canonical_file(path):
 p=Path(path).absolute();require(p.resolve()==p and not any(x.is_symlink() for x in (p,*p.parents)),'canonical nonsymlink fixed source required')
 st=p.stat();require(stat.S_ISREG(st.st_mode) and st.st_nlink==1,'private regular fixed source required');return p

def _accepted_resource_codes():
 # Only immutable reviewed metadata supplies expected values; never live code.
 p=_canonical_file(QUALIFICATION);require(p.stat().st_size<=2*1024**2 and _sha(p)==QUALIFICATION_SHA,'fixed reviewed M33 metadata drift')
 doc=json.loads(p.read_bytes());codes=doc['code_sha256'];expected={str(CODE/'reliability_v2'/name) for name in M33_NAMES}
 require(type(codes) is dict and len(codes)==33 and set(codes)==expected,'exact33 canonical reviewed M33 source paths required')
 require(all(type(v) is str and len(v)==64 and set(v)<=set('0123456789abcdef') for v in codes.values()),'fixed M33 SHA256 values required')
 for path in codes:_canonical_file(path)
 require(codes[str(CODE/'reliability_v2/source_best_analysis.py')]=='0ec974eec85a0a5204b8804707d627fa908a80eb1a4e76cfa81729ce0a875a32' and codes[str(CODE/'reliability_v2/source_best_runtime.py')]=='69b3d590d95fd3004c7c64617fbad642526b02862a97f61546d3cd375fc18742','accepted analysis/runtime identities required')
 require(_sha(p)==QUALIFICATION_SHA,'fixed reviewed metadata changed while reading');return dict(codes)

RESOURCE_CODES=_accepted_resource_codes()

def _check_resource_codes():
 require(_sha(_canonical_file(QUALIFICATION))==QUALIFICATION_SHA,'fixed reviewed M33 metadata drift')
 actual={str(p.absolute()) for p in (CODE/'reliability_v2').glob('*.py')}
 require(actual==set(RESOURCE_CODES),'exact reviewed M33 source inventory drift')
 for path,digest in RESOURCE_CODES.items():require(_sha(_canonical_file(path))==digest,'fixed reviewed transitive M33 source drift '+path)

_AUDIT_MODULE=None

def _load_audit():
 global _AUDIT_MODULE
 if _AUDIT_MODULE is None:
  _check_resource_codes();require(_sha(_canonical_file(AUDIT))==AUDIT_SHA,'accepted single-job consumer source changed')
  try:
   spec=importlib.util.spec_from_file_location('_selected_M_summary_audit',AUDIT);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  finally:
   require(_sha(_canonical_file(AUDIT))==AUDIT_SHA,'consumer changed while importing');_check_resource_codes()
  _AUDIT_MODULE=module
 return _AUDIT_MODULE

class _AuditProxy:
 # Default PREPARE_ONLY reads fixed metadata,never executes M33 source imports.
 def __getattr__(self,name):return getattr(_load_audit(),name)

audit=_AuditProxy()
SEEDS=(7,42,123,2026,3407)
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
AGGS=('mean_logit','mean_probability')
METRICS=('pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba')
SCHEMA='M_SOURCE_BEST_FIVE_SEED_SUMMARY_V1'
REQUEST_SCHEMA='M_SOURCE_BEST_FIVE_SEED_SUMMARY_REQUEST_V1'
BUDGET_SCHEMA='M_SOURCE_BEST_FIVE_SEED_SUMMARY_RESOURCE_BUDGET_V1'
KEYS=('core','domain','cohort','aggregation','representation','threshold_kind','metric','comparison','grouping')
DERIVATION='downstream arithmetic from aligned frozen normal draws'

def dependencies():
 base={str(CODE/'reliability_v2'/name):RESOURCE_CODES[str(CODE/'reliability_v2'/name)] for name in ('multiplicity_bundle.py','multiplicity_codec.py','sensitivity_io.py')}
 return base|RESOURCE_CODES|{str(AUDIT):AUDIT_SHA,str(PLAN):PLAN_SHA,str(Path(__file__).absolute()):ENTRY_SHA,str(QUALIFICATION):QUALIFICATION_SHA,str(ROOT/'engineering/source_best_M/audits/h_pair_result_audit.py'):'6cd3bff403cc2f7c27a870d092023e4c8677f5d9386668abbc439829eadaa4e5',str(CODE/'tools/m_source_best_d_numeric.py'):'a7b079965fd34acbf2e974480aa197ded476cbc21f1447aeac523e1c4aab1dd0'}

def number(value):
 if value is None:return None
 require(not isinstance(value,(bool,)) and isinstance(value,numbers.Real),'finite numeric effect or explicit NA required')
 value=float(value)
 if math.isnan(value):return None
 require(math.isfinite(value),'infinite effect is not NA');return value

def close(a,b):
 x,y=number(a),number(b)
 require((x is None and y is None) or (x is not None and y is not None and abs(x-y)<=1e-10),'saved independently accepted effect inconsistent')

def five_seed_statistics(values):
 require(type(values) is dict and set(values)==set(SEEDS) and all(type(k) is int for k in values),'exact five registered training seeds required')
 q={s:number(values[s]) for s in SEEDS};bad=[s for s in SEEDS if q[s] is None]
 out=dict(registered_n=5,finite_n=5-len(bad),unavailable_seeds=bad,mean=None,sample_sd=None,ddof=1,positive_count=sum(x is not None and x>1e-10 for x in q.values()),negative_count=sum(x is not None and x < -1e-10 for x in q.values()),zero_count=sum(x is not None and abs(x)<=1e-10 for x in q.values()),na_count=len(bad),sign_valid_count=5-len(bad),sign_unknown_count=len(bad),sign_tolerance=1e-10,status='UNAVAILABLE_NA_REGISTERED_SEED' if bad else 'AVAILABLE')
 if bad:return out
 vals=[q[s] for s in SEEDS];f=[Fraction.from_float(x) for x in vals];mean=sum(f,Fraction())/5;var=sum((x-mean)**2 for x in f)/4
 m,sd=float(mean),math.sqrt(float(var))
 require(math.isclose(m,statistics.mean(vals),rel_tol=1e-12,abs_tol=1e-12) and math.isclose(sd,statistics.stdev(vals),rel_tol=1e-12,abs_tol=1e-12),'independent five-seed arithmetic disagreement')
 out.update(mean=m,sample_sd=sd);return out


def five_seed_reversal_counts(pairs):
 require(type(pairs) is dict and set(pairs)==set(SEEDS) and all(type(s) is int for s in pairs),'exact five registered reversal seed pairs required')
 valid=reversal=0
 for seed in SEEDS:
  pair=pairs[seed];require(type(pair) in (tuple,list) and len(pair)==2,'both correct/collision deltas required')
  a,b=(number(v) for v in pair)
  if a is None or b is None:continue
  valid+=1
  reversal+=int((a>1e-10 and b < -1e-10) or (a < -1e-10 and b>1e-10))
 return dict(reversal_count=reversal,non_reversal_count=valid-reversal,reversal_valid_count=valid,reversal_unknown_count=5-valid,reversal_tolerance=1e-10)

def _quantile(values,p):
 q=sorted(Fraction.from_float(x) for x in values);at=(len(q)-1)*p;i=at.numerator//at.denominator;part=at-i
 return float(q[i] if i==len(q)-1 else q[i]*(1-part)+q[i+1]*part)

def normal_delta_interval(raw,l2,point):
 base=dict(status='UNAVAILABLE_UNALIGNED_FROZEN_NORMAL_DRAWS',lower=None,upper=None,valid_repeats=0,invalid_repeats=2000,requested_repeats=2000,interval_type='pointwise_conditional_percentile',derivation=DERIVATION,reason=None)
 for obj in (raw,l2):
  require(type(obj) is dict and type(obj.get('repeats')) is int and obj['repeats']==2000 and type(obj.get('seed')) is int and obj['seed']==20261023,'frozen normal TEST2000/seed required')
  groups=obj.get('source_groups');require(type(groups) is list and groups and all(type(g) is str and g for g in groups) and len(groups)==len(set(groups)),'registered ordered normal groups required')
  require(type(obj.get('n_source_groups')) is int and obj['n_source_groups']==len(groups) and type(obj.get('min_groups')) is int and obj['min_groups']==20 and type(obj.get('max_group_fraction')) in (int,float) and obj['max_group_fraction']==.5,'original normal group-gate protocol required')
  require(type(obj.get('largest_group_fraction')) in (int,float) and math.isfinite(obj['largest_group_fraction']) and 0<obj['largest_group_fraction']<=1,'finite group concentration required')
 if raw['source_groups']!=l2['source_groups'] or raw['largest_group_fraction']!=l2['largest_group_fraction']:return dict(base,reason='group identities/order/concentration differ')
 gate=len(raw['source_groups'])<20 or raw['largest_group_fraction']>.5
 require(all((x.get('status')=='insufficient_source_groups')==gate for x in (raw,l2)),'frozen normal group status contradicts gate')
 if gate:
  require(all(x.get('status')=='insufficient_source_groups' and x.get('draws')==[] for x in (raw,l2)),'contradictory frozen group gate')
  return dict(base,status='UNAVAILABLE_NORMAL_GROUP_GATE',reason='insufficient_source_groups')
 require(all(x.get('status') in ('ok','no_valid_draws') and type(x.get('draws')) is list and len(x['draws'])==2000 for x in (raw,l2)),'complete original normal draws required')
 groups=raw['source_groups'];values=[]
 for i,(a,b) in enumerate(zip(raw['draws'],l2['draws'])):
  for row in (a,b):
   require(type(row.get('repeat')) is int,'typed repeat required');counts=row.get('group_draw_counts')
   require(type(counts) is dict and counts and set(counts)<=set(groups) and all(type(v) is int and v>0 for v in counts.values()) and sum(counts.values())==len(groups),'exact integer group multiplicities required')
   require(type(row.get('pooled_valid')) is bool,'typed pooled validity required')
  if a['repeat']!=i or b['repeat']!=i or list(a['group_draw_counts'].items())!=list(b['group_draw_counts'].items()):return dict(base,reason='repeat or exact ordered multiplicities differ')
  x,y=number(a.get('pooled_auc')),number(b.get('pooled_auc'))
  require(all(v is None or 0<=v<=1 for v in (x,y)),'frozen normal AUC outside native range')
  if a['pooled_valid'] and b['pooled_valid'] and x is not None and y is not None:values.append(y-x)
 base.update(valid_repeats=len(values),invalid_repeats=2000-len(values))
 if not values or number(point) is None:return dict(base,status='UNAVAILABLE_NUMERIC_NA',reason='no finite paired draws or point estimate NA')
 lower,upper=_quantile(values,Fraction(1,40)),_quantile(values,Fraction(39,40))
 if len(values)>1:
  check=statistics.quantiles(values,n=40,method='inclusive');require(math.isclose(lower,check[0],abs_tol=1e-12,rel_tol=1e-12) and math.isclose(upper,check[38],abs_tol=1e-12,rel_tol=1e-12),'independent frozen-draw quantile disagreement')
 return dict(base,status='AVAILABLE_DERIVED_FROZEN_NORMAL_DRAWS',reason=None,lower=lower,upper=upper)

def validate_audit_report(report,job,prefix):
 require(job==audit.job_spec(job['route'],job['backbone'],job['seed']),'exact registered source-best job required')
 require(report.get('schema')==audit.SCHEMA and report.get('status')=='SELECTED_CORE_METADATA_AND_NUMERIC_ACCEPTED' and report.get('accepted_for_selected_core') is True and report.get('scientific_final_acceptance') is False and report.get('audited_scope')=='SELECTED_FOUR_CORE_ONLY','accepted independent four-core audit required')
 require(report.get('job')==job and report.get('core_endpoints')==audit.CORE_ENDPOINTS and report.get('layers')==audit.LAYERS,'audit job/core/layers differ')
 require(report.get('artifact_count')==(60 if job['route']=='D' else 37) and report.get('all_declared_artifacts_SHA_and_codec_verified') is True and report.get('metadata_status')=='ACTUAL_BOUND_SIX_ROLE_PRODUCER_EXIT0_AND_RESOURCE_EVIDENCE_RECHECKED' and report.get('fixed_SHA_rechecked_before_return') is True,'real complete/producer/six-role accepted evidence required')
 require(report.get('model_or_GPU_inference_started') is False and report.get('all_five_seed_conclusions_available') is False,'single-job source-best trust scope required')
 require(report.get('selected_prefix')==prefix and set(prefix)==set(job['runs']),'exact selected source prefixes required')
 for q in prefix.values():
  require(set(q)=={'seed','epoch','attempts','successful_updates','AMP_skips'} and q['seed']==job['seed'],'source prefix fields/seed required')
  require(all(type(q[k]) is int and q[k]>=0 for k in q) and q['epoch']>=1 and q['successful_updates']+q['AMP_skips']==q['attempts'],'source prefix ledger invalid')
 n=report.get('numeric',{});require(n.get('accepted_for_selected_core') is True and n.get('blockers_for_selected_core',[])==[],'numeric selected core not accepted')
 require(n.get('status')==('PASS_SELECTED_D_CORE' if job['route']=='D' else 'NUMERIC_SELECTED_NATIVE_B_CORE_RECOMPUTED'),'route numeric oracle status')
 if job['route']=='D':require(n.get('schema')=='M_SOURCE_BEST_D_INDEPENDENT_NUMERIC_CORE_V1' and n.get('scientific_final_acceptance') is False,'D core-only trust scope')
 return report

def _reference(reader,name,selector,layer):
 require(name in reader.records,'accepted artifact reference missing')
 return dict(output=str(reader.output),complete_sha256=reader.complete_sha256,artifact=name,files=dict(reader.records[name]['files']),selector=selector,layer=layer)

def _row(job,report,core,domain,aggregation,representation,threshold,metric,comparison,grouping,value):
 value=number(value);require(value is None or abs(value)<=(2 if core=='RQ3' else 1)+1e-10,'native effect range differs')
 return dict(core=core,domain=domain,cohort='all_B' if core=='RQ1' else 'fixed_CDF',aggregation=aggregation,representation=representation,threshold_kind=threshold,metric=metric,comparison=comparison,grouping=grouping,backbone=job['backbone'],seed=job['seed'],route=job['route'],job_id=job['job_id'],estimate=value,layer='fixed_training_seed_point',units='native_proportion_difference',selected_prefix=report['selected_prefix'],update_matched_primary_claim=False,TEST_ci=None,TEST_reference=None,CAL_reference=None)

def extract_core(job,report,artifacts):
 validate_audit_report(report,job,report['selected_prefix']);rows=[]
 if job['route']=='H':
  cohorts=report['numeric'].get('cohorts');require(type(cohorts) is list and len(cohorts)==4 and {(c['domain'],c['aggregation'],c['cohort']) for c in cohorts}=={(d,a,'all_B') for d in ('SOURCE_TEST','TARGET_TEST') for a in AGGS},'exact four native H cohorts required')
  for c in cohorts:
   name=f'{c["aggregation"]}/{c["domain"]}/all_B/conditional_bootstrap.json'
   for kind in ('default','source_real'):
    for metric in METRICS:
     key=f'{kind}:comparison:H10-H00:{metric}';require(key in c['independent_point_estimates'],'independent H point missing')
     row=_row(job,report,'RQ1','DFD_TARGET_TEST' if c['domain']=='TARGET_TEST' else 'FF_SOURCE_TEST',c['aggregation'],'raw',kind,metric,'H10-H00','correct',c['independent_point_estimates'][key])
     row['TEST_reference']=_reference(artifacts,name,dict(threshold_kind=kind,target_kind='comparison',target='H10-H00',metric=metric),'conditional_TEST_2000')
     row['TEST_ci']=dict(status=c['status'] if row['estimate'] is not None else 'NUMERIC_NA',source='original_saved_conditional_TEST_interval',requested_repeats=2000)
     rows.append(row)
  return rows
 metrics=artifacts['target_full/metrics.csv'];effects=artifacts['target_full/effects.csv']
 require(isinstance(metrics,pd.DataFrame) and len(metrics)==8 and not metrics.duplicated(['grouping','representation','aggregation']).any(),'exact D8 accepted metric cells')
 index={(r['grouping'],r['representation'],r['aggregation']):number(r['pooled_auc']) for r in metrics.to_dict('records')}
 require(set(index)=={(g,r,a) for g in ('correct','collision') for r in ('raw','l2') for a in AGGS},'exact D metric inventory')
 require(isinstance(effects,pd.DataFrame) and len(effects)==2 and set(effects.aggregation)==set(AGGS),'exact saved effect rows')
 sign=lambda x:None if x is None else 1 if x>1e-10 else -1 if x < -1e-10 else 0
 for agg in AGGS:
  delta={g:None if index[(g,'raw',agg)] is None or index[(g,'l2',agg)] is None else index[(g,'l2',agg)]-index[(g,'raw',agg)] for g in ('correct','collision')}
  interaction=None if None in delta.values() else delta['collision']-delta['correct'];a,b=sign(delta['correct']),sign(delta['collision']);valid=a is not None and b is not None;reversal=bool(valid and a*b==-1)
  saved=effects.loc[effects.aggregation.eq(agg)].iloc[0]
  for k,v in [('delta_correct',delta['correct']),('delta_collision',delta['collision']),('interaction',interaction)]:close(saved[k],v)
  require(bool(saved['valid'])==valid and bool(saved['reversal'])==reversal and float(saved['tolerance'])==1e-10,'fixed reversal/validity rule differs')
  for k,v in [('sign_correct',a),('sign_collision',b)]:close(saved[k],v)
  if agg=='mean_logit':
   row=_row(job,report,'RQ2','CDF_TARGET_FULL',agg,'l2-raw','none','pooled_auc','L2-Raw','correct',delta['correct'])
   raw_name='target_full/normal_bootstrap_raw_mean_logit.json';l2_name='target_full/normal_bootstrap_l2_mean_logit.json';raw,l2=artifacts[raw_name],artifacts[l2_name]
   close(raw['point_estimate']['pooled_auc'],index[('correct','raw',agg)]);close(l2['point_estimate']['pooled_auc'],index[('correct','l2',agg)])
   row['TEST_ci']=normal_delta_interval(raw,l2,delta['correct']);row['TEST_reference']=dict(derivation=DERIVATION,layer='conditional_TEST_2000',operation='L2-Raw',source_artifacts=[_reference(artifacts,n,dict(field='draws[].pooled_auc'),'conditional_TEST_2000') for n in (raw_name,l2_name)],stored_original_delta_CI=False)
   rows.append(row)
  row=_row(job,report,'RQ3','CDF_TARGET_FULL',agg,'l2-raw','none','pooled_auc','delta_collision-delta_correct','collision-correct',interaction)
  name='target_full/closure_bootstrap.json';states=[q for q in report['numeric'].get('bootstrap_states',[]) if q['artifact']==name];require(len(states)==1,'audited closure status missing')
  row.update(delta_correct=delta['correct'],delta_collision=delta['collision'],sign_correct=a,sign_collision=b,reversal=reversal,reversal_valid=valid)
  row['TEST_ci']=dict(status=states[0]['status'] if interaction is not None else 'NUMERIC_NA',source='original_common_source_and_collision_closure',requested_repeats=2000)
  row['TEST_reference']=_reference(artifacts,name,dict(comparison=agg+':interaction'),'conditional_TEST_2000');rows.append(row)
 for rep in ('raw','l2'):
  for agg in AGGS:
   folder=f'rq4/{rep}_{agg}';boot_name=folder+'/primary_n20_vs_source_bootstrap.json';cal_name=folder+'/calibration.csv';obj=artifacts[boot_name];cal=artifacts[cal_name]
   require(isinstance(cal,pd.DataFrame),'saved CAL table required');n20=cal[cal.calibration_kind.eq('target_real')&cal.budget.eq(20)]
   require(len(n20)==200 and sorted(n20.repeat.tolist())==list(range(200)),'all200 n20 CAL subset records required')
   unavailable=obj.get('status')=='UNAVAILABLE_CALIBRATION_BUDGET'
   require((n20.status.eq('unavailable').all() if unavailable else n20.status.eq('ok').all()),'n20 availability differs')
   if not unavailable:require(obj.get('calibration_budget')==20 and obj.get('calibration_repeats')==200,'one model/fixed200 CAL sets required')
   for metric in ('fpr','tpr','pooled_ba'):
    row=_row(job,report,'RQ4','CDF_TARGET_TEST',agg,rep,'n20_vs_source_real',metric,'target_real-source_real','correct',None if unavailable else obj['point_estimates']['delta_'+metric])
    row['TEST_ci']=dict(status=obj['status'] if row['estimate'] is not None else ('UNAVAILABLE_CALIBRATION_BUDGET' if unavailable else 'NUMERIC_NA'),source='original_n20_operating_bootstrap',requested_repeats=2000)
    row['TEST_reference']=_reference(artifacts,boot_name,dict(comparison='delta_'+metric),'conditional_TEST_2000')
    row['CAL_reference']=dict(_reference(artifacts,cal_name,dict(calibration_kind='target_real',budget=20),'CAL_subset_spread_on_fixed_TEST'),requested_subsets=200,available_subsets=0 if unavailable else 200,not_training_models=True)
    rows.append(row)
 return rows

def summarize_rows(rows,backbone):
 require(backbone in BACKBONES and type(rows) is list and len(rows)==355,'one backbone/all71 endpoints x5 seeds required')
 grouped=defaultdict(dict)
 for row in rows:
  require(row['backbone']==backbone and row['layer']=='fixed_training_seed_point' and row['seed'] in SEEDS and type(row['seed']) is int,'training seed/backbone/layer differs')
  key=tuple(row[k] for k in KEYS);require(row['seed'] not in grouped[key],'duplicate endpoint seed');grouped[key][row['seed']]=row
 require(set(grouped)==expected_endpoint_keys(),'complete frozen endpoint set required');out=[]
 for key,records in sorted(grouped.items()):
  stats=five_seed_statistics({s:r['estimate'] for s,r in records.items()});row=dict(zip(KEYS,key));row.update(backbone=backbone,layer='training_seed_descriptive',**stats)
  if row['core']=='RQ3':row.update(five_seed_reversal_counts({s:(r['delta_correct'],r['delta_collision']) for s,r in records.items()}))
  out.append(row)
 return out

def expected_endpoint_keys():
 keys=set()
 for domain in ('DFD_TARGET_TEST','FF_SOURCE_TEST'):
  for agg in AGGS:
   for kind in ('default','source_real'):
    for metric in METRICS:keys.add(('RQ1',domain,'all_B',agg,'raw',kind,metric,'H10-H00','correct'))
 keys.add(('RQ2','CDF_TARGET_FULL','fixed_CDF','mean_logit','l2-raw','none','pooled_auc','L2-Raw','correct'))
 for agg in AGGS:
  keys.add(('RQ3','CDF_TARGET_FULL','fixed_CDF',agg,'l2-raw','none','pooled_auc','delta_collision-delta_correct','collision-correct'))
  for rep in ('raw','l2'):
   for metric in ('fpr','tpr','pooled_ba'):keys.add(('RQ4','CDF_TARGET_TEST','fixed_CDF',agg,rep,'n20_vs_source_real',metric,'target_real-source_real','correct'))
 return keys


def _reference_path(ref,pins):
 require(type(ref) is dict and set(ref)=={'path','sha256'},'exact external path/SHA reference required')
 return pins.add(ref['path'],ref['sha256'])


def inputs_identity(request):
 """No cycle: inputs exclude budget; output then binds inputs AND budget SHA."""
 values={k:request[k] for k in ('root','backbone','audits','dependencies_sha256')}
 return hashlib.sha256(audit.typed(values)).hexdigest()


def canonical_output(request):
 identity=dict(schema=SCHEMA,inputs_sha256=inputs_identity(request),budget=request['budget'])
 return ROOT/'reports/M_source_best_five_seed_summary'/request['backbone']/hashlib.sha256(audit.typed(identity)).hexdigest()


def load_summary_request(path,digest):
 pins=audit.Pins();pins.add(path,digest);q=audit.json_read(path,4*1024**2)
 require(set(q)=={'schema','root','backbone','audits','dependencies_sha256','budget','output'} and q['schema']==REQUEST_SCHEMA and q['root']==str(ROOT),'exact external summary request/root required')
 require(q['backbone'] in BACKBONES and q['dependencies_sha256']==dependencies(),'fixed summary/consumer/D/H/plan/resource sources required')
 for p,h in q['dependencies_sha256'].items():pins.add(p,h)
 require(type(q['audits']) is list and len(q['audits'])==10,'all ten D/H x five-seed accepted audit references required')
 seen=set()
 for item in q['audits']:
  require(type(item) is dict and set(item)=={'job','audit','audit_request','selected_prefix'},'exact independently accepted job reference')
  job=item['job'];require(job==audit.job_spec(job['route'],q['backbone'],job['seed']),'registered source-best job/run/seed/backbone required')
  require(job['job_id'] not in seen,'duplicate registered audit job');seen.add(job['job_id'])
  for ref in ('audit','audit_request'):
   value=item[ref];require(type(value) is dict and set(value)=={'path','sha256'},'exact bound accepted audit/original request reference');audit.hash_value(value['sha256']);audit.safe(value['path'],root=ROOT)
  prefix=item['selected_prefix'];require(type(prefix) is dict and set(prefix)==set(job['runs']),'declared exact selected prefixes')
  for v in prefix.values():
   require(set(v)=={'seed','epoch','attempts','successful_updates','AMP_skips'} and all(type(x) is int and x>=0 for x in v.values()) and v['seed']==job['seed'] and v['epoch']>=1 and v['attempts']==v['successful_updates']+v['AMP_skips'],'declared selected prefix ledger')
 require(seen=={audit.job_spec(r,q['backbone'],s)['job_id'] for r in ('D','H') for s in SEEDS},'no missing registered route/seed')
 budget=audit.json_read(_reference_path(q['budget'],pins))
 require(set(budget)=={'schema','root','inputs_sha256','capacity_plan','write_accounting','max_output_bytes','temporary_peak_bytes','max_peak_RSS_bytes','address_space_limit_bytes','runtime_floor_bytes'} and budget['schema']==BUDGET_SCHEMA and budget['root']==str(ROOT) and budget['inputs_sha256']==inputs_identity(q),'external resource budget must bind these exact ten accepted audits')
 for k in ('max_output_bytes','temporary_peak_bytes','max_peak_RSS_bytes','address_space_limit_bytes','runtime_floor_bytes'):require(type(budget[k]) is int and budget[k]>0,'strict positive resource bounds')
 require(budget['max_output_bytes']<=16*1024**2 and budget['temporary_peak_bytes']<=16*1024**2 and budget['max_peak_RSS_bytes']<=32*1024**3 and budget['address_space_limit_bytes']==32*1024**3 and budget['runtime_floor_bytes']>=50*1024**3,'small publication/AS32GiB/RSS cap/50GiB reserve required')
 require(budget['write_accounting'] in ('ADDITIONAL_TO_PLAN','INCLUDED_IN_PRIMARY_LOGS_FIGURES_AUDITS_AND_M_EXTRA'),'explicit external joint-pool write accounting required')
 _reference_path(budget['capacity_plan'],pins)
 require(audit.safe(q['output'],root=ROOT)==canonical_output(q),'single canonical source/input/budget-bound summary namespace required')
 pins.check();return q,pins,budget


def _seal_map(values,pins):
 require(type(values) is dict and values,'nonempty original immutable SHA inventory required')
 for path,digest in values.items():pins.add(path,digest)


def _source_role_pins(reader,request,initial,pins):
 """Rebind saved acceptance's six-role digests; no models/images/feature math."""
 job=request['job'];ident=reader.identity
 roles=('train','source_cal','source_test','target_full','target_cal','target_test') if job['route']=='D' else ('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST')
 expected={h+'/'+r for h in job['runs'] for r in roles}
 require(set(ident['cache_completion_sha256'])==expected and len(expected)==6,'exact six accepted native role markers required')
 def fixed(path,digest):
  require(initial.get(str(path))==digest,'accepted original role/source/head/chunk digest missing');pins.add(path,digest)
 for h,run in job['runs'].items():
  for role in roles:
   folder=ROOT/'features/source_best_M'/ident['input_admission_sha256']/run/role
   marker=folder/'complete.json';fixed(marker,ident['cache_completion_sha256'][h+'/'+role]);m=audit.json_read(marker);v=m['identity']
   require(m.get('status')=='CACHE_COMPLETE' and m.get('precision')=='FP32' and m.get('autocast') is False and v.get('checkpoint_selection')=='source_best' and v.get('parent_run_id')==run and v.get('role')==role and all(v.get(k)==x for k,x in ident['source_contexts'][h].items()),'accepted role/FP32/source-best context changed')
   manifest=ROOT/f'data/{job["route"]}'/(role+'.parquet');require(v['source_manifest_path']==str(manifest),'native manifest identity')
   fixed(manifest,v['source_manifest_sha256']);fixed(folder/'manifest.parquet',v['source_manifest_sha256'])
   head=m['head'];require(head['weight_file']=='head_weight.npy' and head['bias_file']=='head_bias.npy','accepted head files')
   for kind in ('weight','bias'):fixed(folder/head[kind+'_file'],head[kind+'_sha256'])
   require(type(m['chunks']) is list and m['chunks'],'accepted role chunks required')
   for chunk in m['chunks']:
    name=audit.artifact_name(chunk['file']);cp=audit.safe(folder/name,root=folder);fixed(cp,chunk['sha256']);meta=cp.with_suffix('.json');require(str(meta) in initial and audit.json_read(meta)==chunk,'original fixed chunk metadata required');fixed(meta,initial[str(meta)])


def load_accepted_item(item,pins):
 """Trust the externally pinned independent audit; do not rerun its science."""
 ap=_reference_path(item['audit'],pins);report=audit.json_read(ap,16*1024**2)
 rp=_reference_path(item['audit_request'],pins);request,request_pins=audit.load_request(rp,item['audit_request']['sha256'])
 require(request['job']==item['job'],'accepted original request names another job')
 validate_audit_report(report,item['job'],item['selected_prefix'])
 initial=report.get('initial_fixed_sha256');result_initial=report.get('result_initial_fixed_sha256')
 require(type(initial) is dict and initial,'original accepted source identity/pin inventory required')
 for path,digest in initial.items():
  audit.safe(path);audit.hash_value(digest)
  # Historical original weights/images are not consumed or scanned again.
  # The independently accepted proof documents themselves remain fixed/current.
  if Path(path).suffix=='.json':pins.add(path,digest)
 _seal_map(result_initial,pins)
 require(report.get('consumer_dependencies_sha256')==audit.dependency_pins(),'accepted audit dependency inventory differs')
 require(all(initial.get(p)==h for p,h in request_pins.expected.items()),'accepted audit omitted its original external request/source pins')
 for p,h in request_pins.expected.items():pins.add(p,h)
 for ref in request['proofs'].values():
  p=audit.safe(ref['path'],file=True);require(initial.get(str(p))==ref['sha256'],'accepted audit omitted actual producer/admission proof');pins.add(p,ref['sha256'])
 require(report.get('external_complete_sha256')==request['complete_sha256'],'accepted complete digest differs from original request')
 reader=audit.ReadOnlyArtifacts(request['output'],request['complete_sha256'],route=item['job']['route'])
 require(reader.pins.expected==result_initial,'every accepted result payload/receipt digest must remain identical')
 ident=reader.identity;package=audit.producer_package_codes()
 require(ident.get('schema')=='M_STATISTIC_STATIC_IDENTITY_V1' and ident.get('scope')=='SELECTED_FOUR_CORE_SOURCE_BEST_DESCRIPTIVE_SENSITIVITY' and ident.get('job')==item['job'] and ident.get('checkpoint_selection')=='source_best' and ident.get('CAL_repeats')==200 and ident.get('TEST_repeats')==2000 and ident.get('resampling_seed')==20261023 and ident.get('update_matched_primary_claim') is False and ident.get('code_sha256')==package,'accepted producer/source-best whole-code identity differs')
 require(all(initial.get(p)==h for p,h in package.items()),'accepted producer whole-package code pins missing')
 for p,h in package.items():pins.add(p,h)
 require(ident['input_admission_sha256']==request['proofs']['input_admission']['sha256'],'accepted source input binding differs')
 expected=ROOT/'analysis/source_best_M'/hashlib.sha256(audit.typed(ident)).hexdigest()/item['job']['job_id']
 require(reader.output==expected and request['output']==str(expected),'canonical accepted original result required')
 execution=audit.json_read(request['proofs']['producer_execution']['path'])
 require(execution.get('schema')=='M_STATISTIC_EXECUTION_RETURN_V1' and execution.get('status')=='COMPLETE_DECLARED_TRANSPORT_PENDING_INDEPENDENT_SAVED_RESULT_AUDIT' and execution.get('job')==item['job'] and execution.get('output')==str(expected) and execution.get('complete_sha256')==reader.complete_sha256 and execution.get('identity_sha256')==expected.parent.name and execution.get('independent_cache_admission_sha256')==request['proofs']['cache_admission']['sha256'] and execution.get('production_permit_sha256')==request['proofs']['production_permit']['sha256'],'actual saved producer completion differs')
 _source_role_pins(reader,request,initial,pins)
 selection=audit.check_selection(item['job'],ident['source_contexts'],reader['limitations.json'])
 require(selection==item['selected_prefix'],'accepted actual source-best prefix disclosure differs')
 reader.check();return report,reader,ident['input_admission_sha256']


def _resource_module(name):
 _check_resource_codes()
 try:
  module=importlib.import_module('reliability_v2.'+name)
  require(_canonical_file(module.__file__)==CODE/'reliability_v2'/(name+'.py'),'reviewed resource module origin differs')
 finally:_check_resource_codes()
 return module

def _remaining_required(root,plan):
 module=_resource_module('source_best_analysis')
 try:return module.remaining_required(root,plan)
 finally:_check_resource_codes()

def _free_bytes(root):return shutil.disk_usage(root).free

@contextmanager
def _actual_slot(root,slot):
 module=_resource_module('source_best_runtime')
 try:
  with module.cpu_slot(root,slot):
   _check_resource_codes();yield
 finally:_check_resource_codes()


def resource_gate(request,budget,pins,input_sha):
 pins.check();plan=audit.json_read(budget['capacity_plan']['path'],4*1024**2)
 require(plan.get('input_admission_sha256')==input_sha and type(plan.get('runtime_floor_bytes')) is int and plan['runtime_floor_bytes']>=budget['runtime_floor_bytes'],'same source input/fresh full remaining plan and reserve required')
 additional=budget['max_output_bytes']+budget['temporary_peak_bytes'] if budget['write_accounting']=='ADDITIONAL_TO_PLAN' else 0
 required=_remaining_required(ROOT,plan)+additional;free=_free_bytes(ROOT)
 require(free>=required and free>=budget['runtime_floor_bytes'],'fresh joint remaining writes+summary peak+reserve not covered')
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 require(peak<=budget['max_peak_RSS_bytes'],'summary observed peak RSS exceeds external bound')
 return dict(remaining_required_including_summary_bytes=required,additional_summary_reservation_bytes=additional,write_accounting=budget['write_accounting'],free_bytes_observed=free,runtime_floor_bytes=budget['runtime_floor_bytes'],summary_output_cap_bytes=budget['max_output_bytes'],summary_temporary_peak_bytes=budget['temporary_peak_bytes'],observed_peak_RSS_bytes=peak)


def _csv_bytes(rows):
 fields=list(dict.fromkeys(k for row in rows for k in row));stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
 for row in rows:
  writer.writerow({k:('' if v is None else audit.typed(v).decode() if isinstance(v,(dict,list)) else repr(v) if type(v) is float else v) for k,v in row.items()})
 return stream.getvalue().encode()


def _independent_csv_check(seed_bytes,summary_bytes,rows,summary,backbone):
 """Wire round-trip plus independent standard-library ddof1 arithmetic."""
 saved=list(csv.DictReader(io.StringIO(seed_bytes.decode())));means=list(csv.DictReader(io.StringIO(summary_bytes.decode())))
 require(len(saved)==355 and len(means)==71,'full canonical small CSV row inventory')
 require(_csv_bytes(rows)==seed_bytes and _csv_bytes(summary)==summary_bytes,'canonical generated CSV differs')
 grouped=defaultdict(dict);reversal_pairs=defaultdict(dict)
 for original,wire in zip(rows,saved):
  value=None if wire['estimate']=='' else float(wire['estimate']);require(value==original['estimate'],'seed estimate lost in CSV round-trip')
  for k in KEYS:require(wire[k]==original[k],'endpoint identity lost in CSV')
  require(json.loads(wire['selected_prefix'])==original['selected_prefix'],'source prefix disclosure lost in CSV')
  for k in ('TEST_ci','TEST_reference','CAL_reference'):
   require((None if wire[k]=='' else json.loads(wire[k]))==original[k],'conditional TEST/CAL reference lost in CSV')
  key=tuple(wire[k] for k in KEYS);grouped[key][int(wire['seed'])]=value
  if wire['core']=='RQ3':
   pair=tuple(None if wire[k]=='' else float(wire[k]) for k in ('delta_correct','delta_collision'));reversal_pairs[key][int(wire['seed'])]=pair
   require(all((None if wire[k]=='' else float(wire[k]))==original[k] for k in ('delta_correct','delta_collision')),'saved RQ3 delta pair differs')
 for row in means:
  values=grouped[tuple(row[k] for k in KEYS)];require(set(values)==set(SEEDS) and row['backbone']==backbone and row['ddof']=='1' and row['registered_n']=='5','five distinct training seeds required in saved CSV')
  finite=[v for v in values.values() if v is not None]
  require(int(row['finite_n'])==len(finite),'saved NA count differs')
  counts=dict(positive_count=sum(v>1e-10 for v in finite),negative_count=sum(v < -1e-10 for v in finite),zero_count=sum(abs(v)<=1e-10 for v in finite),na_count=5-len(finite),sign_valid_count=len(finite),sign_unknown_count=5-len(finite))
  require(float(row['sign_tolerance'])==1e-10 and all(row[k]==str(v) for k,v in counts.items()),'independent saved CSV descriptive sign/NA counts differ')
  if row['core']=='RQ3':
   pairs=list(reversal_pairs[tuple(row[k] for k in KEYS)].values());known=[(a,b) for a,b in pairs if a is not None and b is not None]
   reversed_n=sum((a>1e-10 and b < -1e-10) or (a < -1e-10 and b>1e-10) for a,b in known)
   checks=dict(reversal_count=reversed_n,non_reversal_count=len(known)-reversed_n,reversal_valid_count=len(known),reversal_unknown_count=5-len(known))
   require(float(row['reversal_tolerance'])==1e-10 and all(row[k]==str(v) for k,v in checks.items()),'independent saved CSV RQ3 paired-delta reversal counts differ')
  if len(finite)<5:require(row['mean']==row['sample_sd']=='' and row['status']=='UNAVAILABLE_NA_REGISTERED_SEED','missing seed NA was averaged away')
  else:
   require(float(row['mean'])==statistics.mean(finite) and math.isclose(float(row['sample_sd']),statistics.stdev(finite),rel_tol=1e-15,abs_tol=1e-15),'independent saved CSV five-seed mean/sample SD differ')


def _write_new(path,data):
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def _namespace_complete(output,files):
 require(output.is_dir() and {p.name for p in output.iterdir()}==set(files),'existing summary must be a complete exact namespace; partial output preserved')
 for name,data in files.items():
  p=audit.safe(output/name,file=True,root=output);require(p.read_bytes()==data,'append-only summary result differs')


def _allocation_bytes(output):
 """Charge every namespace directory/file by blocks AND rounded logical size."""
 output=Path(output);fs=os.statvfs(output);unit=fs.f_frsize or fs.f_bsize
 require(type(unit) is int and unit>0,'positive filesystem allocation unit required')
 def charge(path):
  p=audit.safe(path,root=output);st=p.stat()
  require(stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode),'regular summary namespace member required')
  return max(((st.st_size+unit-1)//unit)*unit,st.st_blocks*512)
 total=charge(output)
 for path in output.rglob('*'):total+=charge(path)
 return total


def _allocation_guard(output,cap,reserved_bytes=0):
 fs=os.statvfs(output);unit=fs.f_frsize or fs.f_bsize
 allocated=_allocation_bytes(output);reserved=((reserved_bytes+unit-1)//unit)*unit
 require(allocated+reserved<=cap,'actual summary allocation cap exceeded; existing files preserved')
 return allocated


def execute_summary(request_path,request_sha256,slot):
 q,pins,budget=load_summary_request(request_path,request_sha256)
 require(type(slot) is int and slot in (0,1),'original shared CPUslot0/1 required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only emptyCUDA/one thread required')
 soft,_=resource.getrlimit(resource.RLIMIT_AS);require(0<soft<=budget['address_space_limit_bytes'],'actual AS32GiB-or-lower bound required')
 with _actual_slot(ROOT,slot):
  capacity=audit.json_read(budget['capacity_plan']['path'],4*1024**2)
  resource_gate(q,budget,pins,capacity.get('input_admission_sha256'))
  rows=[];readers=[];input_sha=None;historical=[]
  # Digest rebinding and result decoding both occur within the original slot.
  for item in sorted(q['audits'],key=lambda x:x['job']['job_id']):
   report,reader,source=load_accepted_item(item,pins)
   require(input_sha is None or input_sha==source,'ten jobs must share the identical source input capsule');input_sha=source
   rows.extend(extract_core(item['job'],report,reader));readers.append(reader)
   historical.append(dict(job_id=item['job']['job_id'],accepted_audit=item['audit'],original_non_consumed_fixed_sha256={p:h for p,h in report.get('initial_fixed_sha256',{}).items() if p not in pins.expected}))
  rows.sort(key=lambda r:(tuple(r[k] for k in KEYS),r['seed']))
  summary=summarize_rows(rows,q['backbone']);resource_gate(q,budget,pins,input_sha)
  seed_bytes,summary_bytes=_csv_bytes(rows),_csv_bytes(summary)
  _independent_csv_check(seed_bytes,summary_bytes,rows,summary,q['backbone'])
  tables={'per_seed_core.csv':seed_bytes,'five_seed_core.csv':summary_bytes}
  manifest=dict(schema=SCHEMA,status='DESCRIPTIVE_SOURCE_BEST_FIVE_SEED_TABLES',backbone=q['backbone'],registered_seeds=list(SEEDS),request={'path':str(audit.safe(request_path,file=True)),'sha256':request_sha256},inputs_sha256=inputs_identity(q),budget=q['budget'],source_input_admission_sha256=input_sha,audits=q['audits'],dependencies_sha256=q['dependencies_sha256'],tables={name:hashlib.sha256(data).hexdigest() for name,data in tables.items()},row_counts={'per_seed_core':355,'five_seed_core':71},training_seed_statistics=dict(n=5,ddof=1,uncertainty='descriptive training-seed sample SD; unavailable if any registered value is NA',training_seeds_resampled=False,separate_backbones=True,sign_counts='descriptive positive/negative/zero/NA over5 registered seed points; tolerance1e-10',RQ3_reversal_counts='opposite correct/collision delta signs at tolerance1e-10; finite pair valid,NA pair unknown; not interaction sign',counts_are_significance_tests=False),conditional_TEST=dict(repeats=2000,intervals='per-seed pointwise conditional intervals referenced independently; never averaged to a training-seed CI',RQ2=DERIVATION,RQ2_original_delta_CI_stored=False),CAL=dict(subsets=200,layer='CAL_subset_spread_on_fixed_TEST',not_training_models=True,distributions_referenced_in_per_seed_table=True),scope='SELECTED_FOUR_CORE_SOURCE_BEST_DESCRIPTIVE_SENSITIVITY',update_matched_primary_claim=False,scientific_final_acceptance=False,new_resampling_or_models=False,original_results_read_only=True,source_files_rebound_by_digest_only=True,upstream_source_evidence=dict(scope='prior non-consumed source pins retained as historical accepted evidence, not freshly rehashed',jobs=historical),current_consumed_fixed_sha256=dict(pins.expected))
  tables['manifest.json']=audit.typed(manifest)
  completion=dict(schema=SCHEMA,status='COMPLETE_DESCRIPTIVE_SOURCE_BEST_FIVE_SEED_SUMMARY',identity_sha256=Path(q['output']).name,files={name:hashlib.sha256(data).hexdigest() for name,data in tables.items()},scientific_final_acceptance=False)
  tables['complete.json']=audit.typed(completion)
  require(sum(len(data) for data in tables.values())+64*1024<=budget['max_output_bytes'],'small summary logical/allocation overhead cap exceeded')
  output=audit.safe(q['output'],root=ROOT)
  def guard():
   for reader in readers:reader.check()
   receipt=resource_gate(q,budget,pins,input_sha)
   if output.exists():receipt['summary_allocated_bytes']=_allocation_guard(output,budget['max_output_bytes'])
   return receipt
  guard()
  if output.exists():_namespace_complete(output,tables)
  else:
   # No overwrite or recovery deletion: a failed partial namespace stays visible.
   output.parent.mkdir(parents=True,exist_ok=True);audit.safe(output.parent,root=ROOT);output.mkdir()
   for name,data in tables.items():
    if name=='complete.json':
     guard()
     _allocation_guard(output,budget['max_output_bytes'],len(data))
    _write_new(output/name,data)
   _namespace_complete(output,tables)
   _allocation_guard(output,budget['max_output_bytes'])
  receipt=guard()
  return dict(schema=SCHEMA,status=completion['status'],output=str(output),complete_sha256=hashlib.sha256(tables['complete.json']).hexdigest(),manifest_sha256=hashlib.sha256(tables['manifest.json']).hexdigest(),summary_executed=True,scientific_final_acceptance=False,training_seed_statistics=manifest['training_seed_statistics'],resource_observation=receipt,lifetime_shared_CPU_slot=slot,all_consumed_SHA_rechecked_before_publication=True,historical_non_consumed_source_pins_freshly_rehashed=False)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--request');parser.add_argument('--request-sha');parser.add_argument('--slot',type=int,choices=(0,1));args=parser.parse_args(argv)
 if not args.execute:return dict(schema=SCHEMA,status='PREPARE_ONLY',summary_executed=False,scientific_final_acceptance=False,three_backbones_reported_separately=True,registered_seeds=list(SEEDS),request_schema=REQUEST_SCHEMA,dependencies_sha256=dependencies())
 if not all((args.request,args.request_sha,args.slot is not None)):parser.error('execute requires request SHA and original shared CPU slot')
 return execute_summary(args.request,args.request_sha,args.slot)

if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,sort_keys=True,allow_nan=False))
