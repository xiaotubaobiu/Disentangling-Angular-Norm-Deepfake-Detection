"""Independent public-path surrogate-root tests. No real P/audit/slots/models/images.
The accepted independent kernel computes tiny <20-group cohorts (no bootstrap draws).
"""
from pathlib import Path
import importlib.util,hashlib,json,csv,math,os,sys,tempfile,copy,traceback
from decimal import Decimal
import datetime
import pyarrow as pa,pyarrow.parquet as pq
REAL=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
SRC=Path(os.environ.get('H_MATRIX_REVIEW_SOURCE',str(Path(__file__).parent/'h_last_matrix_result_audit_confirmed_bb299e51.py')))
OLD=REAL/'engineering/source_best_M/audits/h_pair_result_audit.py'
F=REAL/'qualification/H_parallel_fresh_validation';ORIGINAL=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2/reliability_v2')
OUT=Path(os.environ['H_MATRIX_REVIEW_TMP']) if 'H_MATRIX_REVIEW_TMP' in os.environ else Path(tempfile.mkdtemp(prefix='h_last_matrix_public_',dir='/tmp'))
OUT.mkdir(parents=True,exist_ok=True)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n');return H(p)
def safe(x):
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [safe(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def row(uid,label,method,score):return dict(video_uid=uid,label=label,method_id=method,group_id='g_'+uid,score=float(score),rank=Decimal.from_float(float(score)),ranking_usable=True)
def frame(kind,common=False):
 if kind=='cal':return [row('cal'+str(i),0,'real',v) for i,v in enumerate((-.8,-.3,.2,.6))]
 methods=('Deepfakes','Face2Face','FaceSwap','NeuralTextures') if kind=='SOURCE_TEST' else ('DeepFakeDetection','DeepFakeDetection')
 q=[row(kind+'_r'+str(i),0,'real',v) for i,v in enumerate((-1.,-.25))]+[row(kind+'_f'+str(i),1,m,.2+i*.3) for i,m in enumerate(methods)]
 return [x for x in q if not (common and x['video_uid']==kind+'_r1')]
def shifted(q,delta,agg):
 out=[]
 for r in q:
  v=r['score']+delta
  if agg=='mean_probability':v=1/(1+math.exp(-v))
  out.append(dict(r,score=v,rank=Decimal.from_float(v)))
 return out

def score_csv(p,q,agg):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 rows=[{k:r[k] for k in ('video_uid','label','method_id','group_id','score')}|dict(rank_score_kind='float64',rank_score_decimal=repr(r['score']),rank_score_export_digits=350,ranking_usable=True,representation='raw',grouping='correct',aggregation=agg) for r in q]
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csv_rows(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Fixture:
 def __init__(self,case,bone='clip_vit_b16'):
  self.root=Path(tempfile.mkdtemp(prefix='public_'+case+'_',dir=OUT));self.case=case;self.bone=bone;self.seed=123
  spec=importlib.util.spec_from_file_location('review_'+case,SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);self.m=m
  m.ROOT=self.root;m.HERE=self.root/'engineering/source_best_M/audits';m.F=self.root/'qualification/H_parallel_fresh_validation'
  m.base.ROOT=self.root;m.base.ORIGINAL=self.root/'original/reliability_v2';m.BASE_PATH=m.HERE/'h_pair_result_audit.py'
  # Substitute only literal filesystem anchors. Keep actual _safe logic, metadata,
  # proof loader, public main, file I/O, terminal and independent kernels unmocked.
  m._safe.__defaults__=(self.root,)
  m.HERE.mkdir(parents=True);m.F.mkdir(parents=True);m.base.ORIGINAL.mkdir(parents=True)
  m.BASE_PATH.write_bytes(OLD.read_bytes())
  source_copy=self.root/'tools/h_last_matrix_result_audit.py';source_copy.parent.mkdir();source_copy.write_bytes(SRC.read_bytes());m.__file__=str(source_copy)
  self.refs=m.canonical_refs(self.root,bone,123);self.pair=self.root/f'analysis/H/{bone}/seed123';self.pair.mkdir(parents=True)
  self.refhash={};self.pin={}
  for p in ORIGINAL.glob('*.py'):
   dest=m.base.ORIGINAL/p.name;dest.write_bytes(p.read_bytes());self.pin[str(dest)]=H(dest)
  for n in m.CPU_HASHES:
   dest=m.F/n;dest.write_bytes((F/n).read_bytes())
   if n!='cpu_primary_analysis.py':self.pin[str(dest)]=H(dest)
  self.refs['cache_audit_entry'].write_bytes((REAL/'engineering/source_best_M/audits/h_parallel_pair_admission_audit.py').read_bytes())
  self.refs['source_manifest'].parent.mkdir(parents=True);self.refs['source_manifest'].write_bytes((REAL/'qualification/H_primary_cache_continuation/qualified_source_manifest.json').read_bytes());self.pin[str(self.refs['source_manifest'])]=H(self.refs['source_manifest'])
  runids=[f'H_formal_{bone}_{h}_seed{s}' for s in m.SEEDS for h in ('B','AB')]
  inputs={}
  for rel,obj in [('configs/H_execution.json',{'schema':'H_EXECUTION_V2'}),('H_run_registry.json',dict(status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING',target_export_allowed=True,effective_runs={b:dict(precision='AMP',run_ids=[f'H_formal_{b}_{h}_seed{s}' for s in m.SEEDS for h in ('B','AB')]) for b in m.BACKBONES})),(f'qualification/H_target_export_{bone}.json',{'status':'PASS'}),(f'reports/H_{bone}_AMP_paired_ledger.json',{'status':'PASS','utc':'toy'}),('data/H/freeze.json',{'schema':'toy frozen data'})]:
   p=self.root/rel;inputs[str(p)]=dump(p,obj);self.pin[str(p)]=H(p)
  for i in range(107-len(self.pin)):
   p=self.root/f'reports/extra_pin_{i}.json';self.pin[str(p)]=dump(p,{'synthetic_pin':i})
  assert len(self.pin)==107
  self.tables={};self.records=[];markers={}
  for role in m.base.ROLES:
   kind='cal' if role=='B/SOURCE_CAL' else role.rsplit('/',1)[1];q=frame(kind,role.startswith('common/'));self.tables[role]=q
   dest=self.root/'data/H'/f'{role}.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
   pq.write_table(pa.table({k:[r[k] for r in q for _ in range(8)] for k in ('video_uid','label','method_id','group_id')}),dest)
  self.I=dict(schema='H_PAIR_ANALYSIS_IDENTITY_V1',backbone=bone,seed=123,effective_precision='AMP',run_ids={h:f'H_formal_{bone}_{h}_seed123' for h in ('B','AB')},inputs_sha256=inputs,code_sha256={n:H(m.base.ORIGINAL/n) for n in ('analysis_h.py','cache_h.py','stats_h.py','coverage_h.py','analysis_d.py','scoring_d.py','bootstrap_fast.py','cache_d.py')},cache_completion_sha256={},bootstrap_repeats=2000,resampling_seed=20261023,source_threshold='each_model_allB_SOURCE_CAL_empiricalFPR5',primary='DFD_allB_Raw_mean_logit_H10_minus_H00',test_cohorts='allB_and_common_reported_separately')
  for history,run in self.I['run_ids'].items():
   for role,q in self.tables.items():
    table=self.root/'data/H'/f'{role}.parquet';p=self.root/'features/H'/run/role/'complete.json';head=('a' if history=='B' else 'b')*64
    doc=dict(status='CACHE_COMPLETE',precision='FP32',autocast=False,input_validation='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES',rows=8*len(q),identity=dict(run_id=run,history=history,role=role,backbone=bone,dimension={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}[bone],effective_precision='AMP',checkpoint_path=str(self.root/'runs'/run/'final.pt'),original_head_sha256=head,accepted_table_videos=len(q),source_manifest_path=str(table),source_manifest_sha256=H(table)))
    h=dump(p,doc);markers[str(p)]=h;self.I['cache_completion_sha256'][history+'/'+role]=h
    self.records.append(dict(history=history,run_id=run,role=role,rows=8*len(q),accepted_videos=len(q),chunk_count=1,original_head_sha256=head,source_manifest_path=str(table),source_manifest_sha256=H(table),input_validation=doc['input_validation'],complete_sha256=h))
  dump(self.refs['identity'],self.I)
  # Distinct qualification evidence documents form a raw-SHA-bound graph.
  qbone='clip_vit_b16' if case=='wrong_Q_actual_evidence' else bone
  qstatus='FAILED' if case=='wrong_Q_actual_evidence' else 'PASS'
  qpaths={k:self.root/f'reports/Q_{k}.json' for k in ('candidate_report','independent_actual_audit','primary_entry_review')}
  qi=copy.deepcopy(self.I);qi.update(backbone=qbone,seed=42,run_ids={h:f'H_formal_{qbone}_{h}_seed42' for h in ('B','AB')})
  qld=dict(backbone=qbone,precision='AMP',fresh_source_completions=10,reference_sha256=inputs[str(self.root/f'reports/H_{bone}_AMP_paired_ledger.json')],exact_original_ledger_except_top_utc=True,cross_invocation_receipt_reuse=False)
  qc=dict(status=('PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS' if qstatus=='PASS' else 'FAILED'),backbone=qbone,seed=42,models=2,roles=14,identity=qi,pinned_sha256=self.pin,fresh_ledger_invocations=[copy.deepcopy(qld) for _ in range(14)],all_source_completions_fresh=140,cross_invocation_receipt_reuse=False,production_scoring_executed=False,GPU_launch_attempted=False,live_cache_owner_changed=False)
  qhash={'candidate_report':dump(qpaths['candidate_report'],qc)}
  qa=dict(schema='H_FRESH_PARALLEL_CPU_REAL_RECEIPT_AUDIT_V1',status=qstatus,backbone=qbone,seed=42,models=2,roles=14,receipt_path=str(qpaths['candidate_report']),receipt_sha256=qhash['candidate_report'],receipt_status=qc['status'],original_primary_pair_identity=qi,pinned_sha256=self.pin,independent_review_pass=(qstatus=='PASS'),time_v_exit_status=0,candidate_wrapper_and_child_true_exit_observed=True,fresh_original_ledger_invocations=14,fresh_default_source_checks_per_ledger=10,fresh_default_source_checks_in_ledgers=140,all_original_ledger_checks_exact_except_top_utc=True,cross_invocation_receipt_reuse=False,functional_allowed_backbones=[qbone],intended_functional_validator_use=['independent_role_admission','primary_pair_analysis'],actual_CLIP_original14role_fresh140ledger_source_receipt_audit_passed=True,implementation_sha256={n:h for n,h in m.CPU_HASHES.items() if n!='cpu_primary_analysis.py'})
  qhash['independent_actual_audit']=dump(qpaths['independent_actual_audit'],qa)
  engineering=dict(status='INDEPENDENT_PRIMARY_ANALYSIS_ENTRY_FIX_CONFIRMED_NO_REMAINING_ENGINEERING_REVIEW_BLOCKER',engineering_entry_review_passed=True,candidate_sha256=m.CPU_HASHES,remaining_findings=[],critical_findings=0,important_findings=0,actual_CPU_qualification_accepted=False,resource_admission_accepted=False,primary_scientific_results_accepted=False,production_integration_accepted=False)
  qhash['primary_entry_review']=dump(qpaths['primary_entry_review'],engineering)
  Q=dict(schema='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1',status='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE',allowed_backbones=[bone],allowed_use=['independent_role_admission','primary_pair_analysis'],original_role_checks=14,fresh_default_source_checks=140,all_original_ledger_checks_exact_except_utc=True,independent_review_pass=True,production_GPU_allowed=False,implementation_sha256=m.CPU_HASHES,original_source_manifest_sha256=H(self.refs['source_manifest']))
  for k,p in qpaths.items():Q[k]=str(p);Q[k+'_sha256']=qhash[k]
  dump(self.refs['CPU_qualification'],Q)
  ld=dict(backbone=bone,precision='AMP',fresh_source_completions=10,reference_sha256=inputs[str(self.root/f'reports/H_{bone}_AMP_paired_ledger.json')],exact_original_ledger_except_top_utc=True,cross_invocation_receipt_reuse=False,elapsed_seconds=.01)
  C=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',backbone=bone,seed=123,identity=self.I,models=2,roles=14,pinned_sha256=self.pin,fresh_ledger_invocations=[copy.deepcopy(ld) for _ in range(14)],all_source_completions_fresh=140,cross_invocation_receipt_reuse=False,production_scoring_executed=False,GPU_launch_attempted=False,live_cache_owner_changed=False)
  dump(self.refs['cache_admission'],C)
  br={**self.pin,**markers,**{str(self.refs[k]):H(self.refs[k]) for k in ('cache_admission','primary_entry','CPU_qualification','cache_audit_entry')}}
  A=dict(schema='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1',status='PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND',backbone=bone,seed=123,candidate_identity=self.I,canonical_shared107_pin_sha256=self.pin,reference_pin_inventory_independently_reconstructed=True,qualified_original_code_count=20,canonical_proof_count=37,models=2,roles=14,fresh_ledger_invocations=14,source_checks_per_ledger=10,source_checks_in_ledger_receipts=140,lifetime_CPU_flock_held=True,bracketed_readonly_evidence_sha256=br,role_marker_checks=self.records)
  dump(self.refs['cache_audit'],A)
  self.primary_time=self.refs['primary_time_v'];self.primary_log=self.refs['primary_stdout']
  self.primary_time.write_text('Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.00\nMaximum resident set size (kbytes): 10000\n'+('Exit status: 1\n' if case in ('live_primary_exit1','producer_exit1_only') else 'Exit status: 0\n'));self.primary_log.write_text('\n'.join(f'BEGIN_PRIMARY_FRESH_LEDGER {i} {bone} AMP' for i in range(1,42 if case=='producer_count41' else 43))+'\nCOMPLETE_ORIGINAL_PRIMARY_H_ANALYSIS_PENDING_INDEPENDENT_AUDIT '+str(self.refs['primary_execution'])+'\n')
  if case=='producer_complete_first':
   lines=self.primary_log.read_text().splitlines();self.primary_log.write_text('\n'.join([lines[-1],*lines[:-1]])+'\n')
  L=dict(status='LAUNCHED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_AFTER_INDEPENDENT_CACHE_ADMISSION',backbone=bone,seed=123,pid=os.getpid() if case.startswith('live_primary') else 99999999,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout=str(self.primary_log),time_v=str(self.primary_time),CPU_slot=0,entry=str(self.refs['primary_entry']),entry_sha256=m.CPU_HASHES['cpu_primary_analysis.py'],independent_cache_audit=str(self.refs['cache_audit']),independent_cache_audit_sha256=H(self.refs['cache_audit']),admission_sha256=H(self.refs['cache_admission']),qualification_sha256=H(self.refs['CPU_qualification']),bootstrap_repeats=2000,GPUs_allowed=False,original_F_unchanged=True,M_statistics_started=False)
  dump(self.refs['primary_launch'],L)
  self.make_scores()
  pp={**self.pin,**{str(self.refs[k]):H(self.refs[k]) for k in ('cache_admission','primary_entry','CPU_qualification')}}
  if case=='missing_P_base_pin':pp.pop(str(self.root/'configs/H_execution.json'))
  self.missing_p_pin=str(self.root/'configs/H_execution.json') not in pp
  P=dict(status='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT',backbone=bone,seed=123,CPU_slot=0,source_qualification_sha256=H(self.refs['CPU_qualification']),pair_admission_sha256=H(self.refs['cache_admission']),original_analysis_identity=self.I,original_analysis_complete_sha256=H(self.pair/'complete.json'),fresh_ledger_invocations=[copy.deepcopy(ld) for _ in range(42)],pinned_sha256=pp,GPU_launch_attempted=False,primary_cache_owner_modified=False,M_results_generated=False,scientific_final_acceptance=False)
  dump(self.refs['primary_execution'],P)
  proof={k:dict(path=str(p),sha256=H(p)) for k,p in self.refs.items()}
  b=dict(schema='H_LAST_MATRIX_PAIR_AUDIT_BINDINGS_V1',root=str(self.root),backbone=bone,seed=123,source_checkpoint='original_LAST_final.pt',bootstrap_repeats=2000,resampling_seed=20261023,proofs=proof)
  self.bp=m.HERE/f'H_LAST_{bone}_seed123_matrix_bindings.json';dump(self.bp,b)
  (self.root/'analysis/H/.cpu_slots').mkdir(exist_ok=True)
  for s in (0,1):(self.root/f'analysis/H/.cpu_slots/slot{s}.lock').touch()
  self.terminal_sha=H(self.pair/'complete.json');self.calls=[]
 def make_scores(self):
  m=self.m;guards=[]
  for agg in ('mean_logit','mean_probability'):
   cal={h:shifted(self.tables['B/SOURCE_CAL'],0 if h=='B' else -.1,agg) for h in ('B','AB')}
   for h,q in cal.items():score_csv(self.pair/agg/f'source_cal_{h}.csv',q,agg)
   for domain in ('SOURCE_TEST','TARGET_TEST'):
    for cohort,names in (('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))):
     folder=self.pair/agg/domain/cohort;cells={}
     for c in names:
      version='AB' if c in ('H01','H11') else 'B';role=f'B/{domain}' if cohort=='all_B' else f'common/{version}/{domain}';cells[c]=shifted(self.tables[role],{'H00':0,'H01':.15,'H10':-.1,'H11':-.05}[c],agg);score_csv(folder/f'{c}_scores.csv',cells[c],agg)
     methods=m.base.METHODS if domain=='SOURCE_TEST' else ('DeepFakeDetection',)
     boot=m.base.recompute_cohort(cells,cal,methods,agg);_,groups,_,thresholds,ev=m.base.prepare(cells,cal,methods,agg);values=ev(m.base.np.ones(len(groups),dtype=m.base.np.int64))
     metrics=[dict(cell=c,model_history=m.base.HISTORY[c],threshold_kind=k,**v) for c,ks in values.items() for k,v in ks.items()];csv_rows(folder/'metrics.csv',metrics)
     comp=[]
     for key,v in m.base.flatten(values).items():
      if ':comparison:' in key:
       kind,_,target,metric=key.split(':',3);comp.append(dict(threshold_kind=kind,comparison=target,metric=metric,estimate=v))
     csv_rows(folder/'comparisons.csv',comp);dump(folder/'thresholds.json',thresholds);dump(folder/'conditional_bootstrap.json',boot)
     dump(folder/'independent_kernel_checks.json',[dict(cell=c,threshold_kind=k,status='PASSED',max_abs_error=0.) for c in names for k in m.base.KINDS]);guards.append(dict(aggregation=agg,role=domain,cohort=cohort,checks=[dict(cell=c,threshold_kind=k,status='PASS') for c in names for k in m.base.KINDS]))
  dump(self.pair/'independent_saved_score_recompute.json',guards);dump(self.pair/'limitations.json',dict(numerical_NA=[],training_seed_count=1,completed_model_pairs=1,five_seed_summary_pending=True));dump(self.pair/'coverage.json',{})
  inventory={n:H(self.pair/n) for n in m.base.inventory()};dump(self.pair/'complete.json',dict(schema='H_PAIR_ANALYSIS_COMPLETE_V1',status='ANALYSIS_COMPLETE',identity=self.I,backbone=self.bone,seed=123,effective_precision='AMP',independent_recompute='PASSED_FOR_ALL_REPORTABLE_CELLS',artifact_sha256=inventory))
 def execute(self):
  m=self.m;orig=m.base.check_tables
  def check(folder,cells,cal,methods,agg):
   domain=folder.parts[-2];cohort=folder.parts[-1];assert agg==folder.parts[-3]
   assert tuple(methods)==(m.base.METHODS if domain=='SOURCE_TEST' else ('DeepFakeDetection',))
   for cell,rs in cells.items():
    version='AB' if cell in ('H01','H11') else 'B';role=f'B/{domain}' if cohort=='all_B' else f'common/{version}/{domain}'
    assert m.base.identities(rs)==m.base.identities(self.tables[role])
   self.calls.append((agg,domain,cohort,tuple(cells)));return orig(folder,cells,cal,methods,agg)
  m.base.check_tables=check
  argv=['--execute','--binding-spec',str(self.bp),'--binding-spec-sha',H(self.bp),'--terminal-sha',self.terminal_sha,'--slot','1']
  if self.case.startswith('live_primary'):
   with m.base.cpu_slot(self.root/'analysis/H/.cpu_slots',0):result=m.main(argv)
  else:result=m.main(argv)
  assert len(self.calls)==8 and len(set(self.calls))==8
  return dict(case=self.case,root=str(self.root),public_status=result['status'],output=result['output'],routing_calls=self.calls,producer_time_text=self.primary_time.read_text().strip(),producer_pid_alive=self.case.startswith('live_primary'),P_missing_base_config_pin=self.missing_p_pin,torch_imported='torch' in sys.modules)

if __name__=='__main__':
 out=[]
 for case,bone in [('baseline','clip_vit_b16'),('missing_P_base_pin','clip_vit_b16'),('wrong_Q_actual_evidence','dinov2_vit_b14'),('live_primary_only','clip_vit_b16'),('producer_exit1_only','clip_vit_b16'),('producer_count41','clip_vit_b16'),('producer_complete_first','clip_vit_b16')]:
  try:out.append(Fixture(case,bone).execute())
  except BaseException as e:out.append(dict(case=case,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
 assert out[0].get('public_status')=='SAVED_RESULT_NUMERICS_AND_BOUND_PRIOR_ROLE_ADMISSION_PASS' and len(out[0]['routing_calls'])==8
 assert all(x.get('error_type')=='AssertionError' for x in out[1:]),'a required public negative was accepted'
 print(json.dumps(out,indent=2));(OUT/'public_results.json').write_text(json.dumps(out,indent=2)+'\n')
