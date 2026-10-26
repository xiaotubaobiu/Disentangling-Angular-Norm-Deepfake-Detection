"""Remaining original H LAST pair audit; reuses the accepted independent numeric kernel.
No producer/scoring/torch imports, model execution, pixel checks or resampling.
Existing seed7/42 auditor and accepted outputs stay byte-identical.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,re,time
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');HERE=ROOT/'engineering/source_best_M/audits';F=ROOT/'qualification/H_parallel_fresh_validation'
BASE_PATH=HERE/'h_pair_result_audit.py';BASE_SHA='6cd3bff403cc2f7c27a870d092023e4c8677f5d9386668abbc439829eadaa4e5'
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()==BASE_SHA,'accepted independent kernel source changed'
s=importlib.util.spec_from_file_location('_accepted_original_H_numeric_6cd3bff4',BASE_PATH);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
require=base.require;sha=base.sha
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0');SEEDS=(7,42,123,2026,3407)
CPU_HASHES={'parallel_source.py':'a6133c824913ab3bf361095b415eefc9ef75be39ab32fb49ab62e2656dc2624a','cpu_ledger_adapter.py':'c6b98535e5a28123baae595ab1f4b208f1feab43ff4018e740be94366d044287','cpu_ledger_admission.py':'bd202e85a4990f9246b96a7081815d07cf364041e5d3498f49bfe41d8779b213','cpu_primary_analysis.py':'c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce'}

def canonical_refs(root,bone,seed):
 root=Path(root);label={'clip_vit_b16':'CLIP','dinov2_vit_b14':'DINO','efficientnet_b0':'EFF'}[bone]
 return dict(identity=root/f'analysis/H/{bone}/seed{seed}/identity.json',cache_admission=root/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json',cache_audit=root/f'engineering/source_best_M/audits/H_{label}_seed{seed}_parallel_cache_admission_independent_audit.json',cache_audit_entry=root/'engineering/source_best_M/audits/h_parallel_pair_admission_audit.py',primary_launch=root/f'reports/H_primary_analysis_{bone}_seed{seed}_launch.json',primary_execution=root/f'reports/H_primary_analysis_execution_{bone}_seed{seed}.json',primary_entry=root/'qualification/H_parallel_fresh_validation/cpu_primary_analysis.py',CPU_qualification=root/f'reports/H_parallel_CPU_actual_qualification_{bone}.json',source_manifest=root/'qualification/H_primary_cache_continuation/qualified_source_manifest.json',primary_stdout=root/f'reports/H_primary_analysis_{bone}_seed{seed}.log',primary_time_v=root/f'reports/H_primary_analysis_{bone}_seed{seed}_time_v.log')

def _int(v,n):return type(v) is int and v==n

def validate_ledgers(rows,count,bone,precision,reference):
 require(type(rows) is list and len(rows)==count,'exact fresh ledger invocation count')
 for d in rows:require(type(d) is dict and _int(d.get('fresh_source_completions'),10) and d.get('backbone')==bone and d.get('precision')==precision and d.get('reference_sha256')==reference and d.get('exact_original_ledger_except_top_utc') is True and d.get('cross_invocation_receipt_reuse') is False,'fresh typed10 exact source ledger required')

def validate_chain(root,bone,seed,refs,I,C,A,L,P,Q,*,terminal_sha):
 root=Path(root);require(bone in BACKBONES and type(seed) is int and seed in SEEDS,'registered3bone/five-seed original H pair only')
 require(I.get('schema')=='H_PAIR_ANALYSIS_IDENTITY_V1' and I.get('backbone')==bone and _int(I.get('seed'),seed) and _int(I.get('bootstrap_repeats'),2000) and _int(I.get('resampling_seed'),20261023),'frozen result identity')
 require(I.get('run_ids')=={h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')},'original LAST effective pair run IDs')
 require(C.get('status')=='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS' and C.get('identity')==I and _int(C.get('models'),2) and _int(C.get('roles'),14) and _int(C.get('all_source_completions_fresh'),140) and C.get('cross_invocation_receipt_reuse') is False,'original fresh14role/140source admission')
 precision=I['effective_precision'];reference=I['inputs_sha256'][str(root/f'reports/H_{bone}_{precision}_paired_ledger.json')];validate_ledgers(C['fresh_ledger_invocations'],14,bone,precision,reference)
 require(A.get('schema')=='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1' and A.get('status')=='PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND' and A.get('backbone')==bone and _int(A.get('seed'),seed) and A.get('candidate_identity')==I,'independent cache admission identity')
 require(A.get('canonical_shared107_pin_sha256')==C['pinned_sha256'] and len(C['pinned_sha256'])==107 and A.get('reference_pin_inventory_independently_reconstructed') is True and _int(A.get('qualified_original_code_count'),20) and _int(A.get('canonical_proof_count'),37),'independently reconstructed exact107 source/proof pins')
 for k,n in [('models',2),('roles',14),('fresh_ledger_invocations',14),('source_checks_per_ledger',10),('source_checks_in_ledger_receipts',140)]:require(_int(A.get(k),n),'typed independent admission count '+k)
 require(A.get('lifetime_CPU_flock_held') is True,'completed independent admission audit slot proof')
 require(Q.get('schema')=='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1' and Q.get('status')=='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE' and type(Q.get('allowed_backbones')) is list and bone in Q['allowed_backbones'] and type(Q.get('allowed_use')) is list and 'primary_pair_analysis' in Q['allowed_use'] and Q.get('all_original_ledger_checks_exact_except_utc') is True and Q.get('independent_review_pass') is True and Q.get('production_GPU_allowed') is False and _int(Q.get('original_role_checks'),14) and _int(Q.get('fresh_default_source_checks'),140),'actual CPU qualification for this backbone/use required')
 require(Q.get('implementation_sha256')==CPU_HASHES,'qualified four-file CPU implementation family')
 require(L.get('status')=='LAUNCHED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_AFTER_INDEPENDENT_CACHE_ADMISSION' and L.get('backbone')==bone and _int(L.get('seed'),seed) and _int(L.get('bootstrap_repeats'),2000) and L.get('GPUs_allowed') is False and L.get('original_F_unchanged') is True and L.get('M_statistics_started') is False,'actual original primary launch')
 for key in ('cache_admission','CPU_qualification'):
  lk='admission_sha256' if key=='cache_admission' else 'qualification_sha256';pk='pair_admission_sha256' if key=='cache_admission' else 'source_qualification_sha256';require(L.get(lk)==P.get(pk)==refs[key]['sha256'],'producer/launch raw proof SHA '+key)
 require(L.get('independent_cache_audit')==refs['cache_audit']['path'] and L.get('independent_cache_audit_sha256')==refs['cache_audit']['sha256'],'launch binds accepted independent cache audit')
 require(L.get('entry')==refs['primary_entry']['path'] and L.get('entry_sha256')==CPU_HASHES['cpu_primary_analysis.py'],'qualified primary entry identity')
 expected_pins=dict(C['pinned_sha256'])
 for key in ('CPU_qualification','cache_admission','primary_entry'):
  r=refs[key];require(r['path'] not in expected_pins or expected_pins[r['path']]==r['sha256'],'conflicting producer pin');expected_pins[r['path']]=r['sha256']
 require(P.get('pinned_sha256')==expected_pins and len(expected_pins)==110,'producer must retain exact107 plus qualification/admission/entry110 pins')
 require(L.get('entry_sha256')==refs['primary_entry']['sha256'],'external primary entry proof pin')
 require(P.get('status')=='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT' and P.get('backbone')==bone and _int(P.get('seed'),seed) and P.get('original_analysis_identity')==I and P.get('original_analysis_complete_sha256')==terminal_sha,'actual producer terminal and identity')
 require(type(P.get('CPU_slot')) is int and P['CPU_slot'] in (0,1) and P['CPU_slot']==L.get('CPU_slot') and P.get('GPU_launch_attempted') is False and P.get('primary_cache_owner_modified') is False and P.get('M_results_generated') is False,'CPU-only original execution')
 validate_ledgers(P['fresh_ledger_invocations'],42,bone,precision,reference)
 for name,pin in CPU_HASHES.items():require(P['pinned_sha256'].get(str(root/'qualification/H_parallel_fresh_validation'/name))==pin,'producer qualified CPU source pin')
 for key in ('cache_admission','primary_entry','CPU_qualification','cache_audit_entry'):require(A['bracketed_readonly_evidence_sha256'].get(refs[key]['path'])==refs[key]['sha256'],'independent admission proof lifetime '+key)
 return True


def validate_qualification_evidence(bone,Q,C,A,R,*,precision):
 I=C.get('identity',{});seed=I.get('seed');require(C.get('status')=='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS' and I.get('schema')=='H_PAIR_ANALYSIS_IDENTITY_V1' and I.get('backbone')==bone and type(seed) is int and seed in SEEDS and I.get('effective_precision')==precision and I.get('run_ids')=={h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')},'actual qualification prototype must be its own registered same-backbone identity')
 require(_int(C.get('models'),2) and _int(C.get('roles'),14) and _int(C.get('all_source_completions_fresh'),140) and C.get('cross_invocation_receipt_reuse') is False,'actual qualification prototype full14/140')
 keys=[k for k in I['inputs_sha256'] if Path(k).name==f'H_{bone}_{precision}_paired_ledger.json'];require(len(keys)==1,'qualification ledger reference');validate_ledgers(C['fresh_ledger_invocations'],14,bone,precision,I['inputs_sha256'][keys[0]])
 require(A.get('schema')=='H_FRESH_PARALLEL_CPU_REAL_RECEIPT_AUDIT_V1' and A.get('status')=='PASS' and A.get('receipt_path')==Q['candidate_report'] and A.get('receipt_sha256')==Q['candidate_report_sha256'] and A.get('receipt_status')==C['status'] and A.get('original_primary_pair_identity')==I and A.get('backbone')==bone and _int(A.get('seed'),seed),'independent actual qualification audit must bind the passed prototype')
 require(A.get('independent_review_pass') is True and _int(A.get('time_v_exit_status'),0) and A.get('candidate_wrapper_and_child_true_exit_observed') is True and A.get('all_original_ledger_checks_exact_except_top_utc') is True and A.get('cross_invocation_receipt_reuse') is False,'actual qualification execution/freshness accepted')
 for key,n in [('models',2),('roles',14),('fresh_original_ledger_invocations',14),('fresh_default_source_checks_per_ledger',10),('fresh_default_source_checks_in_ledgers',140)]:require(_int(A.get(key),n),'actual qualification typed count '+key)
 require(A.get('pinned_sha256')==C.get('pinned_sha256') and len(C['pinned_sha256'])==107 and bone in A.get('functional_allowed_backbones',[]) and 'primary_pair_analysis' in A.get('intended_functional_validator_use',[]),'actual qualification independent inventory/use/backbone')
 impl=A.get('implementation_sha256',{});require(set(CPU_HASHES)-{'cpu_primary_analysis.py'}<=set(impl)<=set(CPU_HASHES) and all(Q['implementation_sha256'][n]==h for n,h in impl.items()),'actual prototype implementation pins')
 require(R.get('status')=='INDEPENDENT_PRIMARY_ANALYSIS_ENTRY_FIX_CONFIRMED_NO_REMAINING_ENGINEERING_REVIEW_BLOCKER' and R.get('engineering_entry_review_passed') is True and all(R.get('candidate_sha256',{}).get(n)==h for n,h in CPU_HASHES.items()),'separate qualified primary-entry engineering review')

def validate_producer_terminal(root,L,refs,precision):
 root=Path(root);pid=L.get('pid');require(type(pid) is int and pid>0,'observed producer wrapperPID required');require(L.get('stdout')==refs['primary_stdout']['path'] and L.get('time_v')==refs['primary_time_v']['path'],'external producer log/time references')
 for key in ('primary_stdout','primary_time_v'):
  path=Path(refs[key]['path']);require(path.is_relative_to(root) and path.resolve().is_relative_to(root) and not any(q.is_symlink() for q in (path,*path.parents)) and path.is_file() and path.stat().st_size<=1024**2,'bounded canonical producer witness')
 proc=Path('/proc')/str(pid)/'stat';state='absent'
 if proc.exists():
  fields=proc.read_text().rsplit(')',1)[1].split();state=fields[0]
  if state!='Z':
   boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));born=boot+int(fields[19])/os.sysconf('SC_CLK_TCK');launched=datetime.datetime.fromisoformat(L['utc'].replace('Z','+00:00')).timestamp();require(born>launched+5,'original producer wrapper still live or launch identity uncertain');state='PID_reused_after_original_launch'
 timing=Path(L['time_v']).read_text();require(re.findall(r'^\s*Exit status:\s*(\d+)\s*$',timing,re.M)==['0'],'producer actual time_v must have unique exit0')
 lines=Path(L['stdout']).read_text().splitlines();pattern=rf'^BEGIN_PRIMARY_FRESH_LEDGER (\d+) {re.escape(L["backbone"])} {re.escape(precision)}$';numbers=[int(x[1]) for line in lines if (x:=re.fullmatch(pattern,line))];require(numbers==list(range(1,43)),'producer stdout must retain all42 ordered actual ledger starts');complete='COMPLETE_ORIGINAL_PRIMARY_H_ANALYSIS_PENDING_INDEPENDENT_AUDIT '+refs['primary_execution']['path'];require(sum(x==complete for x in lines)==1,'producer actual final completion witness')
 return dict(wrapper_pid=pid,observed_process_state=state,time_v_exit_status=0,ordered_actual_ledger_starts=42)

def _safe(path,root=ROOT):
 p=Path(path).absolute();allowed=[Path(root),base.ORIGINAL]
 require(any(p.is_relative_to(r) and p.resolve().is_relative_to(r) for r in allowed),'canonical audit input namespace')
 require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink in audit input namespace');return p

def _json(path):
 p=_safe(path);require(p.is_file() and p.stat().st_size<=4*1024**2,'bounded regular audit metadata');return base.read_json(p)

def load_spec(path,pin):
 p=_safe(path);require(sha(p)==pin,'external binding specification SHA');b=_json(p);require(set(b)=={'schema','root','backbone','seed','source_checkpoint','bootstrap_repeats','resampling_seed','proofs'} and b['schema']=='H_LAST_MATRIX_PAIR_AUDIT_BINDINGS_V1' and b['root']==str(ROOT),'matrix binding field inventory')
 bone,seed=b['backbone'],b['seed'];require(bone in BACKBONES and type(seed) is int and seed in SEEDS and b['source_checkpoint']=='original_LAST_final.pt' and _int(b['bootstrap_repeats'],2000) and _int(b['resampling_seed'],20261023),'fixed original LAST binding scope')
 require(p==HERE/f'H_LAST_{bone}_seed{seed}_matrix_bindings.json','canonical matrix binding path');defaults=canonical_refs(ROOT,bone,seed);require(set(b['proofs'])==set(defaults),'exact matrix proof inventory')
 for key,r in b['proofs'].items():
  require(type(r) is dict and set(r)=={'path','sha256'} and type(r['sha256']) is str and re.fullmatch('[0-9a-f]{64}',r['sha256']),'proof path/SHA '+key);q=_safe(r['path'])
  if key in ('identity','primary_execution','primary_entry','source_manifest','cache_audit_entry'):require(q==defaults[key],'canonical proof '+key)
  elif key=='primary_launch':require(q.parent==ROOT/'reports' and q.name.startswith(f'H_primary_analysis_{bone}_seed{seed}_launch') and q.suffix=='.json','canonical launch')
  elif key=='CPU_qualification':require(q.parent==ROOT/'reports' and q.name.startswith(f'H_parallel_CPU_actual_qualification_{bone}') and q.suffix=='.json','canonical CPU qualification')
  elif key in ('primary_stdout','primary_time_v'):require(q.parent==ROOT/'reports' and q.name.startswith(f'H_primary_analysis_{bone}_seed{seed}') and q.suffix=='.log','canonical actual producer witness')
  elif key=='cache_admission':require(q.parent==ROOT/'reports' and q.name.startswith(f'H_{bone}_seed{seed}_independent_7role_parallel_candidate_') and q.suffix=='.json','canonical pair admission')
  else:require(q.parent==HERE and re.search(f'_seed{seed}_',q.name) and q.suffix=='.json','canonical independent admission audit')
 return b

def _metadata(spec,terminal_sha):
 refs=spec['proofs'];docs={}
 for key,r in refs.items():
  require(sha(_safe(r['path']))==r['sha256'],'external matrix proof drift '+key)
  if key not in ('primary_entry','cache_audit_entry','primary_stdout','primary_time_v'):docs[key]=_json(r['path'])
 I=docs['identity'];validate_chain(ROOT,spec['backbone'],spec['seed'],refs,I,docs['cache_admission'],docs['cache_audit'],docs['primary_launch'],docs['primary_execution'],docs['CPU_qualification'],terminal_sha=terminal_sha);pins={}
 def merge(d):
  for p,h in d.items():require(type(h) is str and re.fullmatch('[0-9a-f]{64}',h) and (p not in pins or pins[p]==h),'conflicting/invalid proof pin');pins[p]=h
 merge({r['path']:r['sha256'] for r in refs.values()});C,A,P,Q=(docs[k] for k in ('cache_admission','cache_audit','primary_execution','CPU_qualification'));merge(C['pinned_sha256']);merge(A['bracketed_readonly_evidence_sha256']);merge(P['pinned_sha256']);merge(I['inputs_sha256']);merge({str(base.ORIGINAL/n):h for n,h in I['code_sha256'].items()});merge({str(BASE_PATH):BASE_SHA,str(Path(__file__)):sha(__file__)})
 expected={h+'/'+r for h in ('B','AB') for r in base.ROLES};require(set(I['cache_completion_sha256'])==expected,'exact14 result roles');records={x['history']+'/'+x['role']:x for x in A['role_marker_checks']};require(set(records)==expected and len(A['role_marker_checks'])==14,'exact14 independently accepted headers')
 for key,pin in I['cache_completion_sha256'].items():
  history,role=key.split('/',1);run=I['run_ids'][history];path=ROOT/'features/H'/run/role/'complete.json';m=_json(path);v=m['identity'];r=records[key]
  require(r['complete_sha256']==pin and r['run_id']==run and m['status']=='CACHE_COMPLETE' and m['precision']=='FP32' and m['autocast'] is False and m['input_validation']=='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES','accepted FP32 role marker')
  require(v['run_id']==run and v['role']==role and v['history']==history and v['effective_precision']==I['effective_precision'] and v['checkpoint_path']==str(ROOT/'runs'/run/'final.pt') and v['original_head_sha256']==r['original_head_sha256'] and m['rows']==r['rows'] and v['accepted_table_videos']==r['accepted_videos'],'original LAST source/role/head/header binding')
  table=ROOT/'data/H'/(role+'.parquet');require(v['source_manifest_path']==r['source_manifest_path']==str(table) and v['source_manifest_sha256']==r['source_manifest_sha256'],'accepted canonical role manifest');merge({str(path):pin,str(table):v['source_manifest_sha256']})
 require(Q.get('original_source_manifest_sha256')==refs['source_manifest']['sha256'],'CPU qualification original source family')
 aux={}
 for key in ('candidate_report','independent_actual_audit','primary_entry_review'):
  merge({Q[key]:Q[key+'_sha256']});require(sha(_safe(Q[key]))==Q[key+'_sha256'],'actual qualification evidence SHA');aux[key]=_json(Q[key])
 validate_qualification_evidence(I['backbone'],Q,aux['candidate_report'],aux['independent_actual_audit'],aux['primary_entry_review'],precision=I['effective_precision']);merge(aux['candidate_report']['pinned_sha256']);merge(aux['candidate_report']['identity']['inputs_sha256'])
 validate_producer_terminal(ROOT,docs['primary_launch'],refs,I['effective_precision'])
 registry=_json(ROOT/'H_run_registry.json');effective=registry['effective_runs'][I['backbone']];require(effective['precision']==I['effective_precision'] and effective['run_ids']==[f"H_formal_{I['backbone']}_{h}_seed{s}" for s in SEEDS for h in ('B','AB')],'current registered effective all5 precision/model identities')
 for path,h in pins.items():require(sha(_safe(path))==h,'current matrix source/proof pin '+path)
 return I,pins

def audit_owned(spec,terminal_sha,slot):
 start=time.monotonic();I,pins=_metadata(spec,terminal_sha);pair=ROOT/f"analysis/H/{I['backbone']}/seed{I['seed']}";terminal=base.verify_terminal(pair,I,terminal_sha);table_ids={r:base.manifest_ids(r,I['run_ids']['B']) for r in base.ROLES};results=[]
 for aggregation in ('mean_logit','mean_probability'):
  cal={h:base.read_scores(pair/aggregation/f'source_cal_{h}.csv',aggregation,True) for h in ('B','AB')}
  for rows in cal.values():require(base.identities(rows)==table_ids['B/SOURCE_CAL'],'saved CAL exact admitted natural IDs')
  for domain in ('SOURCE_TEST','TARGET_TEST'):
   for cohort,names in (('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))):
    folder=pair/aggregation/domain/cohort;cells={c:base.read_scores(folder/f'{c}_scores.csv',aggregation) for c in names}
    for cell,rows in cells.items():
     version='AB' if cell in ('H01','H11') else 'B';role=f'B/{domain}' if cohort=='all_B' else f'common/{version}/{domain}';require(base.identities(rows)==table_ids[role],'saved natural score identities '+cell+'/'+role)
    results.append(dict(aggregation=aggregation,domain=domain,cohort=cohort,**base.check_tables(folder,cells,cal,base.METHODS if domain=='SOURCE_TEST' else ('DeepFakeDetection',),aggregation)))
 saved=_json(pair/'independent_saved_score_recompute.json');require(len(saved)==8,'eight producer guard cohorts');expected_na=[]
 for result in results:
  key=(result['aggregation'],result['domain'],result['cohort']);rows=[x for x in saved if (x['aggregation'],x['role'],x['cohort'])==key];require(len(rows)==1,'exact producer guard cohort');names=('H00','H10') if result['cohort']=='all_B' else ('H00','H01','H10','H11');checks=rows[0]['checks'];require(len(checks)==len(names)*2 and {(x['cell'],x['threshold_kind']) for x in checks}=={(c,k) for c in names for k in base.KINDS},'complete saved guard cells')
  for row in checks:
   status='PASS_OPERATING_ONLY_NUMERIC_AUC_NA' if row['cell'] in result['numeric_NA_cells'] else 'PASS';require(row['status']==status,'numeric NA declaration')
   if status!='PASS':expected_na.append(dict(role=result['domain'],cohort=result['cohort'],aggregation=result['aggregation'],**row))
 limitations=_json(pair/'limitations.json');require(limitations['numerical_NA']==expected_na and terminal['status']==('ANALYSIS_COMPLETE_WITH_NUMERIC_NA' if expected_na else 'ANALYSIS_COMPLETE'),'complete numerical NA/terminal agreement');require(_int(limitations['training_seed_count'],1) and _int(limitations['completed_model_pairs'],1) and limitations['five_seed_summary_pending'] is True,'single-pair result scope')
 for path,h in pins.items():require(sha(_safe(path))==h,'matrix source/proof changed during audit '+path)
 require(base.verify_terminal(pair,I,terminal_sha)==terminal,'original result terminal changed')
 return dict(schema='INDEPENDENT_ORIGINAL_H_LAST_PAIR_RESULT_AUDIT_V1',status='SAVED_RESULT_NUMERICS_AND_BOUND_PRIOR_ROLE_ADMISSION_PASS',backbone=I['backbone'],seed=I['seed'],source_checkpoint='original_LAST_final.pt',terminal_sha256=terminal_sha,identity_sha256=sha(pair/'identity.json'),audit_code_sha256=sha(__file__),independent_numeric_kernel_sha256=BASE_SHA,artifact_count=72,current_pins_sha256=pins,cohorts=results,cpu_slot=slot,elapsed_seconds=time.monotonic()-start,evidence_route='ACCEPTED_PARALLEL14_METADATA_REBOUND_ORIGINAL42FRESH_LEDGER_PRODUCER',fresh_pixel_or_source_context_validation=False,fresh_feature_dot_product_recompute=False,weight_or_model_loading=False,all_five_seed_conclusions_available=False,M_source_best_used=False,M_production_authorized=False,GPU_or_target_inference_started=False,snapshot_promoted=False,independent_scientific_scope='Saved natural-video scores, source-CAL thresholds, point metrics, signed effects and every applicable shared-group draw/NA/interval; immutable prior14-role admission reused',pending=['Fresh frame-to-video/feature-dot reconstruction is outside this saved-score audit','Fresh geometry/coverage reconstruction is outside this saved-score audit','Five-seed summary remains pending'])

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--binding-spec');p.add_argument('--binding-spec-sha');p.add_argument('--terminal-sha');p.add_argument('--slot',type=int,choices=(0,1));a=p.parse_args(argv)
 if not a.execute:return dict(execute=False,production_audit_started=False)
 if not all((a.binding_spec,a.binding_spec_sha,a.terminal_sha,a.slot is not None)):p.error('external binding/terminal SHA and shared slot required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k) in ('1','2') for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'bounded CPU-only audit environment');spec=load_spec(a.binding_spec,a.binding_spec_sha);alias='clip' if spec['backbone']=='clip_vit_b16' else spec['backbone'];out=HERE/f"H_LAST_{alias}_seed{spec['seed']}_independent_audit.json";require(not out.exists(),'preserve existing independent result')
 with base.cpu_slot(ROOT/'analysis/H/.cpu_slots',a.slot):
  result=audit_owned(spec,a.terminal_sha,a.slot);require(sha(a.binding_spec)==a.binding_spec_sha,'external binding drift');result.update(binding_spec_sha256=a.binding_spec_sha,lifetime_shared_cpu_slot_held=True)
  with out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
 return dict(status=result['status'],output=str(out),sha256=sha(out))
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
