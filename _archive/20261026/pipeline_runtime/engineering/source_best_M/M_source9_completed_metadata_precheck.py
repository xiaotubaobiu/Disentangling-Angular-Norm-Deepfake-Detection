"""Independent source9 completed METADATA-only precheck. No M package imports.
Only bounded JSON/code/log bytes are read. No tensors/PNG/NPY/Parquet payloads,
locks, CPU slots, process/GPU queries, workers, collector or gate issuer run.
"""
from pathlib import Path
import json,hashlib,datetime,os,re,math
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';W=E/'code/studies/pipeline_reliability/v2';MAX=1024**2
reads=[]
def raw(path,pin=None):
 p=Path(path).absolute();assert p.is_relative_to(R) and p.resolve().is_relative_to(R)
 for q in (p,*p.parents):
  assert not q.is_symlink()
  if q==R:break
 assert p.is_file() and p.stat().st_size<=MAX
 with p.open('rb') as f:b=f.read(MAX+1)
 assert len(b)<=MAX
 h=hashlib.sha256(b).hexdigest()
 if pin is not None:assert h==pin,(str(p),h,pin)
 ref=dict(path=str(p),sha256=h,bytes=len(b));reads.append(ref);return b,ref

def load(path,pin=None):
 b,ref=raw(path,pin)
 def pairs(items):
  result={}
  for k,v in items:assert k not in result;result[k]=v
  return result
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite metadata'))),ref

def typed(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def ishash(v):return type(v) is str and re.fullmatch('[0-9a-f]{64}',v) is not None

def main():
 prep,prep_ref=load(E/'M_actual_four_gate_review_preparation.json','016d1b05fda474af90cba388509a788ae2fe35fbb792164cf4bc1d7a082854f0')
 for p,pin in prep['fixed_code_sha256'].items():raw(p,pin)
 assert len(list((W/'reliability_v2').glob('*.py')))==33
 inp,inp_ref=load(prep['input_ref']['path'],prep['input_ref']['sha256']);source,source_ref=load(inp['source_file_admission']['path'],inp['source_file_admission']['sha256'])
 assert inp['target_inference_authorized'] is False and inp['source_input_scope']=='SELECTED_M_D6_H3_FIVE_SEEDS'
 owner_plan_path=E/'plans/source_gpu9_20261025T1700_candidate/owner_plan.json';owner_plan,owner_plan_ref=load(owner_plan_path,'195fcb122568d54d0022d05c24b9b3ec5c565dc50e1650eebc969723651bf0aa')
 qplan,qplan_ref=load(owner_plan['worker_plan']['path'],owner_plan['worker_plan']['sha256']);assert qplan['target_encoder_inference_authorized'] is False and qplan['input_admission_sha256']==inp_ref['sha256']
 expected=prep['fixed_inventory']['source9_run_ids'];assert owner_plan['run_ids']==qplan['run_ids']==expected and len(expected)==9
 term_path=R/'reports/M_finite_owner/source_qualification'/owner_plan_ref['sha256']/'883a623cc487472892444961790070b1/complete.json'
 term,term_ref=load(term_path,'1c19d78d4bb364578d69f3cc0f76af6df0608cdbb35277e9b063f618de7b50d7');status,status_ref=load(term_path.with_name('status.json'))
 assert term['schema']=='M_FINITE_OWNER_TERMINAL_V1' and term['status']=='ALL_REQUESTED_WORKERS_VERIFIED_OPERATIONALLY_ONLY' and term['mode']=='source_qualification'
 assert term['artifact_root']==str(R) and term['owner_plan_path']==str(owner_plan_path) and term['owner_plan_sha256']==owner_plan_ref['sha256'] and term['owner_code_sha256']==owner_plan['owner_code_sha256']==prep['fixed_code_sha256'][str(W/'tools/source_best_owner.py')]
 assert term['run_ids']==expected and set(term['completed'])==set(expected) and term['pending']==[] and term['failures']=={} and term['scientific_admission'] is False and term['M_production_qualified_here'] is False
 assert status['active']=={} and status['pending']==[] and status['failures']=={} and status['completed']==term['completed']
 assert term['input_admission_sha256']==owner_plan['input']['sha256']==inp_ref['sha256']
 assert re.fullmatch('[0-9a-f]{32}',term['phase_nonce']) and type(term['owner_pid']) is int and term['owner_pid']>0
 config,config_ref=load(R/'configs/H_execution.json');assert config['allowed_gpus']==[0,1,2,3] and len(config['allowed_gpu_uuids'])==4
 uuid_to_index=dict(zip(config['allowed_gpu_uuids'],config['allowed_gpus']))
 primary,primary_ref=load(R/'reports/H_primary_cache_stage_complete.json',term['primary_cache_stage_sha256']);queue,queue_ref=load(R/'H_queue_status.json')
 expected_H=sorted(k for k,v in source['records'].items() if v['route']=='H');assert len(expected_H)==30
 assert primary['status']=='PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION' and type(primary['role_count']) is int and primary['role_count']==210 and sorted(primary['run_ids'])==expected_H
 assert queue['active']==[] and queue['stage']=='H_PRIMARY_CACHE_EXPORT_REQUIRES_INDEPENDENT_ROLE_ADMISSION_COMPLETE' and queue['detail']['run_ids']==primary['run_ids']
 # These existing reports are reused as immutable metadata. No tensor/reference traversal.
 actual45,a45_ref=load(qplan['actual45_source_result']['path'],qplan['actual45_source_result']['sha256']);audit45,ia45_ref=load(qplan['independent_actual45_source_result']['path'],qplan['independent_actual45_source_result']['sha256'])
 a45={x['parent_run_id']:x for x in actual45['records']};ia45={x['run_id']:x for x in audit45['records']};assert len(a45)==len(ia45)==45 and set(a45)==set(ia45)==set(source['records'])
 observations=[];pids=[];UUIDs=set();cal_rows=0;vector_bytes=0
 for run in expected:
  ref=term['completed'][run];receipt,rref=load(ref['path'],ref['sha256']);route=run[0];role='source_cal' if route=='D' else 'B/SOURCE_CAL';canonical_receipt=R/'reports/M_source_only_GPU_qualification'/term['phase_nonce']/(run+'.json');assert Path(ref['path'])==canonical_receipt
  assert receipt['schema']=='M_ACTUAL_SOURCE_ONLY_GPU_QUALIFICATION_RETURN_V1' and receipt['parent_run_id']==run and receipt['input_admission_sha256']==inp_ref['sha256']
  assert receipt['code_sha256']==owner_plan['worker_code_sha256']==qplan['code_sha256'] and all(prep['fixed_code_sha256'][p]==pin for p,pin in receipt['code_sha256'].items())
  assert receipt['qualification_plan_sha256']==qplan_ref['sha256'] and receipt['phase_nonce']==term['phase_nonce'] and receipt['phase_owner_pid']==term['owner_pid'] and receipt['primary_cache_stage_sha256']==primary_ref['sha256']
  assert type(receipt['worker_pid']) is int and receipt['worker_pid']>0;pids.append(receipt['worker_pid'])
  assert receipt['approved_visible_GPU'] in uuid_to_index;UUIDs.add(receipt['approved_visible_GPU'])
  for name in ['GPU_executed','phase_native_mutual_exclusion_verified','repeated_forward_FP32_bitwise_equal']:assert receipt[name] is True
  for name in ['target_encoder_inference','threshold_or_test_metric_computed','full_M_cache_or_statistics_qualified','M_production_admitted']:assert receipt[name] is False
  assert type(receipt['optimizer_updates']) is int and receipt['optimizer_updates']==0 and type(receipt['rows']) is int and receipt['rows']>=416 and type(receipt['fresh_forward_rows']) is int and receipt['fresh_forward_rows']==receipt['rows'] and receipt['source_role']==role
  selected=a45[run];independent=ia45[run]
  assert selected['checkpoint_selection']=='source_best' and selected['original_head_dtype']==independent['original_head_dtype']=='float32' and selected['target_inference_authorized'] is False and independent['target_inference_authorized'] is False
  assert selected['source_best_bytes_tensors_head_verified'] is True and independent['finite_dense_CPU_tensors_verified'] is True
  assert receipt['original_head_sha256']==selected['head_sha256']==independent['head_sha256'] and receipt['checkpoint_sha256']==selected['checkpoint_sha256']==independent['checkpoint_sha256']
  assert type(receipt['source_best_epoch']) is int and receipt['source_best_epoch']==selected['best']['epoch']==independent['independent_best']['epoch']
  folder=R/'features/source_best_M'/inp_ref['sha256']/run/role;meta,mref=load(folder/'complete.json',receipt['source_only_cache_complete_sha256']);ident=meta['identity'];identity,identity_ref=load(folder/'identity.json');assert typed(identity)==typed(ident)
  dimension=selected['dimension'];rows=receipt['rows'];assert dimension in (512,768,1280) and dimension==independent['dimension']
  assert meta['schema']=='M_SOURCE_BEST_RAW_FP32_CACHE_V1' and meta['status']=='CACHE_COMPLETE' and meta['precision']=='FP32' and meta['autocast'] is False and meta['rows']==rows and meta['dimension']==dimension and meta['chunk_rows']==4096
  assert ident['checkpoint_selection']=='source_best' and ident['checkpoint_path']==str(R/'runs'/run/'source_best.pt') and ident['checkpoint_sha256']==receipt['checkpoint_sha256'] and ident['original_head_sha256']==receipt['original_head_sha256']
  assert ident['parent_run_id']==ident['run_id']==run and ident['role']==role and ident['route']==route and ident['input_admission_sha256']==inp_ref['sha256'] and ident['artifact_root']==str(R) and ident['target_inference_authorized'] is False
  assert ident['cache_code_sha256']==inp['cache_code_sha256']==prep['fixed_code_sha256'][str(W/'reliability_v2/source_best_cache.py')] and ident['source_best_epoch']==receipt['source_best_epoch']
  assert ident['source_manifest_path']==str(R/f'data/{route}'/(role+'.parquet')) and ishash(ident['source_manifest_sha256']) and ident['source_manifest_sha256']==meta['cached_manifest_sha256'] and ident['accepted_table_rows']==rows
  assert set(meta['head'])=={'weight_file','bias_file','weight_sha256','bias_sha256'} and meta['head']['weight_file']=='head_weight.npy' and meta['head']['bias_file']=='head_bias.npy' and ishash(meta['head']['weight_sha256']) and ishash(meta['head']['bias_sha256'])
  assert len(meta['chunks'])==math.ceil(rows/4096);position=0
  for i,item in enumerate(meta['chunks']):
   count=min(4096,rows-position);assert item['file']==f'chunk_{i:06d}.npy' and item['start']==position and item['rows']==count and item['dimension']==dimension and item['dtype']=='float32' and ishash(item['sha256'])
   chunk,cjref=load(folder/f'chunk_{i:06d}.json');assert typed(chunk)==typed(item);position+=count
  assert position==rows and receipt['FP32_feature_bytes']==rows*dimension*4
  cal_rows+=rows;vector_bytes+=receipt['FP32_feature_bytes']
  observations.append(dict(run_id=run,route=route,history=selected['history'],seed=selected['seed'],source_role=role,receipt=rref,complete_marker=mref,identity_metadata=identity_ref,source_best_epoch=receipt['source_best_epoch'],checkpoint_sha256=receipt['checkpoint_sha256'],original_FP32_head_sha256=receipt['original_head_sha256'],declared_head_NPY_SHA256=meta['head'],rows=rows,fresh_forward_rows=receipt['fresh_forward_rows'],dimension=dimension,declared_FP32_vector_bytes=receipt['FP32_feature_bytes'],approved_physical_GPU_index=uuid_to_index[receipt['approved_visible_GPU']],approved_GPU_UUID=receipt['approved_visible_GPU'],shared_phase_nonce=receipt['phase_nonce'],metadata_bound_to_existing_independent45=True,head_file_bytes_or_payload_freshly_read=False))
 assert len(set(pids))==9 and UUIDs==set(uuid_to_index) and cal_rows==10080 and vector_bytes==34406400
 # Wrapper stdout and time report are tiny exit evidence, not process queries.
 stdout,stdout_ref=raw(E/'actual_source9_GPU_qualification_20261025T1925.log');result=json.loads(stdout.decode().strip().splitlines()[-1]);assert result['terminal']==str(term_path) and result['terminal_sha256']==term_ref['sha256'] and result['status']==term['status']
 timing,timing_ref=raw(E/'actual_source9_GPU_qualification_20261025T1925_time_v.log');text=timing.decode();assert re.search(r'Exit status:\s*0\s*$',text) and '1:40:14' in text and owner_plan_ref['sha256'] in text
 opobs,opobs_ref=load(E/'actual_source9_GPU_operational_closure_20261025T2110.json','9743b770def794c40df7a7a586de6fcae1c466f75bdfe1be5bcd217f2690c53a')
 # Read original phase binding metadata only, never the referenced payloads or locks.
 collector=Path(prep['future_completion_paths']['source9_collector_bridge']);collector_presence=collector.exists()
 report=dict(schema='M_SOURCE9_COMPLETED_METADATA_INDEPENDENT_PRECHECK_V1',status='METADATA_CONSISTENT_SOURCE9_OPERATIONAL_CLOSURE_ONLY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_root=str(R),preparation_reused=prep_ref,terminal=term_ref,owner_plan=owner_plan_ref,qualification_plan=qplan_ref,input_admission=inp_ref,source_file_admission=source_ref,existing_actual45_report=a45_ref,existing_independent45_audit=ia45_ref,original_primary_stage=primary_ref,original_queue_metadata=queue_ref,configuration_metadata=config_ref,final_progress=status_ref,wrapper_stdout=stdout_ref,wrapper_GNU_time=timing_ref,root_operational_observation=opobs_ref,checks=dict(exact_D3_H6_seed7=True,all9_receipt_SHA_and_typed_source_only_flags=True,all9_complete_marker_SHA=True,all9_identity_and_chunk_JSON_metadata=True,all9_original_FP32_head_checkpoint_epoch_metadata_match_existing_independent45=True,all9_code_and_same_input_bindings=True,original_H30_210_operational_stage_and_terminal_queue_metadata=True,worker_identity_fields_unique_and_owner_nonce_consistent=True,four_approved_device_UUID_metadata_map_to_indices0_3=True,no_protected4_7_device_in_receipts=True,wrapper_printed_same_terminal_SHA_and_exit0=True,active_package33_and_fixed_tools_unchanged=True),four_device_phase_scope=dict(shared_owner_phase_nonce=term['phase_nonce'],interpretation='One qualified owner phase/nonce spans all9 cases on4 allowed physicalGPU indices; do not invent four distinct nonces.',allowed_UUID_to_index=uuid_to_index,actual_GPU_or_lock_state_queried=False,release_evidence='Qualified owner code closes native handles and phase lease; wrapper exited0. Current GPU occupancy/process disappearance not independently queried by this metadata precheck; root observation is SHA-bound separately.'),rows=cal_rows,declared_FP32_vector_bytes=vector_bytes,results=observations,source_selection_and_data_permission=dict(checkpoint_selection='source_best',actual_selected_paths='runs/<registered endpoint>/source_best.pt',original_heads_reused_from_existing45_audit=True,roles='D/source_cal and H/B/SOURCE_CAL only',sourceCAL_manifest_path_and_SHA_linkage='Checked in completion metadata; original Parquet bytes/pixels not reopened.',target_encoder_inference=False,threshold_or_test_metric_computed=False,optimizer_updates=0,full_cache_or_statistics_permission=False,unequal_H_best_epochs_do_not_imply_matched_updates=True,no_training_or_F_expansion=True),collector=dict(path=str(collector),present_at_precheck=collector_presence,independently_accepted_by_this_precheck=False,executed_here=False),all_small_metadata_code_log_reads=reads,missing_before_four_actual_gates=['Accepted collector06eda must independently read actual9 SOURCE_CAL NPY headers/original-head bytes/manifests/chunk payload checksums and final snapshots under a genuinely available original sharedCPU slot. This audit did not open those payloads.','External SHA-bound actual source9 collector bridge and independent final gate review; existing terminal is operational-only and neither root observation nor this metadata precheck is that acceptance.','Separate actual_D_storage and actual_H_storage wrappers must bind D3/H6 currentinput/source9 qualification to reusable existing D/H exact transport plus their independent audits without relabeling LAST evidence into newM scientific results.','Fresh sameinput complete future-write/staging/livefree acceptance for165195057321B plus1MiB publication margin, keeping50GiB floor and6GiBM temp. Candidate arithmetic/preparation are not capacity acceptance.','Only later, applicable D6/H2x3 full cache admissions and bounded real publicD/H canaries can precede full30 statistics/scientific/production acceptance.'],declined_to_judge=['No fresh tensor/head-array/NPYheader/Parquet/chunk/bootstrap payload bytes, PNG or45 checkpoint validation.','No reopened native GPU lease, CPU slot/ledger access, PID/GPU query or process launch/termination/wait.','No cache collector, gate issuer, statistics, canary or scientific job executed.','No realcapacity qualification, budget credit, matched-update inference or wholeM admission.'],verdict='METADATA_PRECHECK_ONLY_PENDING_ACTUAL_COLLECTOR_AND_FOUR_GATE_REVIEW',findings=dict(Critical=[],Important=[],Minor=[]),independent_source9_payload_acceptance=False,four_actual_gates_written=False,gate_or_permit_issued=False,CPU_slot_taken=False,heavy_payload_validation_performed=False,implementation_changed=False,M_production_cache_admission=False,production_admitted=False,whole_M_admitted=False,real_budget_credit_bytes=0)
 target=E/'M_source9_completed_metadata_independent_precheck.json';assert not target.exists()
 with target.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 assert load(target)[0]['production_admitted'] is False
 for p,pin in prep['fixed_code_sha256'].items():raw(p,pin)
 print(json.dumps(dict(report=str(target),report_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),terminal_sha256=term_ref['sha256'],receipts=9,SOURCE_CAL_rows=cal_rows,source_only_vector_bytes_declared=vector_bytes,shared_phase_nonce=term['phase_nonce'],approved_device_indices=sorted(uuid_to_index.values()),collector_present=collector_presence,production_admitted=False),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
