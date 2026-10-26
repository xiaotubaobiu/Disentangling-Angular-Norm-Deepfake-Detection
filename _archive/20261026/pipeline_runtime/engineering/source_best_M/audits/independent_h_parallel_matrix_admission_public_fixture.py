"""Surrogate-root metadata tests only. No producer code execution or actual slots.
Root paths + fixture metadata trust-anchor digests are retargeted; validators and
hash comparisons remain intact; compiled harness/F/old helper code hashes stay exact.
"""
from pathlib import Path
import importlib.util,hashlib,json,copy,os,datetime,tempfile,traceback
import pyarrow as pa,pyarrow.parquet as pq
REAL=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');A=REAL/'engineering/source_best_M/audits';OUT=Path(os.environ['H_PARALLEL_MATRIX_REVIEW_TMP']) if 'H_PARALLEL_MATRIX_REVIEW_TMP' in os.environ else Path(tempfile.mkdtemp(prefix='parallel_admission_public_',dir='/tmp'))
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path(os.environ.get('H_PARALLEL_MATRIX_REVIEW_SOURCE',str(Path(__file__).parent/'h_parallel_matrix_admission_audit_reviewed_3642a04a.py')))
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n');return H(p)

def fixture(case,bone='dinov2_vit_b14',seed=7):
 root=Path(tempfile.mkdtemp(prefix=case+'_',dir=OUT));sp=importlib.util.spec_from_file_location('probe_'+case,SOURCE);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 realold=m.old;realref=json.loads(realold.REFERENCE.read_text());realmanifest=realold.MANIFEST;realregistry=realold.ROOT/'H_run_registry.json';realCODE=realold.CODE;realF=realold.F
 m.ROOT=root;m.HERE=root/'engineering/source_best_M/audits';m.HERE.mkdir(parents=True);m.OLD_PATH=m.HERE/'h_parallel_pair_admission_audit.py';m.OLD_PATH.write_bytes((A/'h_parallel_pair_admission_audit.py').read_bytes())
 shadow_source=root/'tools/h_parallel_matrix_admission_audit.py';shadow_source.parent.mkdir();shadow_source.write_bytes(SOURCE.read_bytes());m.__file__=str(shadow_source)
 old=m.old;old.ROOT=root;old.HERE=m.HERE;old.F=root/'qualification/H_parallel_fresh_validation';old.F.mkdir(parents=True);old.CODE=root/'original';(old.CODE/'reliability_v2').mkdir(parents=True)
 old.REFERENCE=old.F/'independent_actual_CPU_parallel_pair_admission_audit.json';old.MANIFEST=root/'qualification/H_primary_cache_continuation/qualified_source_manifest.json';old.MANIFEST.parent.mkdir(parents=True);old.MANIFEST.write_bytes(realmanifest.read_bytes());manifest=json.loads(old.MANIFEST.read_text())
 old.ALL3_ACCEPTED=root/'reports/H_final_all3_effective_registry_acceptance_20261025T0555.json';old.SHARED_PATH=m.HERE/'h_pair_result_audit_before_seed7_binding.py';old.SHARED_PATH.write_bytes((A/'h_pair_result_audit_before_seed7_binding.py').read_bytes())
 for name in manifest['source_sha256']:
  p=old.CODE/name;p.write_bytes((realCODE/name).read_bytes())
 for name in dict(old.ADMISSION_IMPL,**{'cpu_primary_analysis.py':m.PRIMARY_SHA}):(old.F/name).write_bytes((realF/name).read_bytes())
 regpath=root/'H_run_registry.json';regpath.write_bytes(realregistry.read_bytes());reg=json.loads(regpath.read_text());seals={};run_i={}
 for b,e in reg['effective_runs'].items():
  for run in e['run_ids']:
   rr=root/'runs'/run;ri=dict(config='1'*64,train_manifest='2'*64,dev_manifest='3'*64,initialization='4'*64,environment='5'*64,code=realref['source_training_code_sha256'],qualification='6'*64)
   arts={n:dump(rr/n,d) for n,d in [('identity.json',ri),('resolved_config.json',{'run':run}),('history.json',{'run':run,'history':'synthetic'}),('complete.json',{'run':run,'status':'complete'})]};arts['final.pt']='f'*64
   seal=dict(schema='H_COMPLETION_SEAL_V1',status='SEALED',run_id=run,target_used=False,identity=ri,h_code_bundle_sha256=realref['source_training_h_code_bundle_sha256'],artifact_sha256=arts);seals[run]=seal;run_i[run]=ri;dump(rr/'H_completion_seal.json',seal)
 for b in old.BACKBONES:
  dump(root/f'reports/H_{b}_AMP_paired_ledger.json',{'utc':'synthetic','backbone':b,'status':'PASS'})
  dump(root/f'qualification/H_target_export_{b}.json',{'backbone':b,'status':'PASS'})
 for rel,body in [('configs/H_execution.json',{'schema':'H_EXECUTION_V2'}),('data/H/freeze.json',{'schema':'toy'}),('qualification/H_gate.json',{'status':'PASS'}),('configs/H_frozen_recipes.json',{'schema':'toy'})]:dump(root/rel,body)
 req=root/'configs/requirements.execution.lock.txt';req.write_text('synthetic environment metadata\n')
 accepted_map={str(regpath):H(regpath)}
 for b,e in reg['effective_runs'].items():
  accepted_map.update({str(root/'runs'/run/'H_completion_seal.json'):H(root/'runs'/run/'H_completion_seal.json') for run in e['run_ids']})
  for rel in (f'reports/H_{b}_AMP_paired_ledger.json',f'qualification/H_target_export_{b}.json'):accepted_map[str(root/rel)]=H(root/rel)
 dump(old.ALL3_ACCEPTED,dict(current_canonical_proof_sha256=accepted_map));old.ALL3_SHA=H(old.ALL3_ACCEPTED)
 baseline={str(old.CODE/n):h for n,h in manifest['source_sha256'].items()};baseline.update({str(old.F/n):h for n,h in old.ADMISSION_IMPL.items()});baseline[str(old.MANIFEST)]=old.MANIFEST_SHA;baseline.update(accepted_map);baseline[str(old.ALL3_ACCEPTED)]=old.ALL3_SHA
 for p in old.extra_paths(reg):baseline.setdefault(str(p),H(p))
 assert len(baseline)==107
 ref=copy.deepcopy(realref);ref['pinned_sha256']=baseline;ref['protected_review_sha256']={}
 for p,h in realref['protected_review_sha256'].items():
  dest=old.F/Path(p).name;dest.write_bytes(Path(p).read_bytes());ref['protected_review_sha256'][str(dest)]=h
 iref=copy.deepcopy(ref['original_primary_pair_identity']);iref['inputs_sha256']={str(root/'configs/H_execution.json'):H(root/'configs/H_execution.json'),str(regpath):H(regpath),str(root/'qualification/H_target_export_clip_vit_b16.json'):H(root/'qualification/H_target_export_clip_vit_b16.json'),str(root/'reports/H_clip_vit_b16_AMP_paired_ledger.json'):H(root/'reports/H_clip_vit_b16_AMP_paired_ledger.json'),str(root/'data/H/freeze.json'):H(root/'data/H/freeze.json')};ref['original_primary_pair_identity']=iref
 if case=='protected_pin_conflict':ref['protected_review_sha256'][str(old.F/'cpu_primary_analysis.py')]='0'*64
 dump(old.REFERENCE,ref);old.REFERENCE_SHA=H(old.REFERENCE)
 expected=m.reconstruct_pins(bone,ref,manifest,reg,json.loads(old.ALL3_ACCEPTED.read_text()))
 if case=='run_metadata_drift':dump(root/'runs'/f'H_formal_{bone}_B_seed{seed}'/'history.json',{'changed_after_seal':True})
 if case=='seal_drift':
  rr=root/'runs'/f'H_formal_{bone}_B_seed{seed}'/'H_completion_seal.json';d=json.loads(rr.read_text());d['extra']='changed_after_accepted37';dump(rr,d)
 proto=m.expected_identity(bone,seed,expected,{},ref);heads={h:('a' if h=='B' else 'b')*64 for h in ('B','AB')}
 for role in old.ROLES:
  p=root/'data/H'/f'{role}.parquet';p.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table({k:[i//8 for i in range(80)] for k in old.IDENTITY_COLUMNS}),p)
 markers={}
 for history,run in proto['run_ids'].items():
  rr=root/'runs'/run;seal=seals[run];ri=run_i[run]
  for role in old.ROLES:
   table=root/'data/H'/f'{role}.parquet';v=dict(run_id=run,history=history,role=role,backbone=bone,artifact_root=str(root),config_path=str(root/'configs/H_execution.json'),config_sha256=expected[str(root/'configs/H_execution.json')],training_precision='AMP',effective_precision='AMP',epochs=30,dimension=m.DIM[bone],checkpoint_path=str(rr/'final.pt'),checkpoint_sha256=seal['artifact_sha256']['final.pt'],training_identity=ri,training_complete_sha256=expected[str(rr/'complete.json')],completion_seal_sha256=expected[str(rr/'H_completion_seal.json')],H_run_registry_sha256=old.REGISTRY_SHA,effective_run_ids=reg['effective_runs'][bone]['run_ids'],paired_ledger_path=str(root/f'reports/H_{bone}_AMP_paired_ledger.json'),paired_ledger_sha256=expected[str(root/f'reports/H_{bone}_AMP_paired_ledger.json')],target_export_gate_sha256=expected[str(root/f'qualification/H_target_export_{bone}.json')],H_data_freeze_sha256=expected[str(root/'data/H/freeze.json')],h_code_bundle_sha256=ref['source_training_h_code_bundle_sha256'],training_code_sha256=ref['source_training_code_sha256'],cache_code_sha256=expected[str(old.CODE/'reliability_v2/cache_h.py')],cache_helper_code_sha256=expected[str(old.CODE/'reliability_v2/cache_d.py')],original_head_sha256=heads[history],source_manifest_path=str(table),source_manifest_sha256=H(table),accepted_table_rows=80,accepted_table_videos=10)
   doc=dict(schema='pipeline-reliability-v2-H-fp32-cache-1',status='CACHE_COMPLETE',identity=v,precision='FP32',autocast=False,dimension=m.DIM[bone],rows=80,input_validation='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES',cached_manifest_sha256=H(table),chunks=[{}])
   markers[history+'/'+role]=dump(root/'features/H'/run/role/'complete.json',doc)
 proto=m.expected_identity(bone,seed,expected,markers,ref);ld=dict(backbone=bone,precision='AMP',fresh_source_completions=10,reference_sha256=expected[str(root/f'reports/H_{bone}_AMP_paired_ledger.json')],exact_original_ledger_except_top_utc=True,cross_invocation_receipt_reuse=False,elapsed_seconds=1.)
 C=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',backbone=bone,seed=seed,models=2,roles=14,identity=proto,pinned_sha256=expected,fresh_ledger_invocations=[copy.deepcopy(ld) for _ in range(14)],all_source_completions_fresh=140,cross_invocation_receipt_reuse=False,production_scoring_executed=False,GPU_launch_attempted=False,live_cache_owner_changed=False,elapsed_seconds=20.)
 _,slot,stem=m.HARNESS[(bone,seed)];names=dict(launch=root/'reports'/f'{stem}_launch.json',candidate=root/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json',stdout=root/'reports'/f'{stem}.log',time_v=root/'reports'/f'{stem}_time_v.log',process_witness=m.HERE/f'H_{bone}_seed{seed}_actual_producer_process_witness.json')
 realL=REAL/'reports'/f'{stem}_launch.json';L=json.loads(realL.read_text());now=datetime.datetime.now(datetime.timezone.utc);boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));ticks=int((now.timestamp()-boot)*os.sysconf('SC_CLK_TCK'));pid=99999101;childpid=99999102
 if case in ('live_true_birth','live_birth_plus1'):
  pid=os.getpid();fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();ticks=int(fields[19])
 L.update(pid=pid,utc=datetime.datetime.fromtimestamp(boot+ticks/os.sysconf('SC_CLK_TCK'),datetime.timezone.utc).isoformat(),stdout=str(names['stdout']),time_v=str(names['time_v']),candidate=str(names['candidate']))
 if bone=='clip_vit_b16':L['entry_template']=str(root/'engineering/source_best_M/primary_admit_clip_seed123_slot1.py')
 else:L['entry']=str(old.F/'cpu_ledger_admission.py')
 dump(names['launch'],L);dump(names['candidate'],C)
 lines=[f'BEGIN_FRESH_ORIGINAL_LEDGER {i} {bone} AMP models10_no_receipt_reuse' for i in range(1,15)]+['PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(names['candidate'])]
 if case=='log_complete_first':lines=[lines[-1],*lines[:-1]]
 names['stdout'].write_text('\n'.join(lines)+'\n');names['time_v'].write_text('Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.1\nMaximum resident set size (kbytes): 10000\nExit status: '+('1' if case=='time_exit1' else '0')+'\n')
 child=str(root/'environment/bin/python')+' -B -u -c '+L['parameterized_entry_code']+' ';recorded=ticks+1 if case=='live_birth_plus1' else ticks
 w=dict(schema='H_FINITE_PRODUCER_ACTUAL_PROCESS_WITNESS_V1',read_only=True,launch=str(names['launch']),launch_sha256=H(names['launch']),wrapper_pid=pid,utc=now.isoformat(),observed_process_tree=[dict(pid=pid,start_ticks=recorded,state='S',cmdline='/usr/bin/time -v -o '+L['time_v']+' '+child),dict(pid=childpid,start_ticks=recorded,state='R',cmdline=child)])
 dump(names['process_witness'],w)
 binding=m.HERE/f'H_{bone}_seed{seed}_matrix_admission_bindings.json';dump(binding,dict(schema='H_MATRIX_ADMISSION_EXTERNAL_BINDINGS_V1',root=str(root),backbone=bone,seed=seed,proofs={k:dict(path=str(p),sha256=H(p)) for k,p in names.items()}))
 # No Q is in the externally pinned five-proof binding. Produce a qualification
 # file afterwards; bad evidence/outside-namespace cases remain cryptographically bound.
 if bone=='dinov2_vit_b14' and case.startswith('qualified_'):
  bad=case=='qualified_bad_Q';aux=[]
  for key in ('independent_actual_audit','primary_entry_review'):
   p=(OUT/f'outside_{root.name}_{key}.json') if bad else root/f'reports/{key}.json';dump(p,dict(schema='synthetic_metadata',status='FAILED' if bad else 'PASS',backbone='clip_vit_b16' if bad else bone));aux.append(p)
  Q=dict(schema='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1',status='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE',allowed_backbones=[bone],independent_review_pass=True,allowed_use=['primary_pair_analysis'],original_role_checks=14,fresh_default_source_checks=140,implementation_sha256=dict(old.ADMISSION_IMPL,**{'cpu_primary_analysis.py':m.PRIMARY_SHA}),candidate_report=str(names['candidate']),candidate_report_sha256=H(names['candidate']),independent_actual_audit=str(aux[0]),independent_actual_audit_sha256=H(aux[0]),primary_entry_review=str(aux[1]),primary_entry_review_sha256=H(aux[1]))
  dump(root/f'reports/H_parallel_CPU_actual_qualification_{bone}.json',Q)
 elif bone=='clip_vit_b16':
  q=root/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json';q.write_bytes((REAL/'reports'/q.name).read_bytes())
 lock=root/'analysis/H/.cpu_slots';lock.mkdir(parents=True);[(lock/f'slot{i}.lock').touch() for i in (0,1)]
 mode='qualified_pair' if bone=='clip_vit_b16' or case.startswith('qualified_') else 'actual_qualification_prototype'
 return m,root,binding,mode,names

if __name__=='__main__':
 results=[]
 cases=[('prototype_positive','dinov2_vit_b14',7),('clip_positive','clip_vit_b16',3407),('live_true_birth','dinov2_vit_b14',7),('live_birth_plus1','dinov2_vit_b14',7),('qualified_bad_Q','dinov2_vit_b14',7),('protected_pin_conflict','dinov2_vit_b14',7),('run_metadata_drift','dinov2_vit_b14',7),('seal_drift','dinov2_vit_b14',7),('time_exit1','dinov2_vit_b14',7),('log_complete_first','dinov2_vit_b14',7)]
 for case,bone,seed in cases:
  try:
   m,root,b,mode,names=fixture(case,bone,seed);result=m.main(['--execute','--binding',str(b),'--binding-sha',H(b),'--mode',mode,'--slot','1' if bone=='dinov2_vit_b14' else '0']);d=json.loads(Path(result['output']).read_text());results.append(dict(case=case,status=result['status'],output=result['output'],mode=mode,manifest_footers=len(d['role_manifest_footer_metadata']),marker_count=len(d['role_marker_checks']),pin_count=len(d['canonical_shared107_pin_sha256']),Q_issued=result['CPU_qualification_issued'],own_live_review_PID_still_alive=Path(f'/proc/{os.getpid()}').exists() if case.startswith('live_') else None,real_torch_imported='torch' in __import__('sys').modules))
  except BaseException as e:results.append(dict(case=case,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
 print(json.dumps(results,indent=2));(OUT/'public_admission_results.json').write_text(json.dumps(results,indent=2)+'\n')
