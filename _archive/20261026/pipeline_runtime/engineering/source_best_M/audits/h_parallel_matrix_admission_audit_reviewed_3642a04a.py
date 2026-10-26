"""Finite independent H admission metadata adapter; original validators remain unchanged.
Only the two actually launched, externally pinned harnesses are executable here.
No original F imports, model/image/chunk reads, scoring or dispatcher exists.
"""
import argparse,copy,datetime,hashlib,importlib.util,json,math,os,re,time
from pathlib import Path
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
HERE=ROOT/'engineering/source_best_M/audits'
OLD_PATH=HERE/'h_parallel_pair_admission_audit.py'
OLD_SHA='318a248fc2baf3aa3e3e0e697dd9d04ae4190eaeb7f6061eae79f9a34123bb75'
if OLD_PATH.is_symlink() or hashlib.sha256(OLD_PATH.read_bytes()).hexdigest()!=OLD_SHA:raise AssertionError('accepted metadata primitives drift')
spec=importlib.util.spec_from_file_location('_preserved_admission_318a',OLD_PATH);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
require=old.require;sha=old.sha
DIM={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}
HARNESS={('clip_vit_b16',3407):('b6aca21f368587e26430c5ef3f7bf9a13d170083f19b58ddff3874f13bda8131',1,'H_primary_admit_clip_seed3407_slot1_20261025T2000'),('dinov2_vit_b14',7):('2e05c7339e89c259a368d0a5f8592404f446704093ab0134347d6aa32b4b4545',0,'H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030')}
PRIMARY_SHA='c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce'
def read(p):return json.loads(Path(p).read_text())
def typed(v,n):return type(v) is int and v==n

def reconstruct_pins(bone,reference,manifest,registry,accepted):
 require(bone in DIM,'registered backbone')
 # Reuse the reviewed independent reconstruction of every shared pin. Replace
 # only the 40 non-seal per-run metadata paths, using the accepted37 seals.
 baseline=old.reconstruct_pins(reference,manifest,registry,accepted)
 pins={p:h for p,h in baseline.items() if not (Path(p).parent.parent==ROOT/'runs' and Path(p).parent.name.startswith('H_formal_clip_vit_b16_') and Path(p).name!='H_completion_seal.json')}
 for run in registry['effective_runs'][bone]['run_ids']:
  r=ROOT/'runs'/run;seal=read(r/'H_completion_seal.json')
  require(seal['schema']=='H_COMPLETION_SEAL_V1' and seal['status']=='SEALED' and seal['run_id']==run and seal['target_used'] is False,'accepted sealed run metadata')
  for name in ('identity.json','resolved_config.json','history.json','complete.json'):
   h=seal['artifact_sha256'][name];require(old.hash_string(h),'seal-bound metadata hash');pins[str(r/name)]=h
 require(len(pins)==107 and all(old.hash_string(h) for h in pins.values()),'independent exact107 inventory')
 return pins

def expected_identity(bone,seed,pins,markers,reference):
 require(bone in DIM and type(seed) is int and seed in old.SEEDS,'registered identity')
 i=copy.deepcopy(reference['original_primary_pair_identity']);i.update(backbone=bone,seed=seed,run_ids={h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')},cache_completion_sha256=markers)
 names=[ROOT/'configs/H_execution.json',ROOT/'H_run_registry.json',ROOT/f'qualification/H_target_export_{bone}.json',ROOT/f'reports/H_{bone}_AMP_paired_ledger.json',ROOT/'data/H/freeze.json']
 i['inputs_sha256']={str(p):pins[str(p)] for p in names}
 require(all(pins[str(old.CODE/'reliability_v2'/name)]==h for name,h in i['code_sha256'].items()),'qualified original analysis code identity')
 return i

def validate_candidate(c,bone,seed,reference,pins):
 require(c.get('status')=='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS' and c.get('backbone')==bone and typed(c.get('seed'),seed) and typed(c.get('models'),2) and typed(c.get('roles'),14),'actual completed same-backbone pair candidate')
 require(c.get('pinned_sha256')==pins and all(c.get(k) is False for k in ('cross_invocation_receipt_reuse','production_scoring_executed','GPU_launch_attempted','live_cache_owner_changed')),'exact107 and original boundaries')
 rows=c.get('fresh_ledger_invocations');require(typed(c.get('all_source_completions_fresh'),140) and type(rows) is list and len(rows)==14,'actual14 x typed10 source checks')
 ledger=pins[str(ROOT/f'reports/H_{bone}_AMP_paired_ledger.json')]
 for row in rows:
  require(type(row) is dict and typed(row.get('fresh_source_completions'),10) and row.get('backbone')==bone and row.get('precision')=='AMP' and row.get('reference_sha256')==ledger and row.get('exact_original_ledger_except_top_utc') is True and row.get('cross_invocation_receipt_reuse') is False,'each original fresh10 exact ledger')
  require(type(row.get('elapsed_seconds')) in (int,float) and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>=0,'finite ledger timing')
 require(type(c.get('elapsed_seconds')) in (int,float) and math.isfinite(c['elapsed_seconds']) and c['elapsed_seconds']>0,'finite actual candidate timing')
 i=c['identity'];require(i==expected_identity(bone,seed,pins,i['cache_completion_sha256'],reference),'independently reconstructed original full identity')
 markers=i['cache_completion_sha256'];require(type(markers) is dict and set(markers)=={h+'/'+r for h in ('B','AB') for r in old.ROLES} and all(old.hash_string(h) for h in markers.values()),'exact14 marker pins')
 return i

def validate_launch(L,bone,seed):
 require((bone,seed) in HARNESS and type(seed) is int,'only actual reviewed harness scope')
 h,slot,stem=HARNESS[(bone,seed)];code=L['parameterized_entry_code']
 require(L['backbone']==bone and typed(L['seed'],seed) and typed(L['CPU_slot'],slot) and type(L['pid']) is int and L['pid']>0,'bound producer identity/slot/PID')
 require(L['parameterized_code_sha256']==h==hashlib.sha256(code.encode()).hexdigest() and L['new_dispatcher'] is False and L['repeat_on_failure'] is False and L['production_or_F_files_changed'] is False,'exact finite reviewed launch bytes')
 require(L['stdout']==str(ROOT/'reports'/f'{stem}.log') and L['time_v']==str(ROOT/'reports'/f'{stem}_time_v.log') and L['candidate']==str(ROOT/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json'),'canonical finite witness/candidate paths')
 if bone=='clip_vit_b16':require(L['entry_template']==str(ROOT/'engineering/source_best_M/primary_admit_clip_seed123_slot1.py') and L['entry_template_sha256']=='cd33c85b28dcf82ea0307c0786603a49cf67014f1406ef0d92668a28a6151fa7' and L['qualification_sha256']=='0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7','qualified CLIP harness reference')
 else:require(L['entry']==str(old.F/'cpu_ledger_admission.py') and L['entry_sha256']==old.ADMISSION_IMPL['cpu_ledger_admission.py'] and L['CLIP_qualification_reused_for_DINO'] is False and L['primary_scientific_scoring_started'] is False,'independent DINO functional prototype only')

def validate_exit(L,witness=None):
 # GNU time waits for the direct child; the recorded process tree also proves
 # that both exact birth identities are now gone, including PID-reuse handling.
 nodes=(witness or {}).get('observed_process_tree',[dict(pid=L['pid'],start_ticks=None)])
 for node in nodes:
  p=Path('/proc')/str(node['pid'])/'stat'
  if p.exists():
   fields=p.read_text().rsplit(')',1)[1].split()
   require(fields[0]=='Z' or (node['start_ticks'] is not None and int(fields[19])!=node['start_ticks']),'original producer still active')
 return [x['pid'] for x in nodes]

def validate_witness(w,L,launch_path,launch_sha):
 require(w.get('schema')=='H_FINITE_PRODUCER_ACTUAL_PROCESS_WITNESS_V1' and w.get('read_only') is True and w.get('launch')==str(launch_path) and w.get('launch_sha256')==launch_sha and w.get('wrapper_pid')==L['pid'],'external actual process witness')
 ns=w['observed_process_tree'];require(type(ns) is list and len(ns)==2 and ns[0]['pid']==L['pid'] and len({x['pid'] for x in ns})==2 and all(type(x['pid']) is int and x['pid']>0 and type(x['start_ticks']) is int and x['start_ticks']>0 for x in ns),'actual GNUtime wrapper/direct Python child identities')
 child=str(ROOT/'environment/bin/python')+' -B -u -c '+L['parameterized_entry_code']+' '
 require(ns[1]['cmdline']==child and ns[0]['cmdline']=='/usr/bin/time -v -o '+L['time_v']+' '+child,'actual exact wrapper/child executable and code witness')
 boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));launched=datetime.datetime.fromisoformat(L['utc'].replace('Z','+00:00')).timestamp();observed=datetime.datetime.fromisoformat(w['utc']).timestamp()
 require(observed>=launched and all(abs(boot+n['start_ticks']/os.sysconf('SC_CLK_TCK')-launched)<=5 for n in ns),'actual producer birth identities match finite launch time')
 return validate_exit(L,w)

def validate_logs(log,timing,bone,seed):
 lines=[x for x in log.splitlines() if x.strip()];pattern=rf'BEGIN_FRESH_ORIGINAL_LEDGER (\d+) {re.escape(bone)} AMP models10_no_receipt_reuse'
 starts=[int(m[1]) for x in lines if (m:=re.fullmatch(pattern,x))]
 require(starts==list(range(1,15)) and lines[-1]=='PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(ROOT/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json'),'actual ordered14 starts and final publication')
 require(sum(x.startswith('PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT ') for x in lines)==1,'unique final candidate publication')
 require(re.findall(r'^\s*Exit status:\s*(\d+)\s*$',timing,re.M)==['0'],'true unique GNUtime exit0')
 wall=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)',timing);rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',timing);require(wall is not None and rss is not None,'actual resource witness')
 return dict(exit_status=0,wall_clock=wall[1],wall_seconds=sum(float(v)*60**n for n,v in enumerate(reversed(wall[1].split(':')))),max_RSS_KiB=int(rss[1]))

def validate_marker(m,bone,history,role,i,pins,reference,seal,ri):
 run=i['run_ids'][history];r=ROOT/'runs'/run;v=m['identity']
 require(m.get('schema')=='pipeline-reliability-v2-H-fp32-cache-1' and m.get('status')=='CACHE_COMPLETE' and m.get('precision')=='FP32' and m.get('autocast') is False and m.get('input_validation')=='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES','fresh original FP32 role')
 require(v['run_id']==run and v['history']==history and v['role']==role and v['backbone']==bone and v['artifact_root']==str(ROOT) and v['config_path']==str(ROOT/'configs/H_execution.json') and v['config_sha256']==pins[v['config_path']] and v['training_precision']==v['effective_precision']=='AMP' and typed(v['epochs'],30) and typed(v['dimension'],DIM[bone]) and typed(m['dimension'],DIM[bone]),'same registered backbone/history/role/config')
 require(v['checkpoint_path']==str(r/'final.pt') and v['checkpoint_sha256']==seal['artifact_sha256']['final.pt'] and seal['identity']==ri==v['training_identity'] and seal['run_id']==run and seal['status']=='SEALED' and seal['target_used'] is False,'accepted original LAST source-seal identity')
 require(v['training_complete_sha256']==pins[str(r/'complete.json')] and v['completion_seal_sha256']==pins[str(r/'H_completion_seal.json')] and v['H_run_registry_sha256']==old.REGISTRY_SHA and v['effective_run_ids']==[f'H_formal_{bone}_{h}_seed{s}' for s in old.SEEDS for h in ('B','AB')],'exact all-five effective registry and completion pins')
 require(v['paired_ledger_path']==str(ROOT/f'reports/H_{bone}_AMP_paired_ledger.json') and v['paired_ledger_sha256']==pins[v['paired_ledger_path']] and v['target_export_gate_sha256']==pins[str(ROOT/f'qualification/H_target_export_{bone}.json')] and v['H_data_freeze_sha256']==pins[str(ROOT/'data/H/freeze.json')],'accepted gate/ledger/freeze')
 require(v['h_code_bundle_sha256']==seal['h_code_bundle_sha256']==reference['source_training_h_code_bundle_sha256'] and v['training_code_sha256']==ri['code']==reference['source_training_code_sha256'] and v['cache_code_sha256']==pins[str(old.CODE/'reliability_v2/cache_h.py')] and v['cache_helper_code_sha256']==pins[str(old.CODE/'reliability_v2/cache_d.py')],'unchanged qualified science/cache source')
 require(type(m['rows']) is int and m['rows']==v['accepted_table_rows'] and m['rows']>0 and type(v['accepted_table_videos']) is int and 0<v['accepted_table_videos']<=m['rows'] and old.hash_string(v['original_head_sha256']),'role accepted counts/head metadata')
 p=ROOT/'data/H'/f'{role}.parquet';require(v['source_manifest_path']==str(p) and old.hash_string(v['source_manifest_sha256']) and m['cached_manifest_sha256']==v['source_manifest_sha256'],'canonical input/cached manifest identity')
 return dict(history=history,run_id=run,role=role,rows=m['rows'],accepted_videos=v['accepted_table_videos'],chunk_count=len(m['chunks']),original_head_sha256=v['original_head_sha256'],source_manifest_path=str(p),source_manifest_sha256=v['source_manifest_sha256'],input_validation=m['input_validation'])

def audit_owned(binding,mode,slot):
 b=read(binding['path']);require(sha(binding['path'])==binding['sha256'] and b['schema']=='H_MATRIX_ADMISSION_EXTERNAL_BINDINGS_V1' and b['root']==str(ROOT),'external canonical binding SHA')
 bone,seed=b['backbone'],b['seed'];require((bone,seed) in HARNESS and mode in ('qualified_pair','actual_qualification_prototype'),'finite actual scope')
 require(mode!='actual_qualification_prototype' or bone=='dinov2_vit_b14','actual independent prototype scope')
 refs=b['proofs'];require(set(refs)=={'launch','candidate','stdout','time_v','process_witness'},'exact external proof inventory')
 _,_,stem=HARNESS[(bone,seed)];names=dict(launch=ROOT/'reports'/f'{stem}_launch.json',candidate=ROOT/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json',stdout=ROOT/'reports'/f'{stem}.log',time_v=ROOT/'reports'/f'{stem}_time_v.log',process_witness=HERE/f'H_{bone}_seed{seed}_actual_producer_process_witness.json')
 pins={str(OLD_PATH):OLD_SHA,str(old.SHARED_PATH):old.SHARED_SHA,str(old.REFERENCE):old.REFERENCE_SHA,str(old.MANIFEST):old.MANIFEST_SHA,str(old.ALL3_ACCEPTED):old.ALL3_SHA,str(Path(__file__)):sha(__file__),binding['path']:binding['sha256'],str(old.F/'cpu_primary_analysis.py'):PRIMARY_SHA}
 for key,r in refs.items():require(set(r)=={'path','sha256'} and r['path']==str(names[key]) and old.hash_string(r['sha256']),'canonical finite proof '+key);pins[r['path']]=r['sha256']
 def check():
  for p,h in pins.items():
   path=Path(p);require(path.resolve()==path and not path.is_symlink() and sha(path)==h,'current read-only metadata pin '+p)
 check();reference=read(old.REFERENCE);expected=reconstruct_pins(bone,reference,read(old.MANIFEST),read(ROOT/'H_run_registry.json'),read(old.ALL3_ACCEPTED));
 for source in (expected,reference['protected_review_sha256']):
  for path,h in source.items():require(path not in pins or pins[path]==h,'conflicting immutable audit evidence');pins[path]=h
 check()
 L=read(names['launch']);validate_launch(L,bone,seed);PIDs=validate_witness(read(names['process_witness']),L,names['launch'],refs['launch']['sha256']);C=read(names['candidate']);i=validate_candidate(C,bone,seed,reference,expected);resources=validate_logs(names['stdout'].read_text(),names['time_v'].read_text(),bone,seed)
 if mode=='qualified_pair':
  q=ROOT/'reports'/('H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json' if bone=='clip_vit_b16' else f'H_parallel_CPU_actual_qualification_{bone}.json');Q=read(q)
  require(Q.get('schema')=='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1' and Q.get('status')=='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE' and Q.get('allowed_backbones')==[bone] and Q.get('independent_review_pass') is True and 'primary_pair_analysis' in Q.get('allowed_use',[]) and typed(Q.get('original_role_checks'),14) and typed(Q.get('fresh_default_source_checks'),140),'actual same-backbone qualified use')
  require(Q.get('implementation_sha256')==dict(old.ADMISSION_IMPL,**{'cpu_primary_analysis.py':PRIMARY_SHA}),'qualified implementation family')
  if bone=='clip_vit_b16':require(sha(q)=='0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7','accepted actual CLIP qualification SHA')
  else:
   require(Q.get('candidate_report')==str(names['candidate']) and Q.get('candidate_report_sha256')==refs['candidate']['sha256'],'same actual independent DINO prototype')
   for key in ('candidate_report','independent_actual_audit','primary_entry_review'):
    require(old.hash_string(Q.get(key+'_sha256')) and sha(Q[key])==Q[key+'_sha256'],'actual qualification auxiliary current proof');pins[Q[key]]=Q[key+'_sha256']
  pins[str(q)]=sha(q)
 records=[];footers={};heads={};tables={}
 for history,run in i['run_ids'].items():
  r=ROOT/'runs'/run;seal=read(r/'H_completion_seal.json');ri=read(r/'identity.json')
  for role in old.ROLES:
   p=ROOT/'features/H'/run/role/'complete.json';h=i['cache_completion_sha256'][history+'/'+role];require(sha(p)==h,'actual14 marker pin');pins[str(p)]=h;record=validate_marker(read(p),bone,history,role,i,expected,reference,seal,ri)
   require(history not in heads or heads[history]==record['original_head_sha256'],'same FP32 head across7roles');heads[history]=record['original_head_sha256'];table=(record['source_manifest_sha256'],record['rows'],record['accepted_videos']);require(role not in tables or tables[role]==table,'same role cohort acrosshistories');tables[role]=table
   source=record['source_manifest_path']
   if source not in footers:footers[source]=old.manifest_footer(source)
   old.validate_footer(footers[source],record['rows']);record['complete_sha256']=h;records.append(record)
 check();validate_exit(L,read(names['process_witness']))
 for p,v in footers.items():require(old.manifest_footer(p)==v,'manifest footer changed during audit')
 return dict(schema='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1',status='PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND',backbone=bone,seed=seed,mode=mode,reference_audit_sha256=old.REFERENCE_SHA,canonical_shared107_pin_sha256=expected,reference_pin_inventory_independently_reconstructed=True,qualified_original_code_count=20,canonical_proof_count=37,models=2,roles=14,fresh_ledger_invocations=14,source_checks_per_ledger=10,source_checks_in_ledger_receipts=140,producer_resources=resources,producer_PIDs_exited=PIDs,candidate_identity=i,role_marker_checks=records,role_manifest_footer_metadata=list(footers.values()),bracketed_readonly_evidence_sha256=pins,audit_CPU_slot=slot,lifetime_CPU_flock_held=True,source_validation_freshness_established_by='SHA-bound original14role/140fresh source validation, exact ledger summaries, launch bytes, ordered final log and observed wrapper/child exit0; no repeated payload checks',fresh_model_image_or_chunk_validation_repeated=False,numeric_ledger_regenerated=False,raw_role_manifest_bytes_rehashed=False,primary_statistics_started=False,scientific_metric_or_bootstrap_results_accepted=False,original_or_F_changed=False,production_GPU_allowed=False,M_authorized=False,primary_resource_or_speedup_qualified=False,original_F_primary_preflight_checks_must_be_preserved=True,actual_CPU_qualification_issued=False)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--binding');p.add_argument('--binding-sha');p.add_argument('--mode',choices=('qualified_pair','actual_qualification_prototype'));p.add_argument('--slot',type=int,choices=(0,1));a=p.parse_args(argv)
 if not a.execute:return dict(production_admission_started=False,preparation_only=True)
 if a.slot is None or not a.mode or not a.binding or not old.hash_string(a.binding_sha):p.error('external binding/sha/mode/free shared CPU slot required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only1thread metadata audit')
 bp=Path(a.binding);require(bp.parent==HERE and bp.name.startswith('H_') and bp.name.endswith('_matrix_admission_bindings.json'),'canonical finite external binding')
 b=read(bp);label={'clip_vit_b16':'CLIP','dinov2_vit_b14':'DINO'}[b['backbone']];out=HERE/(f'H_{label}_seed{b["seed"]}_actual_CPU_qualification_metadata_audit.json' if a.mode=='actual_qualification_prototype' else f'H_{label}_seed{b["seed"]}_parallel_cache_admission_independent_audit.json');require(not out.exists() and not out.is_symlink(),'preserve independent outputs')
 with old.SHARED.cpu_slot(ROOT/'analysis/H/.cpu_slots',a.slot):
  started=time.monotonic();d=audit_owned(dict(path=str(bp),sha256=a.binding_sha),a.mode,a.slot);d['audit_elapsed_seconds']=time.monotonic()-started
  with out.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return dict(output=str(out),sha256=sha(out),status=d['status'],mode=a.mode,CPU_qualification_issued=False)
if __name__=='__main__':print(json.dumps(main()))
