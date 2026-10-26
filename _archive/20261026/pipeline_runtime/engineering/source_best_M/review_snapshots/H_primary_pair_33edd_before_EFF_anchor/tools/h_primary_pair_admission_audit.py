"""Finite original H LAST admission metadata audit, preparation-only by default.

No dispatcher, training, scoring, GPU, model/image/chunk read or Q issuance.
Qualified mode: CLIP/DINO only, anchored to their accepted own-backbone real Q.
Prototype mode: DINO7/EFF7 only. EFF formal remains disabled.

Future producer contract (root launches once; this module never launches it):
  wrapper_format H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1; attempt_utc YYYYMMDDTHHMMSSZ
  reports/H_primary_admit_<backbone>_seed<N>_slot<S>_<attempt>_launch.json
  stdout same stem.log; GNUtime same stem_time_v.log
  witness audits/<stem>_actual_producer_process_witness.json
  binding audits/<stem>_<mode>_matrix_admission_bindings.json
Code is generated_wrapper_code's exact registered substitutions of the pinned
DINO launch template. Fields include template_original_DINO_launch/SHA,
parameterized_entry_code/SHA, backbone/seed/CPU_slot/pid/utc, original entry/SHA,
stdout/time_v/candidate, new_dispatcher=False, repeat_on_failure=False,
production_or_F_files_changed=False and primary_scientific_scoring_started=False.
Existing three actual harness paths/bytes remain supported exactly. Each mode uses
a distinct binding; historical a65 five-ref bindings and their accepted A/Q stay
unchanged. A qualified binding adds Q to the same five producer proof refs.

--execute requires a byte-identical absolute canonical snapshot at
E/qualification_snapshots/H_primary_pair_admission_<this SHA>/entry.py, externally
SHA-bound5 prototype or6 qualified refs, terminal proof and the existing lifetime
CPU slot. Engineering tests replace that slot with a non-acquiring counter.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,importlib.util,json,os,re,stat,tempfile,time
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
E=ROOT/'engineering/source_best_M';HERE=E/'audits'
HELPER_SHA='a65b198602e66662de005c6162563f7286041808c9cafd3b4630b1755c0e3196'
HELPER_PATH=E/f'qualification_snapshots/H_matrix_admission_{HELPER_SHA}/entry.py'
BB_SHA='bb299e512e355fa0a8c9b25f901b1788cee6a2b445a4eb1ddb67b30cb4c3a0e2'
BB_PATH=E/'code/studies/pipeline_reliability/v2/tools/h_last_matrix_result_audit.py'
TEMPLATE_PATH=ROOT/'reports/H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json'
TEMPLATE_SHA='d77d73e3ab0c4f3ec6d9338f842990df9db3b85044074b3443fdce968fa1a46a'
LABELS={'clip_vit_b16':'CLIP','dinov2_vit_b14':'DINO','efficientnet_b0':'EFF'}
SEEDS=(7,42,123,2026,3407)
CPU_HASHES={'parallel_source.py':'a6133c824913ab3bf361095b415eefc9ef75be39ab32fb49ab62e2656dc2624a','cpu_ledger_adapter.py':'c6b98535e5a28123baae595ab1f4b208f1feab43ff4018e740be94366d044287','cpu_ledger_admission.py':'bd202e85a4990f9246b96a7081815d07cf364041e5d3498f49bfe41d8779b213','cpu_primary_analysis.py':'c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce'}
ENTRY_REVIEW_SHA='67c9588c6960b9c5f29e513ae22bb8ba57d10b8e201b4bf1d190f93a74d8fc23'
QUALIFICATIONS={
 'clip_vit_b16':dict(path=str(ROOT/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json'),sha256='0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7',prototype_seed=42,candidate_sha256='3e6c6366089ac131804e26d260e40f7af7c0bac92c87ee2b763d8a507c8c48f7',audit_path=str(ROOT/'qualification/H_parallel_fresh_validation/independent_actual_CPU_parallel_pair_admission_audit.json'),audit_sha256='a7d29d89b9dcfc9fea48c6e6488e1632273fd3980a907161ed3e7de045165bf3'),
 'dinov2_vit_b14':dict(path=str(ROOT/'reports/H_parallel_CPU_actual_qualification_dinov2_vit_b14.json'),sha256='ab853480f704f24d4b080e04f0bcefb81fd31c4426e74aa36a351d7e8925bebe',prototype_seed=7,candidate_sha256='ff866b5a2504fa56d0c447b4ae2b23dba20af297a272331978dd13ee0ba050c8',audit_path=str(ROOT/'qualification/H_parallel_fresh_validation/independent_actual_CPU_parallel_pair_admission_audit_dinov2_vit_b14.json'),audit_sha256='0b5b9c1011fa2159066112f0ac5d530b866f168882df76a1ab874b1deb3baa85')}
LEGACY={('clip_vit_b16',3407):('H_primary_admit_clip_seed3407_slot1_20261025T2000',1,'b6aca21f368587e26430c5ef3f7bf9a13d170083f19b58ddff3874f13bda8131'),('dinov2_vit_b14',7):('H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030',0,'2e05c7339e89c259a368d0a5f8592404f446704093ab0134347d6aa32b4b4545'),('efficientnet_b0',7):('H_primary_EFF_CPU_qualification_candidate_seed7_slot0_20261025T2155',0,'261e6ecebc24990098ecaad3d961304c5db5bfe262ced047f8e01c3f9b532057')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(test,message):
 if not test:raise AssertionError(message)
def check_pins(pins):
 for p,h in pins.items():
  path=Path(p);require(path.resolve()==path and not path.is_symlink() and path.is_file() and sha(path)==h,'current canonical metadata pin '+p)
def merge_pins(pins,new):
 for p,h in new.items():
  require(type(p) is str and type(h) is str and re.fullmatch('[0-9a-f]{64}',h) and (p not in pins or pins[p]==h),'conflicting immutable metadata pin '+str(p));pins[p]=h
check_pins({str(HELPER_PATH):HELPER_SHA,str(BB_PATH):BB_SHA})
_spec=importlib.util.spec_from_file_location('_private_frozen_a65_metadata',HELPER_PATH);helpers=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(helpers)
old=helpers.old
# Compile only the exact accepted pure JSON functions, never import numeric code.
_tree=ast.parse(BB_PATH.read_text());_functions=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_int','validate_ledgers','validate_qualification_evidence')]
require(len(_functions)==3,'accepted pure qualification helpers');_qns=dict(Path=Path,require=require,SEEDS=SEEDS,CPU_HASHES=CPU_HASHES);exec(compile(ast.Module(body=_functions,type_ignores=[]),str(BB_PATH),'exec'),_qns);qualification_evidence=_qns['validate_qualification_evidence'];check_pins({str(HELPER_PATH):HELPER_SHA,str(BB_PATH):BB_SHA})

def validate_mode(bone,seed,mode):
 require(bone in LABELS and type(seed) is int and seed in SEEDS,'registered original H3bone/five-seed scope')
 require((mode=='qualified_pair' and bone in QUALIFICATIONS) or (mode=='actual_qualification_prototype' and bone in ('dinov2_vit_b14','efficientnet_b0') and seed==7),'own actual qualification required; EFF formal disabled')

def generated_wrapper_code(bone,seed,slot):
 require(bone in LABELS and type(seed) is int and seed in SEEDS and type(slot) is int and slot in (0,1),'finite wrapper parameters')
 check_pins({str(TEMPLATE_PATH):TEMPLATE_SHA});code=read(TEMPLATE_PATH)['parameterized_entry_code']
 require(hashlib.sha256(code.encode()).hexdigest()==LEGACY[('dinov2_vit_b14',7)][2],'exact original DINO code template')
 return code.replace('dinov2_vit_b14',bone).replace('DINO',LABELS[bone]).replace('seed7',f'seed{seed}').replace("'--seed','7'",f"'--seed','{seed}'").replace('entry.cpu_slot(0)',f'entry.cpu_slot({slot})')

def launch_layout(L,bone,seed,mode=None):
 slot=L['CPU_slot'];require(type(slot) is int and slot in (0,1),'typed original slot')
 if L.get('wrapper_format')=='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1':
  stamp=L['attempt_utc'];require(type(stamp) is str and re.fullmatch(r'\d{8}T\d{6}Z',stamp),'restricted UTC attempt suffix');require(datetime.datetime.strptime(stamp,'%Y%m%dT%H%M%SZ').strftime('%Y%m%dT%H%M%SZ')==stamp,'valid UTC attempt')
  stem=f'H_primary_admit_{bone}_seed{seed}_slot{slot}_{stamp}';witness=HERE/f'{stem}_actual_producer_process_witness.json';binding=HERE/f'{stem}_matrix_admission_bindings.json'
 else:
  require((bone,seed) in LEGACY and 'wrapper_format' not in L,'only exact legacy or registered template');stem=LEGACY[(bone,seed)][0];witness=HERE/f'H_{bone}_seed{seed}_actual_producer_process_witness.json';binding=HERE/f'H_{bone}_seed{seed}_matrix_admission_bindings.json'
 if mode is not None:
  helpers.expected_refs(mode);binding=HERE/f'{stem}_{mode}_matrix_admission_bindings.json'
 return dict(launch=ROOT/'reports'/f'{stem}_launch.json',candidate=ROOT/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json',stdout=ROOT/'reports'/f'{stem}.log',time_v=ROOT/'reports'/f'{stem}_time_v.log',process_witness=witness),binding

def validate_launch(L,bone,seed,path):
 require(bone in LABELS and type(seed) is int and seed in SEEDS and L['backbone']==bone and helpers.typed(L['seed'],seed) and type(L['pid']) is int and L['pid']>0,'bound original finite launch identity')
 names,_=launch_layout(L,bone,seed);require(Path(path)==names['launch'] and all(L[k]==str(names[k]) for k in ('candidate','stdout','time_v')),'canonical launch/output paths')
 require(all(L.get(k) is False for k in ('new_dispatcher','repeat_on_failure','production_or_F_files_changed')),'finite original launch boundaries')
 require(L.get('primary_scientific_scoring_started',False) is False and (L.get('wrapper_format') is None and bone=='clip_vit_b16' or 'primary_scientific_scoring_started' in L),'finite no-scoring launch declaration or exact historical CLIP schema')
 code=L['parameterized_entry_code'];require(L['parameterized_code_sha256']==hashlib.sha256(code.encode()).hexdigest(),'launch exact code digest')
 if L.get('wrapper_format')=='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1':
  require(L['template_original_DINO_launch']==str(TEMPLATE_PATH) and L['template_original_DINO_launch_sha256']==TEMPLATE_SHA and code==generated_wrapper_code(bone,seed,L['CPU_slot']),'exact finite template substitutions')
 else:
  _,slot,code_sha=LEGACY[(bone,seed)];require(L['CPU_slot']==slot and L['parameterized_code_sha256']==code_sha,'exact existing actual launch bytes/slot')
  if bone!='efficientnet_b0':helpers.validate_launch(L,bone,seed)
  else:require(L['template_original_DINO_launch']==str(TEMPLATE_PATH) and L['template_original_DINO_launch_sha256']==TEMPLATE_SHA and L['other_backbone_qualification_reused'] is False,'EFF independent actual prototype')
 if bone!='clip_vit_b16' or L.get('wrapper_format'):
  require(L['entry']==str(old.F/'cpu_ledger_admission.py') and L['entry_sha256']==CPU_HASHES['cpu_ledger_admission.py'],'qualified original CPU entry')

def validate_qualification_body(bone,Q,C,A,R,reference,expected):
 require(Q.get('schema')=='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1' and Q.get('status')=='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE' and Q.get('allowed_backbones')==[bone] and Q.get('allowed_use')==['independent_role_admission','primary_pair_analysis'],'own qualified use/backbone')
 require(Q.get('independent_review_pass') is True and Q.get('production_GPU_allowed') is False and Q.get('all_original_ledger_checks_exact_except_utc') is True and helpers.typed(Q.get('original_role_checks'),14) and helpers.typed(Q.get('fresh_default_source_checks'),140),'qualified typed14/140 boundaries')
 require(Q.get('implementation_sha256')==CPU_HASHES and Q.get('original_source_manifest_sha256')==old.MANIFEST_SHA,'qualified four-file/source20 family')
 require(C['pinned_sha256']==A.get('pinned_sha256')==expected,'qualification and target current107 same family')
 helpers.validate_candidate(C,bone,C['identity']['seed'],reference,expected)
 qualification_evidence(bone,Q,C,A,R,precision='AMP')
 require(all(A.get(k) is True for k in ('independently_reconstructed_exact107_pin_inventory','all107_current_hashes_match_receipt_and_accepted_expectations','pins_rechecked_at_audit_end','original_identity_reconstructed_exactly','all14_current_marker_hashes_and_identity_headers_match','all14_roles_retain_fresh_PNG_byte_and_decoded_RGB_validation_identity')),'accepted independent real metadata body')
 for k,n in (('current_pin_count',107),('qualified_original_source_count',20),('canonical_accepted_proof_count',37)):require(helpers.typed(A.get(k),n),'qualification audit typed inventory')

def validate_qualification(bone,qref,reference,expected,pins):
 require(bone in QUALIFICATIONS,'own accepted Q exists');spec=QUALIFICATIONS[bone]
 require(qref==dict(path=spec['path'],sha256=spec['sha256']),'external canonical fixed own Q')
 qpath=Path(spec['path']);merge_pins(pins,{str(qpath):spec['sha256']});check_pins(pins);Q=read(qpath)
 aux={str(ROOT/f'reports/H_{bone}_seed{spec["prototype_seed"]}_independent_7role_parallel_candidate_20261025T0645.json'):spec['candidate_sha256'],spec['audit_path']:spec['audit_sha256'],str(old.F/'independent_cpu_primary_analysis_entry_fix_confirmation.json'):ENTRY_REVIEW_SHA}
 keys=('candidate_report','independent_actual_audit','primary_entry_review')
 require({Q[k]:Q[k+'_sha256'] for k in keys}==aux,'canonical own Q three auxiliary paths/SHA')
 merge_pins(pins,aux);check_pins(pins);C,A,R=[read(Q[k]) for k in keys];require(helpers.typed(C['seed'],spec['prototype_seed']),'own actual prototype registered seed');validate_qualification_body(bone,Q,C,A,R,reference,expected);return Q

def load_binding(binding,mode):
 path=Path(binding['path']);require(path.parent==HERE and path.name.startswith('H_') and path.name.endswith('_matrix_admission_bindings.json'),'canonical external binding');check_pins({str(path):binding['sha256']});b=read(path);require(b['schema']=='H_MATRIX_ADMISSION_EXTERNAL_BINDINGS_V1' and b['root']==str(ROOT),'external binding root/schema');bone,seed=b['backbone'],b['seed'];validate_mode(bone,seed,mode);refs=b['proofs'];require(set(refs)==helpers.expected_refs(mode),'exact5/6 external proof inventory')
 lp=Path(refs['launch']['path']);require(lp.parent==ROOT/'reports' and lp.name.endswith('_launch.json'),'canonical launch namespace');check_pins({str(lp):refs['launch']['sha256']});L=read(lp);validate_launch(L,bone,seed,lp);names,bp=launch_layout(L,bone,seed,mode=mode);require(path==bp,'canonical per-launch binding')
 if mode=='qualified_pair':names['CPU_qualification']=Path(QUALIFICATIONS[bone]['path'])
 for k,r in refs.items():require(set(r)=={'path','sha256'} and r['path']==str(names[k]) and old.hash_string(r['sha256']),'canonical finite external proof '+k)
 return b,L,names

def manifest_fingerprint(path):
 path=Path(path);require(path.resolve()==path and not path.is_symlink(),'canonical manifest fingerprint path')
 observed=path.stat();require(stat.S_ISREG(observed.st_mode),'regular manifest fingerprint file')
 return dict(dev=observed.st_dev,inode=observed.st_ino,size=observed.st_size,mtime_ns=observed.st_mtime_ns,ctime_ns=observed.st_ctime_ns)

def check_manifest_fingerprints(d):
 fingerprints=d['role_manifest_fingerprint_metadata'];require(len(fingerprints)==7 and set(fingerprints)=={v['path'] for v in d['role_manifest_footer_metadata']},'exact7 manifest fingerprint inventory')
 for path,expected in fingerprints.items():require(manifest_fingerprint(path)==expected,'manifest fingerprint drift during audit '+path)

def final_check(d):
 require_snapshot(d['audit_source_sha256']);require(d['audit_source_path']==str(Path(__file__)),'original audit source path')
 check_pins(d['bracketed_readonly_evidence_sha256']);check_manifest_fingerprints(d);helpers.validate_exit(read(d['producer_launch_path']),read(d['producer_process_witness_path']))
 for v in d['role_manifest_footer_metadata']:require(old.manifest_footer(v['path'])==v,'manifest footer changed during audit')
 # A later footer read must not hide a change to an earlier manifest. Rebind
 # the entire initial stat inventory after the sweep and again after code/Q pins.
 check_manifest_fingerprints(d);check_pins(d['bracketed_readonly_evidence_sha256']);require_snapshot(d['audit_source_sha256']);check_manifest_fingerprints(d)

def audit_owned(binding,mode,slot,*,source_sha):
 require_snapshot(source_sha)
 b,L,names=load_binding(binding,mode);bone,seed=b['backbone'],b['seed'];refs=b['proofs'];pins={}
 merge_pins(pins,{str(HELPER_PATH):HELPER_SHA,str(BB_PATH):BB_SHA,str(TEMPLATE_PATH):TEMPLATE_SHA,str(helpers.OLD_PATH):helpers.OLD_SHA,str(old.SHARED_PATH):old.SHARED_SHA,str(old.REFERENCE):old.REFERENCE_SHA,str(old.MANIFEST):old.MANIFEST_SHA,str(old.ALL3_ACCEPTED):old.ALL3_SHA,str(Path(__file__)):source_sha,binding['path']:binding['sha256'],str(old.F/'cpu_primary_analysis.py'):CPU_HASHES['cpu_primary_analysis.py']});merge_pins(pins,{r['path']:r['sha256'] for r in refs.values()});check_pins(pins)
 reference=read(old.REFERENCE);expected=helpers.reconstruct_pins(bone,reference,read(old.MANIFEST),read(ROOT/'H_run_registry.json'),read(old.ALL3_ACCEPTED));merge_pins(pins,expected);merge_pins(pins,reference['protected_review_sha256']);check_pins(pins)
 PIDs=helpers.validate_witness(read(names['process_witness']),L,names['launch'],refs['launch']['sha256']);C=read(names['candidate']);identity=helpers.validate_candidate(C,bone,seed,reference,expected);resources=helpers.validate_logs(names['stdout'].read_text(),names['time_v'].read_text(),bone,seed)
 if mode=='qualified_pair':validate_qualification(bone,refs['CPU_qualification'],reference,expected,pins)
 records=[];footers={};fingerprints={};heads={};tables={}
 for history,run in identity['run_ids'].items():
  rr=ROOT/'runs'/run;seal=read(rr/'H_completion_seal.json');ri=read(rr/'identity.json')
  for role in old.ROLES:
   p=ROOT/'features/H'/run/role/'complete.json';h=identity['cache_completion_sha256'][history+'/'+role];merge_pins(pins,{str(p):h});check_pins({str(p):h});record=helpers.validate_marker(read(p),bone,history,role,identity,expected,reference,seal,ri)
   require(history not in heads or heads[history]==record['original_head_sha256'],'same FP32 head across7roles');heads[history]=record['original_head_sha256'];table=(record['source_manifest_sha256'],record['rows'],record['accepted_videos']);require(role not in tables or tables[role]==table,'same cohort across histories');tables[role]=table;source=record['source_manifest_path']
   if source not in footers:
    before=manifest_fingerprint(source);footers[source]=old.manifest_footer(source);require(manifest_fingerprint(source)==before,'manifest changed during initial footer binding');fingerprints[source]=before
   old.validate_footer(footers[source],record['rows']);record['complete_sha256']=h;records.append(record)
 d=dict(schema='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1',status='PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND',backbone=bone,seed=seed,mode=mode,reference_audit_sha256=old.REFERENCE_SHA,canonical_shared107_pin_sha256=expected,reference_pin_inventory_independently_reconstructed=True,qualified_original_code_count=20,canonical_proof_count=37,models=2,roles=14,fresh_ledger_invocations=14,source_checks_per_ledger=10,source_checks_in_ledger_receipts=140,producer_resources=resources,producer_PIDs_exited=PIDs,candidate_identity=identity,role_marker_checks=records,role_manifest_footer_metadata=list(footers.values()),role_manifest_fingerprint_metadata=fingerprints,audit_source_path=str(Path(__file__)),audit_source_sha256=source_sha,bracketed_readonly_evidence_sha256=pins,producer_launch_path=str(names['launch']),producer_process_witness_path=str(names['process_witness']),audit_CPU_slot=slot,lifetime_CPU_flock_held=True,source_validation_freshness_established_by='SHA-bound original14role/140fresh receipt, qualified source mechanics, exact ledger summaries, canonical launch/log/time and observed birth/exit; no repeated payload checks',fresh_model_image_or_chunk_validation_repeated=False,numeric_ledger_regenerated=False,raw_role_manifest_bytes_rehashed=False,primary_statistics_started=False,scientific_metric_or_bootstrap_results_accepted=False,original_or_F_changed=False,production_GPU_allowed=False,M_authorized=False,primary_resource_or_speedup_qualified=False,original_F_primary_preflight_checks_must_be_preserved=True,actual_CPU_qualification_issued=False)
 final_check(d);return d

def require_snapshot(expected_sha=None):
 path=Path(__file__);digest=sha(path)
 require(expected_sha is None or (old.hash_string(expected_sha) and digest==expected_sha),'audit source digest changed after initial snapshot check')
 require(path==E/f'qualification_snapshots/H_primary_pair_admission_{digest}/entry.py' and path.resolve()==path and not path.is_symlink(),'actual execution requires exact immutable source snapshot')
 return digest

def publish(out,d):
 payload=(json.dumps(d,indent=2,allow_nan=False)+'\n').encode();final_check(d)
 with tempfile.NamedTemporaryFile(dir=HERE,prefix='.'+out.name+'.',suffix='.partial',delete=False) as f:
  temp=Path(f.name);f.write(payload);f.flush();os.fsync(f.fileno())
 final_check(d);os.link(temp,out);temp.unlink()

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--binding');p.add_argument('--binding-sha');p.add_argument('--mode',choices=('qualified_pair','actual_qualification_prototype'));p.add_argument('--slot',type=int,choices=(0,1));a=p.parse_args(argv)
 if not a.execute:return dict(preparation_only=True,production_admission_started=False,future_wrapper_format='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1',future_attempt_utc='YYYYMMDDTHHMMSSZ',future_stem='H_primary_admit_<backbone>_seed<N>_slot<S>_<attempt>',future_witness='<stem>_actual_producer_process_witness.json',future_binding='<stem>_<mode>_matrix_admission_bindings.json',qualified_backbones=list(QUALIFICATIONS),prototype_pairs=[['dinov2_vit_b14',7],['efficientnet_b0',7]],actual_CPU_qualification_issued=False)
 if a.slot is None or not a.mode or not a.binding or not old.hash_string(a.binding_sha):p.error('external binding/sha/mode/free shared CPU slot required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only1thread metadata audit');source_sha=require_snapshot();binding=dict(path=str(Path(a.binding)),sha256=a.binding_sha);b,_,_=load_binding(binding,a.mode);label=LABELS[b['backbone']];out=HERE/(f'H_{label}_seed{b["seed"]}_actual_CPU_qualification_metadata_audit.json' if a.mode=='actual_qualification_prototype' else f'H_{label}_seed{b["seed"]}_parallel_cache_admission_independent_audit.json');require(not out.exists() and not out.is_symlink(),'preserve independent outputs')
 with old.SHARED.cpu_slot(ROOT/'analysis/H/.cpu_slots',a.slot):
  started=time.monotonic();d=audit_owned(binding,a.mode,a.slot,source_sha=source_sha);d['audit_elapsed_seconds']=time.monotonic()-started;publish(out,d)
 return dict(output=str(out),sha256=sha(out),status=d['status'],mode=a.mode,CPU_qualification_issued=False)
if __name__=='__main__':print(json.dumps(main()))
