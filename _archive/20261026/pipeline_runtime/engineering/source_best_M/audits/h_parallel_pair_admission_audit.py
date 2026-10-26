"""Independent completed CLIP parallel pair-admission metadata rebind.
Default preparation only. No F imports, original validation, models, image/chunk reads or scoring.
"""
import argparse,copy,hashlib,importlib.util,json,math,os,re,stat,time
from pathlib import Path
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
HERE=ROOT/'engineering/source_best_M/audits';F=ROOT/'qualification/H_parallel_fresh_validation'
CODE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
REFERENCE=F/'independent_actual_CPU_parallel_pair_admission_audit.json'
REFERENCE_SHA='a7d29d89b9dcfc9fea48c6e6488e1632273fd3980a907161ed3e7de045165bf3'
MANIFEST=ROOT/'qualification/H_primary_cache_continuation/qualified_source_manifest.json'
MANIFEST_SHA='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747'
ALL3_ACCEPTED=ROOT/'reports/H_final_all3_effective_registry_acceptance_20261025T0555.json'
ALL3_SHA='49a620f6de1c58e259d8a214ce6cae188c911e2118d22a3d5f5fcf39c24f83ed'
REGISTRY_SHA='93d9336f5fe57c8af09b0d0177f0e1f9dd1bbab51331347127dec95e4e8dda27'
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0');SEEDS=(7,42,123,2026,3407)
ADMISSION_IMPL={'parallel_source.py':'a6133c824913ab3bf361095b415eefc9ef75be39ab32fb49ab62e2656dc2624a','cpu_ledger_adapter.py':'c6b98535e5a28123baae595ab1f4b208f1feab43ff4018e740be94366d044287','cpu_ledger_admission.py':'bd202e85a4990f9246b96a7081815d07cf364041e5d3498f49bfe41d8779b213'}
IDENTITY_COLUMNS=('video_uid','label','method_id','group_id','mother_id')
SHARED_PATH=HERE/'h_pair_result_audit_before_seed7_binding.py'
SHARED_SHA='12ed094960dc7082215bf6261093d54962ccfb80d557c4ba3e0136baf467b32f'
if SHARED_PATH.is_symlink() or hashlib.sha256(SHARED_PATH.read_bytes()).hexdigest()!=SHARED_SHA:raise AssertionError('accepted shared auditor bytes changed')
_spec=importlib.util.spec_from_file_location('_accepted_H_audit_shared_12ed0949',SHARED_PATH);SHARED=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(SHARED)
sha=SHARED.sha;require=SHARED.require;ROLES=SHARED.ROLES

def hash_string(h):return type(h) is str and re.fullmatch('[0-9a-f]{64}',h) is not None

def candidate_path(seed):return ROOT/f'reports/H_clip_vit_b16_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json'

def extra_paths(registry):
    paths=[ALL3_ACCEPTED,ROOT/'H_run_registry.json',ROOT/'configs/H_execution.json',ROOT/'data/H/freeze.json',ROOT/'qualification/H_gate.json',ROOT/'configs/H_frozen_recipes.json',ROOT/'configs/requirements.execution.lock.txt']
    for b in BACKBONES:paths.extend([ROOT/f'qualification/H_target_export_{b}.json',ROOT/f'reports/H_{b}_AMP_paired_ledger.json'])
    for run in registry['effective_runs']['clip_vit_b16']['run_ids']:
        paths.extend(ROOT/'runs'/run/name for name in ('identity.json','resolved_config.json','history.json','complete.json','H_completion_seal.json'))
    return paths

def reconstruct_pins(reference,manifest,registry,accepted):
    require(reference['schema']=='H_FRESH_PARALLEL_CPU_REAL_RECEIPT_AUDIT_V1' and reference['status']=='PASS' and reference['independent_review_pass'] is True
        and reference['backbone']=='clip_vit_b16' and reference['seed']==42 and reference['current_pin_count']==107
        and reference['qualified_original_source_count']==20 and reference['canonical_accepted_proof_count']==37,'accepted seed42 reference scope')
    require(registry['status']=='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING' and registry['target_export_allowed'] is True and set(registry['effective_runs'])==set(BACKBONES),'all3 final registry')
    proof_paths={str(ROOT/'H_run_registry.json')}
    for b in BACKBONES:
        e=registry['effective_runs'][b];expected=[f'H_formal_{b}_{h}_seed{s}' for s in SEEDS for h in ('B','AB')]
        require(e['precision']=='AMP' and e['run_ids']==expected,'exact accepted all5 AMP effective models '+b)
        proof_paths.update(str(ROOT/'runs'/run/'H_completion_seal.json') for run in expected)
        proof_paths.update([str(ROOT/f'qualification/H_target_export_{b}.json'),str(ROOT/f'reports/H_{b}_AMP_paired_ledger.json')])
    accepted_map=accepted['current_canonical_proof_sha256'];require(set(accepted_map)==proof_paths and len(accepted_map)==37,'exact37 accepted canonical proofs')
    require(manifest['qualified_commit']=='5ef6d06bf7aeda02a5a11113e4c35a2c9909f215' and type(manifest['source_sha256']) is dict and len(manifest['source_sha256'])==20,'qualified20 source manifest')
    pins={str(CODE/n):h for n,h in manifest['source_sha256'].items()}
    require(all(Path(p).parent==CODE/'reliability_v2' and Path(p).suffix=='.py' for p in pins),'canonical20 Python module inventory')
    pins.update({str(F/n):h for n,h in ADMISSION_IMPL.items()});pins[str(MANIFEST)]=MANIFEST_SHA
    pins.update(accepted_map);pins[str(ALL3_ACCEPTED)]=ALL3_SHA
    frozen=reference['pinned_sha256']
    for path in extra_paths(registry):
        name=str(path);require(name in frozen,'expected reference metadata path '+name)
        if name in pins:require(pins[name]==frozen[name],'accepted snapshot conflict '+name)
        else:pins[name]=frozen[name]
    require(len(pins)==107 and pins==frozen and all(hash_string(h) for h in pins.values()),'exact accepted107 hashes/inventory')
    require(pins[str(ROOT/'H_run_registry.json')]==REGISTRY_SHA,'accepted registry byte pin')
    return pins

def validate_candidate(receipt,seed,reference,pins):
    require(type(seed) is int and seed in (123,2026),'prepared original CLIP123/2026 scope only')
    require(receipt.get('status')=='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS'
        and receipt.get('backbone')=='clip_vit_b16' and type(receipt.get('seed')) is int and receipt['seed']==seed
        and type(receipt.get('models')) is int and receipt['models']==2 and type(receipt.get('roles')) is int and receipt['roles']==14,'completed candidate pair/schema')
    require(receipt.get('pinned_sha256')==pins,'candidate exact accepted107 snapshot')
    require(all(receipt.get(k) is False for k in ('cross_invocation_receipt_reuse','production_scoring_executed','GPU_launch_attempted','live_cache_owner_changed')),'candidate operation/authority boundary')
    require(type(receipt.get('all_source_completions_fresh')) is int and receipt['all_source_completions_fresh']==140,'typed140 fresh source-check summary')
    rows=receipt.get('fresh_ledger_invocations');require(type(rows) is list and len(rows)==14,'exact14 ledger receipts')
    ledger_sha=pins[str(ROOT/'reports/H_clip_vit_b16_AMP_paired_ledger.json')]
    for row in rows:
        require(type(row) is dict and type(row.get('fresh_source_completions')) is int and row['fresh_source_completions']==10
            and row.get('backbone')=='clip_vit_b16' and row.get('precision')=='AMP' and row.get('reference_sha256')==ledger_sha
            and row.get('exact_original_ledger_except_top_utc') is True and row.get('cross_invocation_receipt_reuse') is False,'each exact original ledger/fresh10/no reuse')
        require(type(row.get('elapsed_seconds')) in (int,float) and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>=0,'finite ledger timing metadata')
    require(type(receipt.get('elapsed_seconds')) in (int,float) and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds']>0,'finite candidate timing metadata')
    require(sum(r['fresh_source_completions'] for r in rows)==140,'14x10 source-check summary')
    i=receipt['identity'];expected=copy.deepcopy(reference['original_primary_pair_identity'])
    expected.update(seed=seed,run_ids={h:f'H_formal_clip_vit_b16_{h}_seed{seed}' for h in ('B','AB')},cache_completion_sha256=i.get('cache_completion_sha256'))
    require(i==expected and type(i['seed']) is int,'exact reconstructed original pair identity/protocol')
    hashes=i['cache_completion_sha256'];require(type(hashes) is dict and set(hashes)=={h+'/'+r for h in ('B','AB') for r in ROLES}
        and all(hash_string(h) for h in hashes.values()),'exact14 canonical marker hashes')
    return i

def validate_marker(m,seed,history,role,identity,pins,reference,seal,run_identity):
    run=identity['run_ids'][history];i=m['identity'];runroot=ROOT/'runs'/run
    require(m.get('schema')=='pipeline-reliability-v2-H-fp32-cache-1' and m.get('status')=='CACHE_COMPLETE' and m.get('precision')=='FP32' and m.get('autocast') is False
        and m.get('input_validation')=='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES','original FP32 complete role marker')
    require(i['run_id']==run and i['history']==history and i['role']==role and i['backbone']=='clip_vit_b16' and i['artifact_root']==str(ROOT)
        and i['config_path']==str(ROOT/'configs/H_execution.json') and i['config_sha256']==pins[str(ROOT/'configs/H_execution.json')]
        and i['training_precision']==i['effective_precision']=='AMP' and i['epochs']==30 and i['dimension']==m['dimension']==512,'original role/model/config header')
    require(i['checkpoint_path']==str(runroot/'final.pt') and i['checkpoint_sha256']==seal['artifact_sha256']['final.pt'],'original LAST checkpoint/seal hash metadata')
    require(seal['schema']=='H_COMPLETION_SEAL_V1' and seal['status']=='SEALED' and seal['run_id']==run and seal['target_used'] is False
        and i['training_identity']==seal['identity']==run_identity,'accepted source training identity/seal')
    require(i['training_complete_sha256']==pins[str(runroot/'complete.json')] and i['completion_seal_sha256']==pins[str(runroot/'H_completion_seal.json')],'training completion/seal accepted pins')
    require(i['H_run_registry_sha256']==REGISTRY_SHA and i['effective_run_ids']==[f'H_formal_clip_vit_b16_{h}_seed{s}' for s in SEEDS for h in ('B','AB')]
        and i['paired_ledger_path']==str(ROOT/'reports/H_clip_vit_b16_AMP_paired_ledger.json') and i['paired_ledger_sha256']==pins[i['paired_ledger_path']]
        and i['target_export_gate_sha256']==pins[str(ROOT/'qualification/H_target_export_clip_vit_b16.json')]
        and i['H_data_freeze_sha256']==pins[str(ROOT/'data/H/freeze.json')],'current all5 gate/ledger/registry/freeze bindings')
    require(i['h_code_bundle_sha256']==seal['h_code_bundle_sha256']==reference['source_training_h_code_bundle_sha256']
        and i['training_code_sha256']==run_identity['code']==reference['source_training_code_sha256']
        and i['cache_code_sha256']==pins[str(CODE/'reliability_v2/cache_h.py')] and i['cache_helper_code_sha256']==pins[str(CODE/'reliability_v2/cache_d.py')],'qualified original source/cache code binding')
    require(type(m['rows']) is int and m['rows']==i['accepted_table_rows'] and m['rows']>0 and type(i['accepted_table_videos']) is int
        and 0<i['accepted_table_videos']<=m['rows'] and hash_string(i['original_head_sha256']),'accepted role row/head metadata')
    path=ROOT/'data/H'/f'{role}.parquet'
    require(i['source_manifest_path']==str(path) and hash_string(i['source_manifest_sha256']) and m['cached_manifest_sha256']==i['source_manifest_sha256'],'canonical accepted source/cached manifest metadata')
    return dict(history=history,run_id=run,role=role,rows=m['rows'],accepted_videos=i['accepted_table_videos'],chunk_count=len(m['chunks']),original_head_sha256=i['original_head_sha256'],source_manifest_path=str(path),source_manifest_sha256=i['source_manifest_sha256'],input_validation=m['input_validation'])

def validate_footer(metadata,rows):
    require(type(metadata['rows']) is int and metadata['rows']==rows and set(IDENTITY_COLUMNS)<=set(metadata['columns']),'role manifest footer row/schema metadata')

def manifest_footer(path):
    import pyarrow.parquet as pq
    path=Path(path);require(path.resolve()==path and not path.is_symlink() and path.is_file(),'canonical role manifest regular file')
    s=path.stat();p=pq.ParquetFile(path)
    return dict(path=str(path),rows=p.metadata.num_rows,columns=p.schema_arrow.names,size=s.st_size,mtime_ns=s.st_mtime_ns)

def validate_log_time(log,timing,seed):
    starts=re.findall(r'^BEGIN_FRESH_ORIGINAL_LEDGER (\d+) clip_vit_b16 AMP models10_no_receipt_reuse\s*$',log,re.M)
    require(starts==[str(n) for n in range(1,15)],'producer log exact14 fresh ledger starts')
    require(re.findall(r'^PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT (.+)\s*$',log,re.M)==[str(candidate_path(seed))],'producer log canonical success publication')
    require(re.findall(r'^\s*Exit status:\s*(\d+)\s*$',timing,re.M)==['0'],'producer GNUtime true exit0')
    wall=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)',timing);rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',timing)
    require(wall is not None and rss is not None,'producer time resource metadata')
    nums=[float(x) for x in wall.group(1).split(':')];seconds=sum(v*60**n for n,v in enumerate(reversed(nums)))
    return dict(exit_status=0,wall_clock=wall.group(1),wall_seconds=seconds,max_RSS_KiB=int(rss.group(1)))

def load_context(expected_sha):
    path=HERE/'H_parallel_pair_admission_audit_context.json';require(hash_string(expected_sha) and sha(path)==expected_sha,'external audit context SHA')
    c=json.loads(path.read_text());require(c['schema']=='H_PARALLEL_PAIR_ADMISSION_AUDIT_CONTEXT_V1' and c['root']==str(ROOT) and c['backbone']=='clip_vit_b16' and set(c['seeds'])=={'123','2026'},'prepared canonical context scope')
    require(c['shared_source']==str(SHARED_PATH) and c['shared_source_sha256']==SHARED_SHA and c['wrapper_template']==str(ROOT/'engineering/source_best_M/primary_admit_clip_seed123_slot1.py')
        and c['wrapper_template_sha256']=='cd33c85b28dcf82ea0307c0786603a49cf67014f1406ef0d92668a28a6151fa7','accepted shared/source wrapper binding')
    canonical={123:('H_primary_admit_clip_seed123_slot1_launch_20261025T1355.json','H_primary_admit_clip_seed123_slot1_20261025T1355.log','H_primary_admit_clip_seed123_slot1_time_v_20261025T1355.log'),
        2026:('H_primary_admit_clip_seed2026_slot0_20261025T1410_launch.json','H_primary_admit_clip_seed2026_slot0_20261025T1410.log','H_primary_admit_clip_seed2026_slot0_20261025T1410_time_v.log')}
    for seed,x in c['seeds'].items():
        n=int(seed);require(x['candidate']==str(candidate_path(n)) and (x['launch'],x['stdout'],x['time_v'])==tuple(str(ROOT/'reports'/name) for name in canonical[n])
            and hash_string(x['launch_sha256']) and type(x['producer_slot']) is int and x['producer_slot']==(1 if n==123 else 0)
            and type(x['producer_pids']) is list and len(set(x['producer_pids']))==2 and all(type(p) is int and p>0 for p in x['producer_pids']),'prepared production launch/PID/path metadata')
    return c

def validate_launch(launch,seed,context):
    x=context['seeds'][str(seed)];require(launch['pid']==x['producer_pids'][0] and launch['CPU_slot']==x['producer_slot'] and launch['stdout']==x['stdout'] and launch['time_v']==x['time_v']
        and launch['new_dispatcher'] is False and launch['repeat_on_failure'] is False,'bound original finite producer launch')
    template=Path(context['wrapper_template']).read_text()
    if seed==123:
        require(launch['entry']==context['wrapper_template'] and launch['entry_sha256']==context['wrapper_template_sha256'] and launch['original_production_and_F_edited'] is False,'original123 wrapper source binding')
    else:
        expected=template.replace('seed123','seed2026').replace('entry.cpu_slot(1)','entry.cpu_slot(0)').replace('SHARED_CPU_SLOT_1','SHARED_CPU_SLOT_0').replace("'--seed','123'","'--seed','2026'")
        code=launch['parameterized_entry_code'];require(code==expected and hashlib.sha256(code.encode()).hexdigest()==launch['parameterized_code_sha256']
            and launch['entry_template']==context['wrapper_template'] and launch['entry_template_sha256']==context['wrapper_template_sha256'] and launch['production_or_F_files_changed'] is False,'exact seed/slot-only2026 wrapper substitutions')

def audit_owned(seed,context_sha,receipt_sha,time_sha,log_sha):
    c=load_context(context_sha);x=c['seeds'][str(seed)]
    require(all(not Path(f'/proc/{p}').exists() for p in x['producer_pids']),'producer wrapper/child must truly exit; no waiting or termination')
    pins={str(HERE/'H_parallel_pair_admission_audit_context.json'):context_sha,str(REFERENCE):REFERENCE_SHA,str(MANIFEST):MANIFEST_SHA,str(ALL3_ACCEPTED):ALL3_SHA,str(ROOT/'H_run_registry.json'):REGISTRY_SHA,str(SHARED_PATH):SHARED_SHA,str(Path(__file__)):sha(__file__),x['candidate']:receipt_sha,x['time_v']:time_sha,x['stdout']:log_sha,x['launch']:x['launch_sha256'],c['wrapper_template']:c['wrapper_template_sha256'],str(F/'cpu_primary_analysis.py'):'c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce',str(ROOT/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json'):'0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7'}
    def check():
        for p,h in pins.items():require(hash_string(h) and sha(p)==h,'current read-only audit evidence pin '+p)
    check();reference=json.loads(REFERENCE.read_text());manifest=json.loads(MANIFEST.read_text());registry=json.loads((ROOT/'H_run_registry.json').read_text());accepted=json.loads(ALL3_ACCEPTED.read_text())
    expected=reconstruct_pins(reference,manifest,registry,accepted)
    for p,h in {**expected,**reference['protected_review_sha256']}.items():require(p not in pins or pins[p]==h,'pin conflict '+p);pins[p]=h
    check();receipt=json.loads(Path(x['candidate']).read_text());identity=validate_candidate(receipt,seed,reference,expected)
    validate_launch(json.loads(Path(x['launch']).read_text()),seed,c)
    resources=validate_log_time(Path(x['stdout']).read_text(),Path(x['time_v']).read_text(),seed)
    markers=[];footers={};heads={};source_tables={}
    for history,run in identity['run_ids'].items():
        seal=json.loads((ROOT/'runs'/run/'H_completion_seal.json').read_text());ri=json.loads((ROOT/'runs'/run/'identity.json').read_text())
        for role in ROLES:
            p=ROOT/'features/H'/run/role/'complete.json';h=identity['cache_completion_sha256'][history+'/'+role];require(sha(p)==h,'current accepted role marker SHA');pins[str(p)]=h
            record=validate_marker(json.loads(p.read_text()),seed,history,role,identity,expected,reference,seal,ri)
            require(history not in heads or heads[history]==record['original_head_sha256'],'one original FP32 head per history across7roles');heads[history]=record['original_head_sha256']
            require(role not in source_tables or source_tables[role]==(record['source_manifest_sha256'],record['rows'],record['accepted_videos']),'same accepted role inputs across histories');source_tables[role]=(record['source_manifest_sha256'],record['rows'],record['accepted_videos'])
            source=record['source_manifest_path']
            if source not in footers:footers[source]=manifest_footer(source)
            validate_footer(footers[source],record['rows']);record['complete_sha256']=h;markers.append(record)
    check()
    for p,v in footers.items():require(manifest_footer(p)==v,'manifest footer metadata changed during audit')
    require(all(not Path(f'/proc/{p}').exists() for p in x['producer_pids']),'producer exit metadata changed')
    return dict(schema='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1',status='PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND',backbone='clip_vit_b16',seed=seed,reference_audit_sha256=REFERENCE_SHA,canonical_shared107_pin_sha256=expected,reference_pin_inventory_independently_reconstructed=True,qualified_original_code_count=20,canonical_proof_count=37,models=2,roles=14,fresh_ledger_invocations=14,source_checks_per_ledger=10,source_checks_in_ledger_receipts=140,source_validation_freshness_established_by='Completed SHA-bound reviewed original validation/exit0 receipt mechanics; not a duplicate model or byte validation',producer_resources=resources,producer_PIDs_exited=x['producer_pids'],candidate_identity=identity,role_marker_checks=markers,role_manifest_footer_metadata=list(footers.values()),bracketed_readonly_evidence_sha256=pins,shared_lifetime_CPU_slot_required=True,scope='107 metadata/source/proof hashes,14 marker/header/current source-seal bindings,7role manifest footer metadata,14x10 ledger-summary identities,canonical launch/log/time/exit checks only',fresh_model_image_or_chunk_validation_repeated=False,numeric_ledger_regenerated=False,raw_role_manifest_bytes_rehashed=False,primary_statistics_started=False,scientific_metric_or_bootstrap_results_accepted=False,original_or_F_changed=False,production_GPU_allowed=False,M_authorized=False,primary_resource_or_speedup_qualified=False,original_F_primary_preflight_checks_must_be_preserved=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--seed',type=int,choices=(123,2026));p.add_argument('--slot',type=int,choices=(0,1));p.add_argument('--context-sha');p.add_argument('--receipt-sha');p.add_argument('--time-sha');p.add_argument('--log-sha');args=p.parse_args(argv)
    if not args.execute:print('PREPARATION_ONLY: no candidate audit, original validation, CPU-slot lease or background wait.');return 0
    if args.seed is None or args.slot is None or not all(hash_string(v) for v in (args.context_sha,args.receipt_sha,args.time_sha,args.log_sha)):p.error('execute requires seed/free shared slot and externally confirmed context/receipt/time/log SHAs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'empty CUDA/exactly1 numerical thread')
    dest=HERE/f'H_CLIP_seed{args.seed}_parallel_cache_admission_independent_audit.json';require(not dest.exists() and not dest.is_symlink(),'preserve prior independent audit')
    with SHARED.cpu_slot(ROOT/'analysis/H/.cpu_slots',args.slot):
        started=time.monotonic();d=audit_owned(args.seed,args.context_sha,args.receipt_sha,args.time_sha,args.log_sha);d.update(audit_CPU_slot=args.slot,lifetime_CPU_flock_held=True,audit_elapsed_seconds=time.monotonic()-started)
        with dest.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=d['status'],output=str(dest),sha256=sha(dest))));return 0

if __name__=='__main__':raise SystemExit(main())
