"""One-shot engineering transport snapshot; no final scientific/resource authority."""
from pathlib import Path
import datetime,gc,hashlib,json,os,resource,time,shutil
from reliability_v2 import multiplicity_bundle as bundle
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';W=E/'code/studies/pipeline_reliability/v2'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def regular(path):
 p=Path(path);assert p.is_file() and p.resolve()==p and not p.is_symlink();return p
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(x)=='2' for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
folder=R/'analysis/H/clip_vit_b16/seed42';identity=regular(folder/'identity.json');code=regular(W/'reliability_v2/multiplicity_bundle.py');assert sha(code)=='3aa5debf7d98bd976bc21fa5be2135d34cdfa8e0b91873f356cc8f57e24360c3'
launch=regular(R/'reports/H_primary_analysis_clip_vit_b16_seed42_launch_20261025T0830.json')
cache_proof=regular(R/'reports/H_clip_vit_b16_seed42_independent_7role_parallel_candidate_20261025T0645.json');assert sha(cache_proof)=='3e6c6366089ac131804e26d260e40f7af7c0bac92c87ee2b763d8a507c8c48f7'
cache_audit=regular(R/'qualification/H_parallel_fresh_validation/independent_actual_CPU_parallel_pair_admission_audit.json');assert sha(cache_audit)=='a7d29d89b9dcfc9fea48c6e6488e1632273fd3980a907161ed3e7de045165bf3'
command=Path('/proc/4056924/cmdline').read_bytes().split(b'\0');assert str(R/'qualification/H_parallel_fresh_validation/cpu_primary_analysis.py').encode() in command and b'42' in command and b'clip_vit_b16' in command
names=[f'{a}/{d}/all_B/conditional_bootstrap.json' for a in ('mean_logit','mean_probability') for d in ('SOURCE_TEST','TARGET_TEST')]
source_files=[regular(folder/n) for n in names];assert all(p.stat().st_size<64*1024**2 for p in source_files)
pins={str(p):sha(p) for p in [identity,code,launch,cache_proof,cache_audit,Path(__file__).resolve(),*source_files]}
limits=dict(max_cells=4000000,max_json_bytes=64*1024**2,max_array_bytes=64*1024**2)
worst_bytes=len(names)*(limits['max_json_bytes']+limits['max_array_bytes']+8*1024**2)
assert shutil.disk_usage(R).free>=114837553152+worst_bytes
out=E/'storage_profiles/H_clip_seed42_live_producer_snapshot';assert not out.exists();out.mkdir(parents=True)
report=E/'actual_H_live_producer_snapshot_profile.json';assert not report.exists();records=[];start=time.monotonic()
for number,(key,p) in enumerate(zip(names,source_files)):
 print('BEGIN_REAL_H_SNAPSHOT',key,p.stat().st_size,flush=True);value=json.loads(p.read_bytes());assert value['repeats']==2000 and len(value['draws'])==2000
 context=dict(scope='ENGINEERING_LIVE_PRIMARY_H_NATIVE_PAYLOAD_SNAPSHOT_NOT_FINAL_SCIENCE',source_relative=key,source_sha256=pins[str(p)],producer_identity_sha256=pins[str(identity)],independent_cache_admission_sha256=pins[str(cache_proof)],source_best_results=False)
 receipt=bundle.write_bundle(out/str(number),value,identity=context,**limits)
 restored=bundle.read_bundle(out/str(number),expected_complete_sha256=receipt['complete_sha256'],expected_identity=context,**limits)
 assert restored==value and receipt['raw_json_sha256']==pins[str(p)] and receipt['raw_json_bytes']==p.stat().st_size
 record=dict(relative_source=key,raw_bytes=p.stat().st_size,receipt=receipt,all_original_fields_NA_count_maps_key_order_and_2000_draws_preserved=True);records.append(record);print('PASS_REAL_H_SNAPSHOT',key,receipt['stored_bytes'],flush=True);del value,restored;gc.collect()
for path,pin in pins.items():assert sha(path)==pin,'input/code changed during snapshot:'+path
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_FOUR_ACTUAL_H_NATIVE_PAYLOAD_SNAPSHOT_ATOMIC_ORDERED_JSON_TRANSPORT_ONLY',producer_scope='Still-running accepted primary LAST H pair; final pair/source revalidation/metrics audit remain pending',results=records,input_sha256=pins,raw_bytes=sum(x['raw_bytes'] for x in records),stored_bytes=sum(x['receipt']['stored_bytes'] for x in records),seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,primary_pair_terminal_accepted=False,source_best_results_generated=False,final_real_H_transport_admission=False,joint_M_budget_admitted=False,compression_capacity_credit_bytes=0,M_production_admitted=False,source_originals_mutated=False,next_gate='After actual primary pair terminal+independent scientific acceptance, match its four artifact SHA and identity to this snapshot; independently re-read these existing bundles and promote measurement without re-encoding or resampling.')
with report.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print('REAL_H_SNAPSHOT_PROFILE_COMPLETE',str(report),result['raw_bytes'],result['stored_bytes'],flush=True)
