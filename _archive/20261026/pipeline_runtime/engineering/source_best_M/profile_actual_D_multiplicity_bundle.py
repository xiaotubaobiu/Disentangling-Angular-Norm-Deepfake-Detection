from pathlib import Path
import json,hashlib,time,gc,datetime,resource,os,argparse
from reliability_v2 import multiplicity_bundle as bundle
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';W=E/'code/studies/pipeline_reliability/v2'
resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def canonical(value):
 h=hashlib.sha256();n=0
 for text in json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(value):
  block=text.encode('utf-8');h.update(block);n+=len(block)
 h.update(b'\n');return h.hexdigest(),n+1
p=argparse.ArgumentParser();p.add_argument('--bundle-sha',required=True);args=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
code=W/'reliability_v2/multiplicity_bundle.py';assert sha(code)==args.bundle_sha
original=R/'analysis/D/D_formal_clip_vit_b16_seed7';marker=original/'complete.json';marker_sha='9aa7620cb6a65fed4e2bfd08db6ff0bbab649f794e9850d853bf17529ad1b6c6';assert sha(marker)==marker_sha;accepted=json.loads(marker.read_text())
selected=['target_full/normal_bootstrap_l2_mean_logit.json','target_full/closure_bootstrap.json','rq4/l2_mean_logit/primary_n20_vs_source_bootstrap.json']
out=E/'storage_profiles/D_clip_seed7_atomic_bundle';assert not out.exists();out.mkdir(parents=True)
report=E/'actual_D_atomic_bundle_profile.json';assert not report.exists();records=[];start=time.monotonic();limits=dict(max_cells=40000000,max_json_bytes=2*1024**3,max_array_bytes=512*1024**2)
for index,key in enumerate(selected):
 source=original/key;expected=accepted['artifact_sha256'][key];assert sha(source)==expected and sha(code)==args.bundle_sha and sha(marker)==marker_sha
 print('BEGIN_REAL_D_BUNDLE',index,key,source.stat().st_size,flush=True);tick=time.monotonic();value=json.loads(source.read_bytes());identity=dict(scope='ENGINEERING_TRANSPORT_ONLY_ACCEPTED_LAST_D_PAYLOAD',source_relative=key,source_sha256=expected,source_complete_sha256=marker_sha,source_best_results=False)
 destination=out/str(index);receipt=bundle.write_bundle(destination,value,identity=identity,**limits);assert receipt['raw_json_sha256']==expected and receipt['raw_json_bytes']==source.stat().st_size
 restored=bundle.read_bundle(destination,expected_complete_sha256=receipt['complete_sha256'],expected_identity=identity,**limits)
 assert restored==value and canonical(restored)==(expected,source.stat().st_size)
 assert sha(source)==expected and sha(code)==args.bundle_sha and sha(marker)==marker_sha
 files=list(x for x in destination.iterdir() if x.is_file());inodes={(x.stat().st_dev,x.stat().st_ino):x.stat().st_blocks*512 for x in files}
 record=dict(relative_source=key,source_sha256=expected,original_bytes=source.stat().st_size,stored_logical_bytes=receipt['stored_bytes'],stored_unique_inode_allocated_bytes=sum(inodes.values()),receipt=receipt,all_original_fields_and_ordered_JSON_bytes_exact=True,seconds=time.monotonic()-tick)
 records.append(record);print('PASS_REAL_D_BUNDLE',json.dumps(record),flush=True);del restored,value;gc.collect()
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_THREE_ACTUAL_ACCEPTED_D_PAYLOAD_SCHEMAS_ATOMIC_BOUNDED_BUNDLE_TRANSPORT_ONLY',results=records,seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_complete_sha256=marker_sha,bundle_source_sha256=args.bundle_sha,source_originals_mutated=False,source_best_results_generated=False,H_real_data_profile_complete=False,M_production_admitted=False,joint_capacity_admitted=False,scope='Largest actual D normal TEST bootstrap plus closure and RQ4 n20 bootstrap; all draws/fields/ordered JSON raw bytes restored. This verifies file transport; existing primary LAST results are neither new source-best results nor H storage measurement.')
with report.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print('COMPLETE_REAL_D_BUNDLE_PROFILE',report,flush=True)
