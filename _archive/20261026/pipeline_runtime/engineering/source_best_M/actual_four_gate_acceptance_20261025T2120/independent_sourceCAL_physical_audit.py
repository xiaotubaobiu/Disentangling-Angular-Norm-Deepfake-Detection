from pathlib import Path
from types import SimpleNamespace
import os,json,hashlib,datetime,time,importlib.util,sys,math,shutil,resource
import numpy as np
import pyarrow.parquet as pq
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';W=E/'code/studies/pipeline_reliability/v2';O=E/'actual_four_gate_acceptance_20261025T2120'
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and all(os.environ[k]=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
sys.path[:0]=[str(W),str(W/'tools')]
from reliability_v2 import source_best_cache as cache,source_best_runtime as runtime,source_best_analysis as analysis
spec=importlib.util.spec_from_file_location('independent_fixed_collector',W/'tools/source_best_cache_admission.py');ca=importlib.util.module_from_spec(spec);spec.loader.exec_module(ca)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def load(p,pin=None):
 p=Path(p);assert p.is_file() and not p.is_symlink() and p.stat().st_size<=1024**2
 if pin:assert digest(p)==pin
 return json.loads(p.read_text())
def ref(p):return dict(path=str(p),sha256=digest(p))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def header(p,shape):
 p=cache._inside(p,R,file=True)
 with p.open('rb') as f:
  assert np.lib.format.read_magic(f)==(1,0);s,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=4096);offset=f.tell()
 assert s==shape and fortran is False and dtype==np.dtype('float32') and offset<=4096 and p.stat().st_size==offset+math.prod(shape)*4
 return dict(shape=list(s),dtype=str(dtype),C_order=True,header_bytes=offset,file_bytes=p.stat().st_size,sha256=digest(p))
prep=load(E/'M_actual_four_gate_review_preparation.json','016d1b05fda474af90cba388509a788ae2fe35fbb792164cf4bc1d7a082854f0');meta=load(E/'M_source9_completed_metadata_independent_precheck.json','2e6977fbadbfdc5fe0679f25112b964f0a248fd41e4ab80a84ab1ebd618f3b85')
for p,pin in prep['fixed_code_sha256'].items():assert digest(p)==pin
bridge_path=Path(prep['future_completion_paths']['source9_collector_bridge']);bridge_pin='db47c6be4993e580ed7bf58023b0bee527d10011f7f84b15f44874e8511cba27';bridge=load(bridge_path,bridge_pin)
assert bridge['schema']=='M_GPU_BRIDGE_QUALIFICATION_V1' and bridge['status']=='PASS' and bridge['real_source_only_GPU_qualification_passed'] is True and bridge['phase_native_mutual_exclusion_verified'] is True
assert bridge['collector_code_sha256']==prep['fixed_code_sha256'][str(W/'tools/source_best_cache_admission.py')] and bridge['qualification_worker_code_sha256']==prep['fixed_code_sha256'][str(W/'reliability_v2/source_best_source_qualification.py')]
assert bridge['input_admission_sha256']==prep['input_ref']['sha256'] and bridge['owner_stage']=={k:meta['terminal'][k] for k in ('path','sha256')}
assert set(bridge['results'])==set(prep['fixed_inventory']['source9_run_ids'])
for p,pin in bridge['source_sha256'].items():assert prep['fixed_code_sha256'][p]==pin
term=load(meta['terminal']['path'],meta['terminal']['sha256']);plan=load(prep['initial_plan_ref']['path'],prep['initial_plan_ref']['sha256']);snapshots=[];results=[];manifests={};total_rows=0;payload_bytes=0
slot_requested=utc();acquired=None;released=None;t=time.monotonic()
try:
 with runtime.cpu_slot(R,1):
  acquired=utc();print('SLOT1_ACQUIRED '+acquired,flush=True)
  for prior in meta['results']:
   run=prior['run_id'];role=prior['source_role'];folder=R/'features/source_best_M'/prep['input_ref']['sha256']/run/role;before=ca._derived_snapshot(R,folder,content_checksums=True);snapshots.append((folder,True,before))
   record=ca.owned_role(R,prep['input_ref']['sha256'],run,role,prior['complete_marker']['sha256'],content_checksums=True);assert record==bridge['results'][run]
   complete=load(prior['complete_marker']['path'],prior['complete_marker']['sha256']);identity=complete['identity'];d=record['dimension'];n=record['rows'];assert record['head_sha256']==prior['original_FP32_head_sha256'] and record['checkpoint_sha256']==prior['checkpoint_sha256'] and record['source_best_epoch']==prior['source_best_epoch']
   w_info=header(folder/'head_weight.npy',(2,d));b_info=header(folder/'head_bias.npy',(2,));assert w_info['sha256']==complete['head']['weight_sha256'] and b_info['sha256']==complete['head']['bias_sha256']
   w=np.memmap(folder/'head_weight.npy',mode='r',dtype=np.float32,offset=w_info['header_bytes'],shape=(2,d));b=np.memmap(folder/'head_bias.npy',mode='r',dtype=np.float32,offset=b_info['header_bytes'],shape=(2,));head=hashlib.sha256(w.tobytes(order='C')+b.tobytes(order='C')).hexdigest();assert head==record['head_sha256'] and bool(np.isfinite(w).all()) and bool(np.isfinite(b).all());del w,b
   chunks=[];position=0
   for item in complete['chunks']:
    count=min(4096,n-position);assert item['start']==position and item['rows']==count;h_info=header(folder/item['file'],(count,d));assert h_info['sha256']==item['sha256']
    x=np.memmap(folder/item['file'],mode='r',dtype=np.float32,offset=h_info['header_bytes'],shape=(count,d));assert bool(np.isfinite(x).all());del x
    chunks.append(dict(file=item['file'],**h_info,all_finite=True));position+=count
   assert position==n
   source=Path(identity['source_manifest_path']);assert source==R/f'data/{run[0]}'/(role+'.parquet');assert digest(source)==digest(folder/'manifest.parquet')==record['source_manifest_sha256']
   if str(source) not in manifests:
    pf=pq.ParquetFile(source);assert pf.metadata.num_rows==n and 'label' in pf.schema_arrow.names
    labels=pq.read_table(source,columns=['label']).column('label').to_pylist();assert len(labels)==n and all(type(label) in (int,bool) and int(label)==0 for label in labels)
    manifests[str(source)]=dict(path=str(source),sha256=record['source_manifest_sha256'],rows=n,all_SOURCE_CAL_labels_zero=True,columns_read=['label'])
   total_rows+=n;payload_bytes+=n*d*4;results.append(dict(run_id=run,role=role,bridge_record=record,original_raw_head_sha256_independently_recomputed=head,head_weight=w_info,head_bias=b_info,chunks=chunks,original_head_file_bytes_and_chunk_SHA_verified=True))
  ca._same_derived(R,snapshots)
  for p,pin in prep['fixed_code_sha256'].items():assert digest(p)==pin
  assert digest(bridge_path)==bridge_pin and digest(meta['terminal']['path'])==meta['terminal']['sha256']
  remaining=analysis.remaining_required(R,plan);staging=runtime.staging_usage(R);assert remaining==165195057321 and sum(staging.values())==0
  free=shutil.disk_usage(R).free;assert free>=remaining+1024**2
  assert total_rows==10080 and payload_bytes==34406400
finally:
 released=utc();print('SLOT1_RELEASED '+released,flush=True)
report=dict(schema='INDEPENDENT_SOURCE9_ACTUAL_SOURCE_CAL_PHYSICAL_FORMAT_AUDIT_V1',status='ACTUAL_9_SOURCE_ONLY_FP32_CACHE_FORMAT_HEAD_PAYLOAD_BINDINGS_VERIFIED',utc=utc(),scope='Only9 fixed SOURCE_CAL caches; no original45 checkpoints/PNG, target features, encoder/model, statistics or oldD/H compact payload is read/recomputed.',bridge=ref(bridge_path),metadata_precheck=ref(E/'M_source9_completed_metadata_independent_precheck.json'),owner_terminal=meta['terminal'],fixed_code_sha256=prep['fixed_code_sha256'],slot=dict(index=1,requested_utc=slot_requested,acquired_utc=acquired,released_utc=released,released=True,seconds_total=time.monotonic()-t),results=results,source_manifests=list(manifests.values()),total_rows=total_rows,FP32_vector_bytes=payload_bytes,remaining_required_bytes=remaining,current_retained_M_staging_bytes=sum(staging.values()),live_free_bytes_at_audit=free,publication_margin_bytes=1024**2,cache_only_evidence_verified=True,cache_permit_issued=False,statistics_permit_issued=False,scientific_M_results_accepted=False,GPU_executed_here=False,source_originals_mutated=False,feature_or_compression_credit_bytes=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
out=O/'independent_sourceCAL_physical_audit.json';assert not out.exists()
with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(report=ref(out),slot=report['slot'],rows=total_rows,FP32_vector_bytes=payload_bytes,remaining=remaining,live_free=free,statistics_or_GPU_started=False)),flush=True)
