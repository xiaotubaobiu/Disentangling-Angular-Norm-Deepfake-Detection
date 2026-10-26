from pathlib import Path
import json,hashlib,gc,time,datetime,resource
import numpy as np
from reliability_v2.sensitivity_io import read_json
from reliability_v2.multiplicity_codec import decode_counts
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';out=E/'storage_profiles/D_clip_seed7_multiplicity';profile=json.loads((E/'existing_D_multiplicity_profile.json').read_text());resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
source=E/'code/studies/pipeline_reliability/v2/reliability_v2/multiplicity_codec.py';before=sha(source);records=[];start=time.monotonic()
for record in profile['results']:
 key=record['relative_source'];metadata=out/(key+'.metadata.json.gz');npz=out/(key+'.arrays.npz');assert sha(npz)==record['arrays_sha256'];t=record['metadata_transport'];b=read_json(metadata,compressed_sha256=t['compressed_sha256'],uncompressed_sha256=t['uncompressed_sha256'],max_uncompressed_bytes=64*1024**2)
 with np.load(npz,allow_pickle=False) as arrays:b['arrays']={k:arrays[k].copy() for k in arrays.files}
 value=decode_counts(b,max_cells=100000000);h=hashlib.sha256()
 for text in json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(value):h.update(text.encode('utf-8'))
 h.update(b'\n');assert h.hexdigest()==record['original_sha256'];records.append(dict(relative_source=key,raw_SHA_matches_original=True));print('REVALIDATED',key,flush=True);del b,value;gc.collect()
assert sha(source)==before;report=E/'matrix_after_MC001_real_readback.json';assert not report.exists();report.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_FIXED_DECODER_ALL29_REAL_D_ORIGINAL_RAW_SHA_RECONSTRUCTION','files':len(records),'records':records,'fixed_source_sha256':before,'seconds':time.monotonic()-start,'M_production_admitted':False,'H_real_profile_complete':False,'scope':'Existing accepted LAST packed artifacts only; no new source-best inference/bootstrap or source data change. This verifies the fail-closed coverage fix preserves all original bytes for every valid D payload.'},indent=2)+'\n');print('ALL_FIXED_REVALIDATION_PASS',len(records),flush=True)
