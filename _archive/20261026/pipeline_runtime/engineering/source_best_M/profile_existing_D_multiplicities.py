from pathlib import Path
import json,hashlib,time,gc,datetime,shutil,resource,zipfile
import numpy as np
from reliability_v2.multiplicity_codec import encode_counts,decode_counts
from reliability_v2.sensitivity_io import write_json,read_json
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';V=E/'code/studies/pipeline_reliability/v2'
resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
source=R/'analysis/D/D_formal_clip_vit_b16_seed7';marker=source/'complete.json';marker_hash=sha(marker);m=json.loads(marker.read_text());out=E/'storage_profiles/D_clip_seed7_multiplicity';assert not out.exists();out.mkdir(parents=True);dependencies={str(V/'reliability_v2'/name):sha(V/'reliability_v2'/name) for name in ['multiplicity_codec.py','sensitivity_io.py']};results=[];start=time.monotonic()
for p in sorted(x for x in source.rglob('*.json') if x!=marker):
 key=str(p.relative_to(source));expected=m['artifact_sha256'][key];assert sha(p)==expected;print('MATRIX_BEGIN',key,p.stat().st_size,flush=True)
 original=json.loads(p.read_bytes());bundle=encode_counts(original,max_cells=100000000);dest=out/(key+'.arrays.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**bundle['arrays']);meta={k:v for k,v in bundle.items() if k!='arrays'};metadata=out/(key+'.metadata.json.gz');transport=write_json(metadata,meta)
 with zipfile.ZipFile(dest) as archive:assert sum(x.file_size for x in archive.infolist())<=sum(a.nbytes for a in bundle['arrays'].values())+4096*max(1,len(bundle['arrays']))
 recovered=read_json(metadata,compressed_sha256=transport['compressed_sha256'],uncompressed_sha256=transport['uncompressed_sha256'],max_uncompressed_bytes=64*1024**2)
 with np.load(dest,allow_pickle=False) as arrays:recovered['arrays']={name:arrays[name].copy() for name in arrays.files}
 decoded=decode_counts(recovered,max_cells=100000000);assert decoded==original
 # Reencoding with the original standard pretty JSON convention checks byte identity too.
 h=hashlib.sha256();n=0
 for text in json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(decoded):
  block=text.encode('utf-8');h.update(block);n+=len(block)
 h.update(b'\n');n+=1
 assert sha(p)==expected and sha(marker)==marker_hash
 record=dict(relative_source=key,original_bytes=p.stat().st_size,packed_bytes=dest.stat().st_size+metadata.stat().st_size,tables=len(bundle['tables']),matrix_cells=sum(a.size for name,a in bundle['arrays'].items() if name.startswith('counts_')),all_original_JSON_values_and_mappings_equal=True,standard_reencoded_raw_SHA_matches_original=h.hexdigest()==expected,original_sha256=expected,arrays_sha256=sha(dest),metadata_transport=transport)
 results.append(record);print('MATRIX_END',key,record['packed_bytes'],record['tables'],record['standard_reencoded_raw_SHA_matches_original'],flush=True)
 del original,bundle,recovered,decoded,meta;gc.collect()
assert all(sha(p)==v for p,v in dependencies.items());result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_EXISTING_D_ALL_JSON_STRUCTURAL_MULTIPLICITY_ROUNDTRIP_PROFILE_ONLY','original_bytes':sum(x['original_bytes'] for x in results),'packed_bytes':sum(x['packed_bytes'] for x in results),'files':len(results),'results':results,'seconds':time.monotonic()-start,'dependency_sha256':dependencies,'original_marker_sha256':marker_hash,'H_real_data_profile_complete':False,'source_best_metrics_generated':False,'M_production_admitted':False,'namespace_reader_atomic_publication_and_peakRSS_admission_complete':False,'source_originals_mutated':False,'scope':'Existing accepted D LAST normal/closure/RQ4 schemas only. Exact integers, present-zero/absent keys and key order plus all residual JSON values retained. File probe publication is experimental; production bundle/resume/NPZ loader qualification remains separate.'};report=E/'existing_D_multiplicity_profile.json';assert not report.exists();report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('ALL_MATRIX_PROFILE_COMPLETE',result['original_bytes'],result['packed_bytes'],result['seconds'],flush=True)
