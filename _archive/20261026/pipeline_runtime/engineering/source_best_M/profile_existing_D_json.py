from pathlib import Path
import json,hashlib,gzip,datetime,shutil,time
from reliability_v2.sensitivity_io import pack_existing
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M'
source=R/'analysis/D/D_formal_clip_vit_b16_seed7';out=E/'storage_profiles/D_clip_seed7_existing_last';assert not out.exists();out.mkdir(parents=True)
marker=source/'complete.json';marker_bytes=marker.read_bytes();m=json.loads(marker_bytes);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=sorted(p for p in source.rglob('*.json') if p!=marker);assert all(str(p.relative_to(source)) in m['artifact_sha256'] for p in files)
total=sum(p.stat().st_size for p in files);assert total<8*1024**3
budget=json.loads((R/'reports/full_capacity_check_close_20261025T0340.json').read_text());bound=total*1.001+len(files)*1024**2;assert shutil.disk_usage(R).free-budget['normal_required_bytes']>bound
records=[];start=time.monotonic()
for p in files:
    key=str(p.relative_to(source));expected=m['artifact_sha256'][key];destination=out/(key+'.gz');print('PACK_BEGIN',key,p.stat().st_size,flush=True)
    receipt=pack_existing(p,destination,expected_source_sha256=expected)
    # Independent streaming gzip decode, retaining exact original byte identity.
    h=hashlib.sha256();n=0
    with gzip.open(destination,'rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block);n+=len(block)
    assert h.hexdigest()==expected==receipt['uncompressed_sha256'] and n==p.stat().st_size==receipt['uncompressed_bytes']
    assert sha(destination)==receipt['compressed_sha256']
    records.append(dict(relative_source=key,source_expected_sha256=expected,output=str(destination),receipt=receipt,independent_uncompressed_SHA_and_size_match=True))
    print('PACK_END',key,receipt['compressed_bytes'],flush=True)
assert marker.read_bytes()==marker_bytes
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_EXISTING_ACCEPTED_D_LAST_ALL_JSON_BYTE_ROUNDTRIP_STORAGE_PROFILE_ONLY','source_run':'D_formal_clip_vit_b16_seed7','marker_sha256':hashlib.sha256(marker_bytes).hexdigest(),'json_files':len(files),'original_bytes':total,'compressed_bytes':sum(x['receipt']['compressed_bytes'] for x in records),'seconds':time.monotonic()-start,'records':records,'all_raw_bytes_and_committed_input_SHA_retained':True,'new_scientific_metrics_or_bootstrap_generated':False,'source_best_inference_started':False,'M_production_admitted':False,'H_real_schema_profile_complete':False,'source_originals_moved_or_deleted':False,'scope':'All existing JSON payload schemas of one accepted D LAST model only. Byte equality preserves every stored draw/metric/NA/string/metadata. This is neither new source-best results nor a bound on future source-best compression or H schemas; no compact budget credit is admitted.'}
report=E/'existing_D_json_lossless_profile.json';assert not report.exists();report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('PROFILE_COMPLETE',result['original_bytes'],result['compressed_bytes'],result['seconds'],flush=True)
