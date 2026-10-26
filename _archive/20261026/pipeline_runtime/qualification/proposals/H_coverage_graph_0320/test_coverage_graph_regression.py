import copy, hashlib, importlib, importlib.util, json, os, sys
from pathlib import Path
import pandas as pd
import pytest
import reliability_v2
if os.environ.get('COVERAGE_UNDER_TEST'):
    spec=importlib.util.spec_from_file_location('reliability_v2.coverage_h',os.environ['COVERAGE_UNDER_TEST']);cov=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cov;spec.loader.exec_module(cov);reliability_v2.coverage_h=cov
from reliability_v2 import coverage_h as cov, geometry_h, data_h
spec=importlib.util.spec_from_file_location('coverage_original_test_helpers',str(Path(os.environ['V2_CODE'])/'tests/test_coverage_h.py'));helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
FIELDS=('video_uid','role','method_id','label','domain','processing_version','mother_id','mother_rgb_sha256','rgb_sha256','mother_frame_index','mother_timestamp_seconds','image_path','sample_id','group_id')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True)+'\n');return {'path':str(p),'sha256':digest(p)}
def fixture(tmp_path,monkeypatch,bridge=False):
 c=[helpers.video(100,'TARGET_TEST',dataset='DFD'),helpers.video(101,'TARGET_TEST',dataset='DFD')];m=[helpers.result(v) for v in c]
 if bridge:
  c.append(helpers.video(102,'SOURCE_UNUSED',dataset='DFD'));m.append(helpers.result(c[-1]));m[2]['frames'][0]['mother_rgb_sha256']=m[0]['frames'][0]['mother_rgb_sha256'];m[2]['frames'][1]['mother_rgb_sha256']=m[1]['frames'][1]['mother_rgb_sha256']
 else:m[1]['frames'][0]['mother_rgb_sha256']=m[0]['frames'][0]['mother_rgb_sha256']
 closed=copy.deepcopy(m);audit,quarantine=geometry_h.h_content_audit(closed);out=tmp_path/'data/H';paths=[];inventory=hashlib.sha256()
 for i,r in enumerate(m):
  p=out/f'processed/{i}.json';item=write(p,r);paths.append(str(p));inventory.update(json.dumps([str(p),item['sha256']]).encode()+b'\n')
 identities={'candidates':write(out/'H_candidates_all.json',c),'processed_paths':write(out/'H_processed_manifest_paths.json',paths),'content_audit':write(out/'H_content_source_audit.json',audit)};tables={}
 for key in cov.TABLE_KEYS:
  parts=key.split('/');role=parts[-1].upper();version=parts[-2];common=len(parts)==3 or role in ('TRAIN','DEV');rows=[]
  for r in closed:
   v=r['video']
   if v['role']!=role or v['group_id'] in quarantine or not (r['common_B_AB'] if common else r['eligible']['B']):continue
   for i,fr in enumerate(r['frames']):
    mid=f'{v["video_uid"]}/mother/{i:03d}/{fr["mother_rgb_sha256"]}';rows.append(dict(zip(FIELDS,(v['video_uid'],role,v['method'],v['label'],v['domain'],version,mid,fr['mother_rgb_sha256'],fr[version]['rgb_sha256'],fr['mother_index'],fr['mother_timestamp'],fr[version]['path'],mid,v['group_id']))))
  p=out/f'{key}.parquet';p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows,columns=FIELDS).to_parquet(p,index=False);tables[key]={'path':str(p),'sha256':digest(p),'rows':len(rows),'videos':len({r['video_uid'] for r in rows})}
 receipt={'identities':identities,'tables':tables,'candidate_videos':len(c),'candidate_manifest_inventory_sha256':inventory.hexdigest(),'artifact_inventory_sha256':'synthetic-not-pixels','independent_review':write(tmp_path/'review.json',{'status':'PASS_SYNTHETIC'}),'independence_scope':'synthetic proxy scope','validation_policy':{'default':'SHALLOW_METADATA_BINDING'}}
 write(out/'freeze.json',receipt);monkeypatch.setattr(data_h,'validate_h_freeze',lambda root:copy.deepcopy(receipt));return receipt,paths,c,m

def test_exact_rgb_closure_is_replayed_before_active_table_validation(tmp_path,monkeypatch):
 receipt,paths,c,m=fixture(tmp_path,monkeypatch);before={str(p):digest(p) for p in tmp_path.rglob('*') if p.is_file()};report=cov.report_coverage(tmp_path)
 assert report['active_tables']['B/TARGET_TEST']['natural_videos']==2
 assert report['summary']['candidate_videos']==2 and not report['target_performance_used']
 assert before=={str(p):digest(p) for p in tmp_path.rglob('*') if p.is_file()}

def test_source_unused_rgb_bridge_is_included_without_becoming_active(tmp_path,monkeypatch):
 receipt,paths,c,m=fixture(tmp_path,monkeypatch,bridge=True);report=cov.report_coverage(tmp_path)
 assert report['summary']['candidate_videos']==3 and report['summary']['administrative_exclusions']['source_unused_videos']==1
 assert report['active_tables']['B/TARGET_TEST']['natural_videos']==2

def test_hash_bound_but_wrong_content_graph_is_rejected(tmp_path,monkeypatch):
 receipt,paths,c,m=fixture(tmp_path,monkeypatch);p=Path(receipt['identities']['content_audit']['path']);content=json.loads(p.read_text());content['rgb_group_edges']+=1;receipt['identities']['content_audit']=write(p,content);write(tmp_path/'data/H/freeze.json',receipt)
 with pytest.raises(ValueError,match='content graph'):cov.report_coverage(tmp_path)

def test_original_candidate_manifest_identity_is_checked_before_closure(tmp_path,monkeypatch):
 receipt,paths,c,m=fixture(tmp_path,monkeypatch);c[1]['group_id']=c[0]['group_id'];receipt['identities']['candidates']=write(receipt['identities']['candidates']['path'],c);write(tmp_path/'data/H/freeze.json',receipt)
 with pytest.raises(ValueError,match='Candidate/manifest identity mismatch'):cov.report_coverage(tmp_path)

def test_active_table_cannot_revert_to_original_unclosed_group(tmp_path,monkeypatch):
 receipt,paths,c,m=fixture(tmp_path,monkeypatch);bound=receipt['tables']['B/TARGET_TEST'];p=Path(bound['path']);table=pd.read_parquet(p);table.loc[table.video_uid.eq(c[1]['video_uid']),'group_id']=c[1]['group_id'];table.to_parquet(p,index=False);bound['sha256']=digest(p);write(tmp_path/'data/H/freeze.json',receipt)
 with pytest.raises(ValueError,match='Active table'):cov.report_coverage(tmp_path)
