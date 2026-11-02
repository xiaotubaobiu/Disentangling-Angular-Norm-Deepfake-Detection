"""Content and source evidence only; never select or migrate experiment roles."""
from pathlib import Path
import datetime,json,sqlite3
from itertools import combinations
import pandas as pd
from .manifest import save_json,source_ids_from_video,digest


def method_content_counts(df):
    rows=[]
    for (domain,method),q in df.groupby(['domain','method_id'],sort=True):
        valid=q[q.decode_ok.fillna(0).eq(1)]
        row=dict(domain=domain,method_id=method,indexed_paths=q.image_path.nunique(),
                 pending_paths=q.loc[q.decode_ok.isna(),'image_path'].nunique(),
                 invalid_paths=q.loc[q.decode_ok.eq(0),'image_path'].nunique())
        cross={}
        for col,name in [('content_sha256','byte'),('rgb_sha256','RGB')]:
            stats=valid.groupby(col).released_role.nunique()
            cross[name]=set(stats.index[stats>1])
            row[name+'_cross_role_hashes']=len(cross[name])
            row[name+'_cross_role_paths']=valid.loc[valid[col].isin(cross[name]),'image_path'].nunique()
        # A pixel collision counted here has no cross-role byte proof in that group.
        byte_proven_pixels=set(valid.loc[valid.content_sha256.isin(cross['byte']),'rgb_sha256'])
        row['RGB_cross_role_hashes_without_byte_copy']=len(cross['RGB']-byte_proven_pixels)
        for role in ['train','test']:
            v=valid[valid.released_role.eq(role)]
            row['valid_'+role+'_images']=v.image_path.nunique()
            row['unique_'+role+'_RGB_images']=v.rgb_sha256.nunique()
            video=v[v.evaluation_unit.eq('video_frame')]
            count=video.groupby('video_id').rgb_sha256.nunique()
            row['eligible_'+role+'_videos']=int((count>=8).sum())
            row['ineligible_'+role+'_videos']=int((count<8).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_counts(df,source_roles):
    """Unapproved count scenario: filter known source roles before test priority.

    EFS image counts are explicitly conditional on accepting unknown ancestry.
    Unknown FS/FR inputs are reported separately, never certified or reassigned.
    """
    keys=['domain','method_type','video_id','storage_group_id']
    units=df.drop_duplicates(keys)[keys].copy()
    def role(r):
        ids=source_ids_from_video(r.domain,r.method_type,r.video_id)
        found={source_roles.get(i,'unknown') for i in ids}
        return next(iter(found)) if len(found)==1 else 'mixed' if found else 'unknown'
    units['source_annotation']=units.apply(role,axis=1)
    q=df.merge(units,on=keys,how='left',validate='many_to_one')
    q=q[q.decode_ok.fillna(0).eq(1)].copy()
    efs=q.method_type.eq('EFS')
    tested=q[q.released_role.eq('test')&(q.domain.eq('cdf')|q.source_annotation.eq('test')|efs)]
    test_hashes=set(tested.rgb_sha256)
    trained=q[q.domain.eq('ff')&q.released_role.eq('train')&(q.source_annotation.eq('train')|efs)]
    trained=trained[~trained.rgb_sha256.isin(test_hashes)]
    rows=[]
    for method,raw in q[q.domain.eq('ff')].groupby('method_id',sort=True):
        a=trained[trained.method_id.eq(method)];b=tested[tested.domain.eq('ff')&tested.method_id.eq(method)]
        def video_count(pool):
            frames=pool[pool.evaluation_unit.eq('video_frame')]
            return int((frames.groupby('video_id').rgb_sha256.nunique()>=8).sum())
        def image_count(pool):return int(pool.loc[pool.evaluation_unit.eq('image'),'rgb_sha256'].nunique())
        unknown=raw[raw.source_annotation.eq('unknown')&raw.released_role.eq('train')]
        excluded=raw[raw.released_role.eq('test')&raw.source_annotation.isin(['train','val','mixed'])]
        rows.append({'method_id':method,'method_type':raw.method_type.iloc[0],
          'known_only_train_videos':video_count(a),'known_only_test_videos':video_count(b),
          'unknown_ancestry_conditional_train_images':image_count(a),'unknown_ancestry_conditional_test_images':image_count(b),
          'unknown_source_train_eligible_videos_not_in_known_only_counts':video_count(unknown),
          'published_test_non_test_source_videos_excluded':excluded.video_id.nunique(),
          'fake_calibration_units':0,'experiment_roles_changed':False})
    return pd.DataFrame(rows)


def cross_method_counts(df):
    valid=df[df.decode_ok.fillna(0).eq(1)];rows=[]
    for (domain,role),q in valid.groupby(['domain','released_role'],sort=True):
        by={m:{'byte':set(g.content_sha256),'RGB':set(g.rgb_sha256)} for m,g in q.groupby('method_id',sort=True)}
        for a,b in combinations(sorted(by),2):
            byte=len(by[a]['byte']&by[b]['byte']);rgb=len(by[a]['RGB']&by[b]['RGB'])
            if not byte and not rgb:continue
            rows.append({'domain':domain,'released_role':role,'method_a':a,'method_b':b,
              'shared_byte_hashes':byte,'shared_RGB_hashes':rgb,
              'unique_RGB_method_a':len(by[a]['RGB']),'unique_RGB_method_b':len(by[b]['RGB'])})
    return pd.DataFrame(rows,columns=['domain','released_role','method_a','method_b','shared_byte_hashes','shared_RGB_hashes','unique_RGB_method_a','unique_RGB_method_b'])


def write_evidence(config,root):
    root=Path(root);data=Path(config['artifact_root'])/'data'
    cols=['domain','released_role','method_id','method_type','evaluation_unit','video_id','storage_group_id','label','image_path']
    df=pd.read_parquet(data/'released_manifest.parquet',columns=cols)
    # Read a consistent committed snapshot while the CPU auditor may still run.
    db=sqlite3.connect(f'file:{data / "image_audit.sqlite"}?mode=ro',uri=True)
    checks=pd.read_sql_query('SELECT path AS image_path,sha AS content_sha256,rgb AS rgb_sha256,ok AS decode_ok FROM images',db);db.close()
    joined=df.merge(checks,on='image_path',how='left',validate='many_to_one')
    counts=method_content_counts(joined)
    counts.to_csv(root/'reports/content_evidence_counts.csv',index=False)
    cross_method_counts(joined).to_csv(root/'reports/cross_method_content_counts.csv',index=False)
    roles=json.loads((data/'source_splits/source_roles.json').read_text())
    candidates=candidate_counts(joined,roles)
    candidates.to_csv(root/'reports/conditional_candidate_counts.csv',index=False)
    valid=joined[joined.decode_ok.fillna(0).eq(1)]
    overall={}
    for col,name in [('content_sha256','byte'),('rgb_sha256','RGB')]:
        g=valid.groupby(col).agg(roles=('released_role','nunique'),domains=('domain','nunique'),labels=('label','nunique'),methods=('method_id','nunique'))
        overall[name]={k:int((g[k]>1).sum()) for k in ['roles','domains','labels','methods']}
    # Exact parser revision is applied only to a new annotation table.
    keys=['domain','released_role','method_id','method_type','evaluation_unit','video_id','storage_group_id']
    units=df.drop_duplicates(keys).copy()
    units['source_ids']=units.apply(lambda r:json.dumps(source_ids_from_video(r.domain,r.method_type,r.video_id)),axis=1)
    units.to_csv(root/'reports/exact_source_annotations.csv',index=False)
    cand=data/'source_real_candidates.parquet';candidate_overlap={}
    if cand.exists():
        c=pd.read_parquet(cand)
        for col,name in [('content_sha256','byte'),('rgb_sha256','RGB')]:
            if col not in c:continue
            candidate_overlap[name]={role:int(c[col].isin(set(valid.loc[valid.released_role.eq(role),col])).sum()) for role in ['train','test']}
    pending=int(joined.loc[joined.decode_ok.isna(),'image_path'].nunique())
    report={'status':'PARTIAL_COMMITTED_AUDIT_SNAPSHOT' if pending else 'COMPLETE_CONTENT_EVIDENCE_POLICY_NOT_FROZEN',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'manifest_sha256':digest(data/'released_manifest.parquet'),'indexed_unique_paths':df.image_path.nunique(),
        'examined_unique_paths':joined.loc[joined.decode_ok.notna(),'image_path'].nunique(),
        'pending_unique_paths':pending,'invalid_unique_paths':joined.loc[joined.decode_ok.eq(0),'image_path'].nunique(),
        'global_proven_collision_hash_counts':overall,'source_real_candidate_content_overlap_rows':candidate_overlap,
        'parser_revision':'full three-digit tokens; five-digit SadTalker prefixes unknown; EFS buckets never sources',
        'source_annotation_only':True,'data_roles_changed':False,'training_allowed':False,
        'candidate_count_scenario':'UNAPPROVED: original released TRAIN only, known original-train FS/FR; original released TEST only, known original-test FF FS/FR; remove retained-test RGB from train; >=8 distinct decoded frames; EFS unique pixels remain conditional/unknown; no fake calibration',
        'limits':'Content collisions prove byte/pixel reuse only. Source annotations prove only known declared FF links. EFS ancestry and audio drivers remain unknown. Partial counts are lower bounds; eligibility is not an approved split.'}
    save_json(root/'reports/content_evidence_summary.json',report)
    return report
