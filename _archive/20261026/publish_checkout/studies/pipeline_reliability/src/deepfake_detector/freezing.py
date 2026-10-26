"""Approved FF source isolation and content-only handling of noise-generated EFS."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import pandas as pd
from .manifest import METHODS,select_eight,save_json,digest,source_ids_from_video
from .training import sampling_weights


def normalize_audit_manifest(df):
    q=df.copy()
    for name in ['label','image_path']:
        if name not in q:
            left,right=name+'_x',name+'_y'
            if left not in q or right not in q or not q[left].equals(q[right]):
                raise ValueError('Audit annotation join conflict: '+name)
            q[name]=q[left];q=q.drop(columns=[left,right])
    return q

def source_role_filter(df,roles):
    q=df.copy()
    def annotation(value):
        ids=json.loads(value);found={roles.get(i,'unknown') for i in ids}
        return next(iter(found)) if len(found)==1 else 'mixed' if found else 'unknown'
    q['upstream_source_role']=q.source_ids.map(annotation);q['experiment_role']='EXCLUDED'
    efs=q.method_type.eq('EFS')
    tr=q.domain.eq('ff')&q.released_role.eq('train')&(q.upstream_source_role.isin(['train','unknown'])|efs)
    te=q.domain.eq('ff')&q.released_role.eq('test')&(q.upstream_source_role.isin(['test','unknown'])|efs)
    cd=q.domain.eq('cdf')&q.released_role.eq('test')
    q.loc[tr,'experiment_role']='train';q.loc[te,'experiment_role']='ff_test';q.loc[cd,'experiment_role']='cdf_test'
    return q


def canonical_training_pool(df):
    associations=df.sort_values(['method_id','video_id','image_path'],kind='stable').reset_index(drop=True).copy()
    associations['sampling_contribution']=sampling_weights(associations)
    if associations.groupby('rgb_sha256').label.nunique().max()>1:raise ValueError('Same content has conflicting training labels')
    totals=associations.groupby('rgb_sha256').sampling_contribution.sum()
    members=associations.groupby('rgb_sha256').method_id.agg(lambda v:json.dumps(sorted(set(v))))
    pool=associations.drop_duplicates('rgb_sha256').drop(columns='sampling_contribution').copy()
    pool['sampling_mass']=pool.rgb_sha256.map(totals);pool['method_memberships']=pool.rgb_sha256.map(members)
    pool['sample_id']='ff/train/content/'+pool.rgb_sha256
    if not np.isclose(pool.sampling_mass.sum(),1.) or not np.isclose(pool.loc[pool.label.eq(0),'sampling_mass'].sum(),.5):raise ValueError('Canonical training mass mismatch')
    return pool.reset_index(drop=True),associations


class Components:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def attach_groups(frames,source_splits):
    comp=Components()
    # Original FF paired sources are known relationships, not claimed identities.
    for pairs in source_splits.values():
        for a,b in pairs:comp.union('ff/source/'+a,'ff/source/'+b)
    def nodes(row):
        ids=json.loads(row.source_ids)
        if row.domain=='ff' and ids:return ['ff/source/'+v for v in ids]
        if row.evaluation_unit=='image':return [f'{row.domain}/storage_surrogate/{row.method_id}/{row.storage_group_id}']
        return [f'{row.domain}/video_surrogate/{row.method_id}/{row.video_id}']
    keys=['domain','method_id','evaluation_unit','video_id','storage_group_id','source_ids']
    units=frames.drop_duplicates(keys)[keys].copy();unit_nodes=units.apply(nodes,axis=1)
    units['group_node']=[v[0] for v in unit_nodes]
    for ids in unit_nodes:
        for other in ids[1:]:comp.union(ids[0],other)
    q=frames.merge(units,on=keys,how='left',validate='many_to_one')
    # Known identical pixels tie otherwise-surrogate groups across method labels.
    dup=q[q.rgb_sha256.duplicated(keep=False)][['rgb_sha256','group_node']].drop_duplicates()
    for _,g in dup.groupby('rgb_sha256',sort=False):
        values=list(g.group_node)
        for v in values[1:]:comp.union(values[0],v)
    q['group_id']=q.group_node.map(lambda x:hashlib.sha256(comp.find(x).encode()).hexdigest())
    q['group_basis']=np.where(q.source_ids.ne('[]'),'known_FF_source_relationship','declared_surrogate_with_known_content_links')
    if q.groupby('group_id').experiment_role.nunique().max()>1:raise ValueError('Known source/content component crosses experiment roles')
    return q.drop(columns='group_node')


def _selected(df):
    videos=df[df.evaluation_unit.eq('video_frame')]
    video_rows=select_eight(videos.to_dict('records'))
    vid=pd.DataFrame(video_rows,columns=df.columns)
    images=df[df.evaluation_unit.eq('image')].sort_values('image_path').drop_duplicates(['domain','method_id','rgb_sha256'])
    return pd.concat([vid,images],ignore_index=True)


def select_with_groups(df,splits):
    return _selected(attach_groups(df,splits))


def freeze(config,project,approval):
    project=Path(project);data=Path(config['artifact_root'])/'data';marker=data/'freeze_complete.json'
    if marker.exists():raise ValueError('A frozen split exists; do not overwrite scientific evidence')
    if approval.get('approved') is not True or not approval.get('user_message'):raise ValueError('Explicit approved source-isolation decision required')
    audit=json.loads((data/'audit_summary.json').read_text())
    if audit['invalid_rows'] or audit['method_count']!=31:raise ValueError('Unresolved full audit inputs')
    official=data/'official_verification/official_metadata'
    for path,sha in audit['source_indices_sha256'].items():
        if digest(path)!=sha or digest(official/Path(path).name)!=sha:raise ValueError('Official/local source index verification no longer matches')
    for file in ['train','val','test']:
        source=data/'source_splits'/f'{file}.json'
        records=json.loads((data/'source_splits/provenance.json').read_text())
        if digest(source)!=records[file]['sha256']:raise ValueError('Original FF source splits changed')
    roles=json.loads((data/'source_splits/source_roles.json').read_text())
    df=pd.read_parquet(data/'audited_manifest.parquet')
    df['source_ids']=df.apply(lambda r:json.dumps(source_ids_from_video(r.domain,r.method_type,r.video_id)),axis=1)
    df=source_role_filter(df,roles)
    df.to_parquet(data/'audited_manifest_corrected.parquet',index=False)
    testfull=df[df.experiment_role.isin(['ff_test','cdf_test'])&df.decode_ok.eq(1)]
    testhashes=set(testfull.rgb_sha256)
    trainfull=df[df.experiment_role.eq('train')&df.decode_ok.eq(1)&~df.rgb_sha256.isin(testhashes)]
    train=_selected(trainfull);tests=_selected(testfull)
    # Source-real original FF validation is outside released DF40 train/test JSON.
    cal=pd.read_parquet(data/'source_real_candidates.parquet');cal=cal[cal.decode_ok.eq(1)].copy()
    if cal.rgb_sha256.isin(set(df.rgb_sha256)).any():raise ValueError('Calibration content overlaps released data')
    for name,value in [('domain','ff'),('label',0),('method_id','real'),('method_type','real'),('evaluation_unit','video_frame'),('experiment_role','calibration'),('storage_group_id',''),('include',True),('exclusion_reason','')]:cal[name]=value
    cal['video_id']=cal.source_id;cal['source_ids']=cal.source_id.map(lambda v:json.dumps([v]));cal['frame_number']=cal.image_path.map(lambda p:int(Path(p).stem))
    cal['original_image_path']=cal.image_path;cal['index_path']=str(data/'source_splits/val.json')
    if any(roles.get(v)!='val' for v in cal.source_id):raise ValueError('Calibration is not original FF source validation')
    cal_all=cal.copy()
    cal=pd.DataFrame(select_eight(cal.to_dict('records')))
    selected=pd.concat([train,tests,cal],ignore_index=True)
    selected['sample_id']=selected.apply(lambda r:f'{r.domain}/{r.experiment_role}/{r.method_id}/'+hashlib.sha256(r.image_path.encode()).hexdigest(),axis=1)
    selected['include']=True;selected['exclusion_reason']=''
    splits={role:json.loads((data/'source_splits'/f'{role}.json').read_text()) for role in ['train','val','test']}
    grouped=attach_groups(pd.concat([trainfull,testfull,cal_all],ignore_index=True),splits)
    lookup_keys=['domain','method_id','image_path','experiment_role']
    lookup=grouped[lookup_keys+['group_id','group_basis']].drop_duplicates(lookup_keys)
    selected=selected.merge(lookup,on=lookup_keys,how='left',validate='one_to_one')
    if selected.group_id.isna().any():raise ValueError('Selected row lacks full-association component')
    expected={m for methods in METHODS.values() for m in methods}
    for role in ['train','ff_test','cdf_test']:
        if set(selected.loc[selected.experiment_role.eq(role)&selected.label.eq(1),'method_id'])!=expected:raise ValueError('Incomplete31method coverage after isolation')
    if selected.groupby('rgb_sha256').experiment_role.nunique().max()>1:raise ValueError('Selected content crosses experiment roles')
    train,associations=canonical_training_pool(selected[selected.experiment_role.eq('train')])
    paths={};train.to_parquet(data/'train.parquet',index=False);paths['train.parquet']=digest(data/'train.parquet')
    associations.to_parquet(data/'training_associations.parquet',index=False)
    for role in ['calibration','ff_test','cdf_test']:
        part=selected[selected.experiment_role.eq(role)].reset_index(drop=True);part.to_parquet(data/f'{role}.parquet',index=False);paths[f'{role}.parquet']=digest(data/f'{role}.parquet')
    selected.to_parquet(data/'selected_associations.parquet',index=False)
    changed=df[['image_path','domain','released_role','upstream_source_role','experiment_role']].copy()
    changed['selected_as_experiment_input']=changed.image_path.isin(set(selected.image_path))
    changed['disposition']=np.where(changed.experiment_role.eq('EXCLUDED'),'known source role mismatched; unknown source kept with released role',np.where(changed.selected_as_experiment_input,'selected association','content duplicate, retained-test conflict, fixed8 selection or insufficient distinctframes'))
    changed.to_parquet(data/'role_dispositions.parquet',index=False)
    counts=selected.groupby(['experiment_role','method_id','evaluation_unit']).agg(input_rows=('image_path','size'),video_or_storage_units=('video_id','nunique'),unique_RGB=('rgb_sha256','nunique')).reset_index()
    counts.to_csv(project/'reports/frozen_method_counts.csv',index=False)
    record={'status':'FROZEN','approval':approval,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':paths,
      'method_registry':METHODS,'method_count':31,'calibration_available':True,'fake_calibration_available':False,
      'source_calibration_scope':'real-only FPR5: known original FF source partitions and exact-content separation; EFS author describes generation from noise; no bucket identity or unknown audio/generation ancestry certification',
      'TPR95':'UNAVAILABLE_NO_INDEPENDENT_COMPLETE_FAKE_CALIBRATION','training_unique_RGB_inputs':len(train),'training_association_rows':len(associations),
      'training_membership_sha256':digest(data/'training_associations.parquet'),'role_dispositions_sha256':digest(data/'role_dispositions.parquet'),
      'original_indices_sha256':audit['source_indices_sha256'],'official_fixed_commit':'ce808160e0df31544b41c7b60b942849ba88bca8',
      'official_metadata_version':'author revised folder1rh-82Rn0pqQ7xzLDKBv9gdjy-sbxYC8-; no legacyval used',
      'local_archive_provenance_limit':'official index bytes match; author original image archive provenance not authenticated',
      'source_original_data_modified':False,'original_test_or_CDF_to_train_or_calibration':False,
      'grouping_limit':'KnownFFsource pairs + knownRGB links; EFS/CDF surrogate groups do not imply real identities or population independence'}
    config['data_policy']='APPROVED_OFFICIAL_ROLE_WITH_KNOWN_FF_SOURCE_ISOLATION_AND_LOCAL_CONTENT_DEDUP'
    config['source_calibration']='REAL_ONLY_FPR5_KNOWN_FF_SOURCE_CONTENT_SCOPE'
    config['numeric_policy']={'AMP':'float16','TF32':False,'cudnn_benchmark':False,'deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8','data_loader_workers':0}
    save_json(project/'configs/v1.1.json',config);record['config_sha256']=digest(project/'configs/v1.1.json')
    # Complete marker last: any earlier interruption leaves no executable freeze.
    save_json(marker,record);return record
