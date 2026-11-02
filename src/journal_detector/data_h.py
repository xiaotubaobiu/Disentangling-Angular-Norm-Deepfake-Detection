"""Portable H inventory and paired mother-frame tables for journal experiments."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
import math
import os
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry_h as geometry
from . import data_d

DATA_ROOT=Path('/Dataset/deepfake_detection/FaceForensics++')
DEFINITION_ROOT=Path(__file__).resolve().parents[2]/'configs/data_protocol'
METHODS=geometry.METHODS
TABLE_COLUMNS=['image_path','label','mother_id','sample_id','video_uid','video_id','method_id',
    'rgb_sha256','content_sha256','group_id','source_ids','domain','dataset','role',
    'mother_rgb_sha256','mother_frame_index','mother_timestamp_seconds','processing_version']


def source_partition(definition_root=DEFINITION_ROOT):
    root=Path(definition_root)
    provenance=json.loads((root/'source_splits_provenance.json').read_text())
    splits={}
    for name in ('train','val','test'):
        if geometry.sha256_file(root/f'{name}.json')!=provenance[name]['sha256']:
            raise ValueError('Official FF split hash mismatch: '+name)
        splits[name]=json.loads((root/f'{name}.json').read_text())
    roles,_=data_d.pair_partition(splits)
    result={}
    for pairs in splits.values():
        for pair in pairs:
            group='FFPP/pair/'+'_'.join(sorted(map(str,pair)))
            for sid in map(str,pair): result[sid]={'role':roles[sid],'group_id':group}
    return result


def build_inventory(data_root=DATA_ROOT,definition_root=DEFINITION_ROOT,expected_videos=8431):
    root=Path(data_root).resolve(); partition=source_partition(definition_root); records=[]
    for method,directory in [('real',root/'original_sequences/youtube/c23/videos'),
            *[(m,root/f'manipulated_sequences/{m}/c23/videos') for m in METHODS]]:
        for path in sorted(directory.glob('*.mp4')):
            ids=path.stem.split('_'); groups={partition[sid]['group_id'] for sid in ids if sid in partition}
            roles={partition[sid]['role'] for sid in ids if sid in partition}
            if len(groups)!=1 or len(roles)!=1 or any(sid not in partition for sid in ids):
                raise ValueError('Unknown or crossing FF++ source pair: '+str(path))
            role=next(iter(roles)); label=int(method!='real')
            if label and role=='source_cal': role='source_unused'
            records.append({'dataset':'FFPP','domain':'source','method_id':method,'label':label,
                'video_id':path.stem,'video_uid':f'FFPP/source/{method}/{path.stem}',
                'role':role,'group_id':next(iter(groups)),'source_ids':ids,
                'source_evidence':'official FF++ split pair and source filename','path':str(path)})
    real_root=root/'original_sequences/actors/c23/videos'
    real_ids={p.stem for p in real_root.glob('*.mp4')}
    dfd=[]
    for method,directory in [('real',real_root),
            ('DeepFakeDetection',root/'manipulated_sequences/DeepFakeDetection/c23/videos')]:
        for path in sorted(directory.glob('*.mp4')):
            label=int(method!='real'); parents=[]; scene='unknown/'+path.stem
            if not label:
                parents=[path.stem]
                if '__' in path.stem: scene=path.stem.split('__',1)[1]
            elif '__' in path.stem:
                actors,scene=path.stem.split('__',1); scene=scene.rsplit('__',1)[0]
                parents=[f'{actor}__{scene}' for actor in actors.split('_') if f'{actor}__{scene}' in real_ids]
            dfd.append({'dataset':'DFD','domain':'target','method_id':method,'label':label,
                'video_id':path.stem,'video_uid':f'DFD/target/{method}/{path.stem}','role':'target_test',
                'group_id':'DFD/filename_scene/'+scene,'source_ids':parents,'unknown_ancestors':True,
                'source_evidence':'filename scene/parent candidates; identity/ancestor not certified','path':str(path)})
    records.extend(dfd)
    if len({r['video_uid'] for r in records})!=len(records): raise ValueError('Duplicate H video UID')
    if expected_videos is not None and len(records)!=expected_videos:
        raise ValueError(f'H candidate video count {len(records)} != {expected_videos}')
    records.sort(key=lambda r:({'train':0,'dev':1,'source_cal':2,'source_test':3,
                              'target_test':4,'source_unused':5}[r['role']],r['method_id'],r['video_id']))
    return records


def weight_training_pool(frames):
    """Preserve logical mothers; RGB coincidences never collapse B/AB draws."""
    out=frames.copy().reset_index(drop=True)
    if out.empty or out.mother_id.duplicated().any(): raise ValueError('Training mother identities must be unique')
    if set(out.method_id)!={'real',*METHODS}: raise ValueError('Missing H training method')
    if not out.groupby('video_uid').size().eq(8).all(): raise ValueError('H training video must have eight mothers')
    if not out['label'].eq(out.method_id.ne('real').astype(int)).all(): raise ValueError('H labels inconsistent')
    videos=out[['method_id','video_uid']].drop_duplicates().groupby('method_id').size()
    out['sampling_mass']=[(.5 if method=='real' else .125)/int(videos[method])/8 for method in out.method_id]
    out['method_contributions']=[json.dumps({method:float(mass)},sort_keys=True)
                                  for method,mass in zip(out.method_id,out.sampling_mass)]
    out['method_memberships']=[json.dumps([m]) for m in out.method_id]
    out['sample_id']=out.mother_id
    if not np.isclose(out.sampling_mass.sum(),1): raise ValueError('H training mass mismatch')
    return out


def pair_training_pools(b,ab):
    if b.mother_id.duplicated().any() or ab.mother_id.duplicated().any() or set(b.mother_id)!=set(ab.mother_id):
        raise ValueError('B/AB training mother identities differ')
    b=weight_training_pool(b); ab=ab.set_index('mother_id',drop=False).loc[b.mother_id].reset_index(drop=True).copy()
    for column in ['video_uid','method_id','label']:
        if b[column].tolist()!=ab[column].tolist(): raise ValueError('B/AB mother metadata differ: '+column)
    for column in ['sampling_mass','method_contributions','method_memberships','sample_id']:
        ab[column]=b[column].to_numpy(copy=True)
    return b,ab


def build_tables(records,manifests):
    buffers=defaultdict(list)
    for record in records:
        role=record['role']; manifest=manifests.get(record['video_uid'])
        if role=='source_unused' or manifest is None: continue
        b_ok=manifest.get('eligible',{}).get('B',False)
        common=manifest.get('common_B_AB',False)
        for frame in manifest.get('frames',[]):
            metadata={key:record[key] for key in ['dataset','domain','role','label','video_id','video_uid','method_id','group_id']}
            mid=frame['mother_id']
            metadata.update(mother_id=mid,sample_id=mid,source_ids=json.dumps(record['source_ids']),
                mother_rgb_sha256=frame['mother_rgb_sha256'],mother_frame_index=frame['mother_frame_index'],
                mother_timestamp_seconds=frame['mother_timestamp_seconds'])
            for version in ['B','AB']:
                if (role in ('train','dev') and not common) or (version=='B' and not b_ok) or (version=='AB' and not common):
                    continue
                info=frame[version]
                row={**metadata,'image_path':info['path'],'rgb_sha256':info['rgb_sha256'],
                     'content_sha256':info['byte_sha256'],'processing_version':version}
                buffers[f'{version}/{role}'].append(row)
                if common: buffers[f'common/{version}/{role}'].append(row.copy())
    tables={}
    for role in ['train','dev','source_cal','source_test','target_test']:
        for version in ['B','AB']:
            for prefix in ['', 'common/']:
                key=f'{prefix}{version}/{role}'
                tables[key]=pd.DataFrame(buffers[key],columns=TABLE_COLUMNS).sort_values(
                    ['video_uid','mother_frame_index','mother_id'],kind='stable').reset_index(drop=True)
    if not tables['B/train'].empty:
        tables['B/train'],tables['AB/train']=pair_training_pools(tables['B/train'],tables['AB/train'])
        tables['common/B/train']=tables['B/train'].copy(); tables['common/AB/train']=tables['AB/train'].copy()
    return tables


def coverage(records,manifests):
    rows=[]
    for record in records:
        manifest=manifests.get(record['video_uid'],{})
        reasons=[]
        if not manifest: reasons=['PENDING_PREPROCESSING']
        elif manifest.get('failure'): reasons=[manifest['failure']]
        for version,failures in manifest.get('version_failures',{}).items():
            reasons.extend(version+':'+reason for reason in failures)
        row={key:record[key] for key in ('video_uid','role','dataset','method_id','label')}
        row.update(processed=manifest.get('status')=='COMPLETE',decode_ok=bool(manifest.get('frames')),
            B=bool(manifest.get('eligible',{}).get('B')),AB=bool(manifest.get('eligible',{}).get('AB')),
            A=bool(manifest.get('eligible',{}).get('A')),common=bool(manifest.get('common_B_AB')),
            failure=';'.join(reasons),processing_seconds=manifest.get('processing_seconds',0.))
        rows.append(row)
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['role','method_id'],dropna=False).agg(candidate_videos=('video_uid','size'),
        processed=('processed','sum'),decode_ok=('decode_ok','sum'),A=('A','sum'),B=('B','sum'),
        AB=('AB','sum'),common=('common','sum')).reset_index()
    return frame,summary


def source_graph_audit(records,audits,manifests=None):
    """All candidate nodes/known pairs and selected mother/output exact RGB edges.

    Unused validation fakes contribute edges but never active-role counts.
    Any component touching multiple active roles is excluded in full.
    """
    comp=data_d.Components(); by_uid={r['video_uid']:r for r in records}; first_rgb={}; first_video={}
    for row in records:
        uid=row['video_uid']; comp.union('video/'+uid,'group/'+row['group_id'])
        audit=audits.get(uid)
        if audit is None: raise ValueError('Missing mother audit for candidate video: '+uid)
        raw=audit.get('source_sha256')
        if raw:
            if raw in first_video: comp.union('video/'+uid,'video/'+first_video[raw])
            else: first_video[raw]=uid
        hashes=[f['rgb_sha256'] for f in audit.get('selected_frames',[])]
        if manifests and uid in manifests:
            for frame in manifests[uid].get('frames',[]):
                hashes.extend(frame[v]['rgb_sha256'] for v in ('A','B','AB') if frame.get(v,{}).get('ok'))
        for digest in hashes:
            if digest in first_rgb: comp.union('video/'+uid,'video/'+first_rgb[digest])
            else: first_rgb[digest]=uid
    members=defaultdict(list)
    for uid in by_uid: members[comp.find('video/'+uid)].append(uid)
    quarantine=[]; conflicts=[]; components={}
    for root,uids in sorted(members.items()):
        roles=sorted({by_uid[uid]['role'] for uid in uids if by_uid[uid]['role']!='source_unused'})
        component_id='H/component/'+root
        for uid in uids: components[uid]=component_id
        if len(roles)>1:
            quarantine.extend(uids); conflicts.append({'component_id':component_id,'roles':roles,'video_uids':sorted(uids)})
    return {'schema':'JOURNAL_H_ALL_CANDIDATE_VIDEO_SELECTED_MOTHER_GRAPH_V1',
        'candidate_videos':len(records),'audited_videos':len(audits),
        'pixel_scope':'every selected mother of every candidate video; available A/B/AB RGB; unselected frames not audited',
        'source_byte_hash_scope':'full video bytes','known_source_scope':'official FF++ source-pair edges; DFD filename proxies only',
        'source_unused_scope':'nodes and exactRGB/byte/known-pair edges included; excluded from active role conflicts',
        'quarantined_video_uids':sorted(quarantine),'conflicts':conflicts,'components':components}


def write_table_freeze(artifact_root,tables,scope,metadata=None,manifest_paths=None):
    """Atomically write a complete cohort snapshot with exact table identities."""
    root=Path(artifact_root).resolve(); directory=root/'data/H'
    name='train_dev_freeze.json' if scope=='SOURCE_TRAIN_DEV_CAL' else 'freeze.json'
    path=directory/name
    if path.exists():
        existing=validate_h_freeze(root,source_only=scope=='SOURCE_TRAIN_DEV_CAL')
        for key,frame in tables.items():
            if key in existing['tables']:
                old=pd.read_parquet(directory/(key+'.parquet'))
                if not old.equals(frame): raise ValueError('Refusing to change frozen H cohort: '+key)
        return existing
    if not {'B/train','AB/train','B/dev','AB/dev','B/source_cal'}<=set(tables):
        raise ValueError('Missing H source tables')
    for key in ('B/train','AB/train','B/dev','AB/dev','B/source_cal'):
        if tables[key].empty: raise ValueError('Empty required H source table: '+key)
    b,ab=pair_training_pools(tables['B/train'],tables['AB/train'])
    if b.mother_id.tolist()!=ab.mother_id.tolist(): raise ValueError('H mother pairing differs')
    if tables['B/dev'].mother_id.tolist()!=tables['AB/dev'].mother_id.tolist():
        raise ValueError('H DEV mother pairing differs')
    identities={}
    # A later full freeze may add test tables but may never modify source tables.
    source_path=directory/'train_dev_freeze.json'
    source=json.loads(source_path.read_text()) if source_path.exists() else None
    for key,frame in sorted(tables.items()):
        target=directory/(key+'.parquet'); target.parent.mkdir(parents=True,exist_ok=True)
        if source and key in source['tables']:
            if not pd.read_parquet(target).equals(frame): raise ValueError('Full H audit changed previously frozen source cohort: '+key)
        else:
            temporary=target.with_suffix('.parquet.tmp'); frame.to_parquet(temporary,index=False); os.replace(temporary,target)
        identities[key]={'path':str(target),'sha256':geometry.sha256_file(target),
            'rows':len(frame),'videos':int(frame.video_uid.nunique()),
            'label_videos':{str(label):int(q.video_uid.nunique()) for label,q in frame.groupby('label')}}
    processed={}
    for uid,item in sorted((manifest_paths or {}).items()):
        item=Path(item); processed[uid]={'path':str(item),'sha256':geometry.sha256_file(item)}
    receipt={'schema':'JOURNAL_H_DATA_FREEZE_V1','status':'FROZEN','scope':scope,
        'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':geometry.protocol_hash(),
        'geometry_code_sha256':geometry.sha256_file(geometry.__file__),
        'data_code_sha256':geometry.sha256_file(__file__),'tables':identities,
        'manifest_sha256':{key+'.parquet':value['sha256'] for key,value in identities.items()},
        'processed_manifests':processed,'draw_count':128*math.ceil(len(b)/128),
        'source_selection_table':'B/dev','training_identity':'ordered logical mother_id, shared across B/AB',
        'metadata':metadata or {}}
    if source: receipt['train_dev_freeze_sha256']=geometry.sha256_file(source_path)
    geometry.write_json(path,receipt)
    return receipt


def validate_h_freeze(artifact_root,source_only=False,deep=False):
    """Validate source readiness without waiting for target preprocessing.

    Default checks frozen table, manifest and code bytes. ``deep`` also checks
    every PNG/mask file and decoded RGB; normal consumers verify bytes they read.
    """
    directory=Path(artifact_root).resolve()/'data/H'
    name='train_dev_freeze.json' if source_only else 'freeze.json'
    freeze=json.loads((directory/name).read_text())
    if freeze.get('status')!='FROZEN': raise ValueError('H data not frozen')
    if freeze.get('protocol_sha256')!=geometry.protocol_hash(): raise ValueError('H geometry protocol changed')
    if freeze.get('geometry_code_sha256')!=geometry.sha256_file(geometry.__file__):
        raise ValueError('H geometry code changed')
    if freeze.get('data_code_sha256')!=geometry.sha256_file(__file__): raise ValueError('H data code changed')
    for key,item in freeze['tables'].items():
        path=directory/(key+'.parquet')
        if not path.is_file() or geometry.sha256_file(path)!=item['sha256']:
            raise ValueError('frozen H table changed: '+key)
    for uid,item in freeze.get('processed_manifests',{}).items():
        path=Path(item['path'])
        if not path.is_file() or geometry.sha256_file(path)!=item['sha256']:
            raise ValueError('frozen H video manifest changed: '+uid)
        if deep:
            manifest=json.loads(path.read_text())
            for frame in manifest.get('frames',[]): geometry.verify_frame_artifacts(frame,deep=True)
    for filename,expected in freeze.get('metadata',{}).get('evidence_files',{}).items():
        if not Path(filename).is_file() or geometry.sha256_file(filename)!=expected:
            raise ValueError('frozen H audit evidence changed: '+filename)
    if 'train_dev_freeze_sha256' in freeze and geometry.sha256_file(directory/'train_dev_freeze.json')!=freeze['train_dev_freeze_sha256']:
        raise ValueError('H source freeze changed')
    return freeze
