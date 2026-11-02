"""D FS9/FR12 manifests, rebuilt from original indices and full associations.

No target scores are read. Original datasets and audit artifacts are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd
from PIL import Image

FS = ['fsgan','faceswap','simswap','inswap','blendface','uniface','mobileswap','e4s','facedancer']
FR = ['fomm','facevid2vid','wav2lip','MRAA','one_shot_free','pirender','tpsm','lia','danet','sadtalker','mcnet','hyperreenact']
METHODS = FS + FR
OLD = Path('/home/z/project/deepfake_detector')
DATA_ROOT = Path('/Dataset/deepfake_detection/DF40_all')
DEFAULT_OUTPUT = Path('/home/z/workspace/deepfake_conference21_20261102')
SCHEMA = ['image_path','label','method_id','video_id','video_uid','group_id','domain','role','rgb_sha256','content_sha256','source_ids']
SEED = 20261023


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def video_uid(domain, method, video):
    return f'DF40/{domain}/{method}/{video}'


def source_ids(domain, video):
    if domain != 'ff': return []
    match = re.match(r'^(\d{3})(?=_|$)(?:_(\d{3})(?=_|$))?', video)
    return [v for v in match.groups() if v] if match else []


def pair_partition(splits, seed=SEED):
    """Order-invariant, explicit FF pair partition shared with H."""
    pairs = sorted(tuple(sorted(map(str, p))) for p in splits['train'])
    if len(pairs) != 360 or len(set(sum((list(p) for p in pairs), []))) != 720:
        raise ValueError('Expected 360 disjoint official FF train source pairs')
    selected = set(np.random.default_rng(seed).permutation(360)[:306].tolist())
    roles = {}; records = []
    for i, pair in enumerate(pairs):
        role = 'train' if i in selected else 'dev'
        records.append({'source_ids': list(pair), 'role': role})
        for value in pair: roles[value] = role
    for split, role in [('val','source_cal'),('test','source_test')]:
        for pair in splits.get(split, []):
            for value in pair:
                if str(value) in roles: raise ValueError('Official source partitions overlap')
                roles[str(value)] = role
    return roles, records


class Components:
    def __init__(self): self.parent = {}
    def find(self, value):
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def attach_components(frames, splits):
    """Build full source/video/RGB graph BEFORE fixed-eight selection."""
    comp = Components()
    for pairs in splits.values():
        for a, b in pairs: comp.union('ff/source/'+str(a), 'ff/source/'+str(b))
    units = frames[['video_uid','domain','source_ids']].drop_duplicates()
    if units.video_uid.duplicated().any(): raise ValueError('Inconsistent video source metadata')
    for row in units.itertuples(index=False):
        node = 'video/'+row.video_uid; comp.find(node)
        if row.domain == 'ff':
            for value in json.loads(row.source_ids): comp.union(node, 'ff/source/'+value)
    links = frames[['rgb_sha256','video_uid']].drop_duplicates()
    links = links[links.rgb_sha256.duplicated(keep=False)]
    for _, group in links.groupby('rgb_sha256', sort=False):
        ids = group.video_uid.tolist()
        for value in ids[1:]: comp.union('video/'+ids[0], 'video/'+value)
    lookup = {uid: hashlib.sha256(comp.find('video/'+uid).encode()).hexdigest() for uid in units.video_uid}
    out = frames.copy(); out['group_id'] = out.video_uid.map(lookup)
    out['group_basis'] = np.where(out.domain.eq('ff') & out.source_ids.ne('[]'),
        'known_FF_filename_source_pair_plus_full_RGB_component', 'video_surrogate_plus_full_RGB_component')
    return out


def assign_roles(frames, roles):
    out = frames.copy(); assigned = []; reasons = []; upstream = []
    for row in out.itertuples(index=False):
        ids = json.loads(row.source_ids); found = {roles.get(value, 'unknown') for value in ids}
        role = next(iter(found)) if len(found) == 1 else 'mixed' if found else 'unknown'
        upstream.append(role)
        if row.domain == 'cdf' and row.released_role == 'test':
            assigned.append('target_full'); reasons.append('')
        elif row.domain == 'ff' and not ids:
            assigned.append('excluded'); reasons.append('unknown_FF_source_quarantine')
        elif row.domain == 'ff' and row.released_role == 'train' and role in ['train','dev']:
            assigned.append(role); reasons.append('')
        elif row.domain == 'ff' and row.released_role == 'test' and role == 'source_test':
            assigned.append(role); reasons.append('')
        elif row.domain == 'ff' and row.released_role == 'official_val_real' and role == 'source_cal':
            assigned.append(role); reasons.append('')
        else:
            assigned.append('excluded'); reasons.append('released_or_official_source_role_mismatch')
    out['upstream_source_role'] = upstream; out['role'] = assigned; out['exclusion_reason'] = reasons
    return out


def apply_dispositions(frames):
    out = frames.copy()
    if 'exclusion_reason' not in out: out['exclusion_reason'] = ''
    out['candidate_role'] = out.role
    active = out[out.role.ne('excluded')]
    cross = set(active.groupby('group_id').role.nunique().loc[lambda s:s>1].index)
    label_hashes = set(active.groupby('rgb_sha256').label.nunique().loc[lambda s:s>1].index)
    bad_label_groups = set(out.loc[out.rgb_sha256.isin(label_hashes),'group_id'])
    for groups, reason in [(cross,'cross_role_component'),(bad_label_groups,'RGB_label_conflict_component')]:
        mask = out.group_id.isin(groups)
        out.loc[mask,'role'] = 'excluded'
        out.loc[mask,'exclusion_reason'] = out.loc[mask,'exclusion_reason'].map(lambda value: ';'.join(filter(None,[value,reason])))
    return out


def select_eight(frames):
    """Numeric frame order, unique decoded RGB first, then linspace eight."""
    selected = []
    for _, group in frames.groupby('video_uid', sort=True):
        q = group[group.decode_ok.eq(1)].sort_values(['frame_number','image_path'], kind='stable').drop_duplicates('rgb_sha256')
        if len(q) >= 8: selected.extend(q.iloc[np.linspace(0,len(q)-1,8).astype(int)].index.tolist())
    return frames.loc[selected].copy().reset_index(drop=True)


def canonical_training_pool(frames):
    associations = frames.sort_values(['method_id','video_uid','image_path'], kind='stable').reset_index(drop=True).copy()
    expected = set(METHODS)
    if set(associations.loc[associations.label.eq(1),'method_id']) != expected: raise ValueError('Missing registered fake method')
    if associations.groupby('rgb_sha256').label.nunique().max() > 1: raise ValueError('Conflicting canonical labels')
    videos = associations[['method_id','video_uid']].drop_duplicates().groupby('method_id').size()
    frames_per_video = associations.groupby(['method_id','video_uid']).image_path.transform('size')
    associations['sampling_contribution'] = [(.5 if m=='real' else .5/21)/videos[m]/n for m,n in zip(associations.method_id,frames_per_video)]
    mass = associations.groupby('rgb_sha256').sampling_contribution.sum()
    contributions = associations.groupby(['rgb_sha256','method_id']).sampling_contribution.sum()
    maps = {rgb: json.dumps({method: float(value) for (_,method),value in series.items()}, sort_keys=True) for rgb,series in contributions.groupby(level=0)}
    pool = associations.drop_duplicates('rgb_sha256').drop(columns='sampling_contribution').copy()
    pool['sampling_mass'] = pool.rgb_sha256.map(mass)
    pool['method_contributions'] = pool.rgb_sha256.map(maps)
    pool['method_memberships'] = pool.method_contributions.map(lambda value:json.dumps(sorted(json.loads(value))))
    pool['sample_id'] = 'DF40/ff/train/RGB/'+pool.rgb_sha256
    if not np.isclose(pool.sampling_mass.sum(),1) or not np.isclose(pool.loc[pool.label.eq(0),'sampling_mass'].sum(),.5): raise ValueError('Training sampling mass mismatch')
    return pool.reset_index(drop=True), associations


def resolve_path(original, method, role, label):
    parts = Path(original.replace('\\','/')).parts
    if parts and parts[0] == 'deepfakes_detection_datasets': parts = parts[1:]
    if not parts or '..' in parts: raise ValueError('Unsafe index path')
    if not label: return DATA_ROOT.joinpath(*parts)
    if parts[0] == 'DF40_train': return DATA_ROOT.joinpath('DF40_train',method,*parts[2:])
    if parts[0] == 'DF40':
        if role == 'train':
            tail = parts[3:] if len(parts)>2 and parts[2]=='ff' else parts[2:]
            return DATA_ROOT.joinpath('DF40_train',method,*tail)
        return DATA_ROOT.joinpath('DF40_test',method,*parts[2:])
    raise ValueError('Unknown fake path prefix '+parts[0])


def rebuild_indices(old):
    audit = json.loads((old/'artifacts/data/audit_summary.json').read_text())
    rows = []; hashes = {}; real_provenance = []; unknown = []
    official = old/'artifacts/data/official_verification/official_metadata'
    for method in METHODS:
        for domain in ['ff','cdf']:
            path = DATA_ROOT/'dataset_json_update'/f'{method}_{domain}.json'
            expected = audit['source_indices_sha256'][str(path)]
            if digest(path) != expected or digest(official/path.name) != expected: raise ValueError('Source index hash mismatch: '+str(path))
            hashes[str(path)] = expected
            obj = json.loads(path.read_text())
            if len(obj)!=1: raise ValueError('Unexpected source index envelope')
            for labelkey, roles in next(iter(obj.values())).items():
                if not labelkey.endswith(('_Fake','_Real')): raise ValueError('Unknown index label')
                label = int(labelkey.endswith('_Fake'))
                for released, videos in roles.items():
                    if released not in ['train','test'] or (domain=='cdf' and released=='train' and videos): raise ValueError('Unexpected published role')
                    for original_id, record in videos.items():
                        if record['label'] != labelkey: raise ValueError('Index record label conflict')
                        for original in record['frames']:
                            resolved = resolve_path(original,method,released,label); video = resolved.parent.name
                            item = {'dataset':'DF40','domain':domain,'released_role':released,'label':label,'method_id':method if label else 'real',
                                'method_type':('FS' if method in FS else 'FR') if label else 'real','evaluation_unit':'video_frame',
                                'video_id':video,'video_uid':video_uid(domain,method if label else 'real',video),
                                'source_ids':json.dumps(source_ids(domain,video)),'image_path':str(resolved),'frame_number':int(resolved.stem) if resolved.stem.isdigit() else -1,
                                'original_image_path':original,'original_index_id':original_id,'index_path':str(path),
                                'driving_source_id':'unknown','relation_evidence':'FF_filename_source_tokens' if domain=='ff' and source_ids(domain,video) else 'unverified_ancestry',
                                'processing_version':'DF40_released_RGB_fixed8_numeric_v2'}
                            rows.append(item)
                            if not label: real_provenance.append({**{k:item[k] for k in ['image_path','domain','released_role','original_index_id','index_path']},'index_method':method})
                        if label and domain=='ff' and not source_ids(domain,video):
                            unknown.append({'method_id':method,'video_id':video,'original_index_id':original_id,'index_path':str(path),'record_fields':sorted(record),'resolution':'unresolved_no_explicit_parent_metadata_in_original_index'})
    frames = pd.DataFrame(rows).drop_duplicates(['domain','released_role','method_id','image_path']).reset_index(drop=True)
    return frames, pd.DataFrame(real_provenance), hashes, unknown


def inspect_image(path):
    before = Path(path).stat(); content = digest(path)
    with Image.open(path) as im:
        rgb = im.convert('RGB'); rgb.load(); width,height = rgb.size
        rgb_hash = hashlib.sha256(f'{width}x{height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
    after = Path(path).stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns): raise ValueError('Input changed during hash audit')
    return {'image_path':str(path),'content_sha256':content,'rgb_sha256':rgb_hash,'width':width,'height':height,'decode_ok':1,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}


def validate_audit(frames, old, out):
    """Verify cached hashes against old snapshot, sqlite metadata, every current stat."""
    snapshot_path = old/'artifacts/data/audited_manifest.parquet'
    snapshot = pd.read_parquet(snapshot_path, columns=['image_path','label','content_sha256','rgb_sha256','width','height','decode_ok','bytes','mtime_ns'])
    if snapshot.image_path.duplicated().any(): raise ValueError('Old full audit snapshot path identity is not unique')
    release = frames[frames.released_role.ne('official_val_real')]
    bound = release[['image_path','label']].merge(snapshot[['image_path','label']],on='image_path',how='left',suffixes=('','_audit'),validate='one_to_one')
    if bound.label_audit.isna().any() or not bound.label.eq(bound.label_audit).all(): raise ValueError('Original-index/audit label-path binding mismatch')
    cal = pd.read_parquet(old/'artifacts/data/source_real_candidates.parquet')
    checks = pd.concat([snapshot,cal.assign(label=0)[snapshot.columns]],ignore_index=True).drop_duplicates('image_path')
    checks = frames[['image_path']].drop_duplicates().merge(checks.drop(columns='label'),on='image_path',how='left',validate='one_to_one')
    if checks.rgb_sha256.isna().any() or not checks.decode_ok.eq(1).all(): raise ValueError('No successful original byte/RGB audit for candidate path')
    db_path = old/'artifacts/data/image_audit.sqlite'
    db = sqlite3.connect('file:'+str(db_path)+'?mode=ro',uri=True)
    cache = pd.read_sql_query('SELECT path AS image_path,sha AS content_sha256,rgb AS rgb_sha256,width,height,ok AS decode_ok,bytes,mtime_ns FROM images WHERE ok=1',db); db.close()
    # Source validation was audited independently, outside the released-path sqlite table.
    cache = pd.concat([cache,cal.assign(label=0)[checks.columns]],ignore_index=True).drop_duplicates('image_path')
    compare = checks.merge(cache,on='image_path',how='left',suffixes=('','_cache'),validate='one_to_one')
    for col in ['content_sha256','rgb_sha256','width','height','decode_ok','bytes','mtime_ns']:
        if not compare[col].eq(compare[col+'_cache']).all(): raise ValueError('Audit sqlite/snapshot metadata mismatch '+col)
    expected = {r.image_path:(r.bytes,r.mtime_ns) for r in checks.itertuples(index=False)}
    def stat(path):
        try:
            p=Path(path); s=p.stat(); return path,s.st_size,s.st_mtime_ns,p.is_file(),''
        except OSError as error: return path,0,0,False,str(error)
    fresh=[]; stale=[]; start=time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,(path,size,mtime,is_file,error) in enumerate(executor.map(stat,checks.image_path),1):
            if is_file and expected[path]==(size,mtime): fresh.append(path)
            else: stale.append(path)
            if i%100000==0:
                print(f'D audit-stat {i}/{len(checks)} stale={len(stale)} seconds={time.time()-start:.1f}',flush=True)
                save_json(out/'audit_progress.json',{'phase':'full_candidate_stat_check','checked':i,'total':len(checks),'stale':len(stale),'pid':os.getpid()})
    refreshed=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for record in executor.map(inspect_image,stale): refreshed.append(record)
    if refreshed:
        replacements=pd.DataFrame(refreshed); checks=checks[~checks.image_path.isin(stale)]
        checks=pd.concat([checks,replacements],ignore_index=True)
    checks['hash_reuse_scope']='cached_RGB_and_byte_digest_current_size_mtime_verified'
    checks.loc[checks.image_path.isin(stale),'hash_reuse_scope']='fresh_decoded_RGB_and_byte_hash'
    checks.to_parquet(out/'audit_metadata.parquet',index=False)
    receipt={'candidate_unique_paths':len(checks),'metadata_fresh_cached':len(fresh),'fresh_rehashed':len(refreshed),
        'hash_protocol':'SHA256(bytes); RGB=SHA256(ASCII(width x height :RGB:)+PIL.convert(RGB).tobytes())',
        'scope':'All candidate paths stat-verified against prior successful audit. Metadata equality is freshness evidence, not a cryptographic assertion that bytes never changed between audits.',
        'prior_manifest_sha256':digest(snapshot_path),'prior_audit_summary_sha256':digest(old/'artifacts/data/audit_summary.json'),
        'prior_audit_code_sha256':digest(old/'src/deepfake_detector/manifest.py'),'elapsed_seconds':time.time()-start}
    save_json(out/'audit_receipt.json',receipt)
    return frames.merge(checks,on='image_path',validate='many_to_one'),receipt


def cdf_proxy_split(selected, full, old, out):
    """Recheck historical real candidate assignment using full known RGB closure."""
    historic = old/'reports/cdf_real_calibration_20261022_v1'
    protocol=json.loads((historic/'protocol.json').read_text())
    split_path=historic/'video_split.csv'
    if digest(split_path)!=protocol['hashes']['video_split.csv']: raise ValueError('Historical CDF split hash mismatch')
    split=pd.read_csv(split_path)
    historic_cal=split[split.role.eq('calibration_pool') & split.label.eq(0)]
    cal_uids={video_uid('cdf','real',v) for v in historic_cal.video_id}
    target=selected[selected.role.eq('target_full')].copy()
    cal=target[target.video_uid.isin(cal_uids)].copy()
    if cal.video_uid.nunique()!=len(cal_uids): raise ValueError('Historical real calibration candidate no longer eligible')
    cal_groups=set(full.loc[full.video_uid.isin(cal_uids),'group_id'])
    def tokens(value): return set(re.findall(r'(?<![A-Za-z0-9])id\d+(?!\d)',value))
    cal_tokens=set().union(*(tokens(v) for v in cal.video_id))
    blocked_groups=set(cal_groups)
    # Expand known full-association groups touched by any conservative identity token.
    target_full=full[full.domain.eq('cdf')]
    proxy_uids=set(target_full.loc[target_full.video_id.map(lambda v:bool(tokens(v)&cal_tokens)),'video_uid'])
    blocked_groups.update(target_full.loc[target_full.video_uid.isin(proxy_uids),'group_id'])
    test=target[~target.group_id.isin(blocked_groups) & ~target.video_uid.isin(cal_uids)].copy()
    real_test_uids=set(split.loc[split.role.eq('heldout_test') & split.label.eq(0),'video_id'])
    test=test[test.label.eq(1)|test.video_id.isin(real_test_uids)]
    cal['role']='target_cal';test['role']='target_test'
    if set(cal.group_id)&set(test.group_id) or set(cal.rgb_sha256)&set(test.rgb_sha256): raise ValueError('CDF CAL/TEST known-content overlap')
    cal.to_parquet(out/'target_cal.parquet',index=False);test.to_parquet(out/'target_test.parquet',index=False)
    excluded=target[~target.video_uid.isin(set(cal.video_uid)|set(test.video_uid))].drop_duplicates('video_uid')
    excluded['exclusion_reason']='full_known_RGB_component_or_conservative_id_name_proxy_connected_to_entire_CAL_pool'
    excluded.to_parquet(out/'target_calibration_exclusions.parquet',index=False)
    evidence={'status':'PROXY_SCOPE_EMPIRICAL_ONLY','cal_real_videos':cal.video_uid.nunique(),'test_real_videos':test.loc[test.label.eq(0),'video_uid'].nunique(),
        'test_fake_videos':test.loc[test.label.eq(1),'video_uid'].nunique(),'excluded_videos':excluded.video_uid.nunique(),
        'historic_split_sha256':digest(split_path),'historic_protocol_sha256':digest(historic/'protocol.json'),
        'cal_candidate_pool':'same prior score-blind80 real candidates, requalified for current fixed8 and full association graph',
        'isolation':'video UID, all candidate RGB-connected components, conservative id-name token proxy for the entire CAL pool',
        'ancestor_independence_certified':False,'strict_NP_population_guarantee':False,
        'unresolved':'Complete generated-parent, donor/driver, audio and YouTube identity ancestry is not available. A filename is a proxy, not authenticated identity.'}
    save_json(out/'cdf_calibration_scope.json',evidence)
    return evidence


def build(output_root=DEFAULT_OUTPUT, old_root=OLD):
    start=time.time(); output_root=Path(output_root);old=Path(old_root);out=output_root/'data/D'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists(): raise ValueError('D is already frozen; refusing to overwrite scientific manifests')
    print('D build original42 indices',flush=True)
    frames,provenance,index_hashes,unknown=rebuild_indices(old)
    provenance.to_parquet(out/'real_index_provenance.parquet',index=False)
    save_json(out/'unknown_source_index_audit.json',unknown)
    splits={};split_hashes={};proof=json.loads((old/'artifacts/data/source_splits/provenance.json').read_text())
    for split in ['train','val','test']:
        path=old/'artifacts/data/source_splits'/f'{split}.json'
        if digest(path)!=proof[split]['sha256']: raise ValueError('Official FF split hash mismatch')
        split_hashes[split]=digest(path);splits[split]=json.loads(path.read_text())
    roles,records=pair_partition(splits)
    save_json(out/'source_pair_partition.json',{'seed':SEED,'algorithm':'lexicographically sorted internally sorted official train pairs; numpy.default_rng(seed).permutation(360) first306 TRAIN last54 DEV','numpy_version':np.__version__,'pairs':records,'source_roles':roles,'official_split_sha256':split_hashes})
    cal=pd.read_parquet(old/'artifacts/data/source_real_candidates.parquet')
    cal=cal[['image_path','source_id']].copy();cal['dataset']='DF40';cal['domain']='ff';cal['released_role']='official_val_real';cal['label']=0;cal['method_id']='real';cal['method_type']='real';cal['evaluation_unit']='video_frame'
    cal['video_id']=cal.source_id;cal['video_uid']=cal.video_id.map(lambda v:video_uid('ff','real',v));cal['source_ids']=cal.source_id.map(lambda v:json.dumps([v]));cal['frame_number']=cal.image_path.map(lambda v:int(Path(v).stem))
    cal['index_path']=str(old/'artifacts/data/source_splits/val.json');cal['original_image_path']=cal.image_path;cal['original_index_id']=cal.source_id;cal['driving_source_id']='unknown';cal['relation_evidence']='official_FF_val_real_source';cal['processing_version']='DF40_released_RGB_fixed8_numeric_v2'
    frames=pd.concat([frames,cal.drop(columns='source_id')],ignore_index=True)
    print(f'D rebuilt association rows={len(frames)} unique_paths={frames.image_path.nunique()}',flush=True)
    frames,audit=validate_audit(frames,old,out)
    frames=assign_roles(frames,roles)
    print('D full source/video/RGB graph',flush=True)
    frames=attach_components(frames,splits);frames=apply_dispositions(frames)
    frames['experiment_role']=frames.role
    frames.to_parquet(out/'full_associations.parquet',index=False)
    eligible=frames[frames.role.ne('excluded')].copy();selected=select_eight(eligible)
    missing=set(eligible.video_uid)-set(selected.video_uid)
    exclusions=frames[frames.role.eq('excluded')].copy()
    insufficient=eligible[eligible.video_uid.isin(missing)].copy();insufficient['exclusion_reason']='fewer_than8_distinct_valid_RGB_frames';insufficient['role']='excluded'
    exclusions=pd.concat([exclusions,insufficient],ignore_index=True)
    exclusions.to_parquet(out/'exclusions.parquet',index=False)
    for role in ['train','dev','source_test','target_full']:
        got=set(selected.loc[selected.role.eq(role)&selected.label.eq(1),'method_id'])
        if got!=set(METHODS): raise ValueError(f'Missing21method coverage in {role}: {set(METHODS)-got}')
    if selected.groupby('rgb_sha256').role.nunique().max()>1: raise ValueError('Selected RGB crosses roles')
    if selected.groupby('group_id').role.nunique().max()>1: raise ValueError('Selected full graph component crosses roles')
    if selected.groupby('video_uid').size().ne(8).any() or selected.groupby('video_uid').rgb_sha256.nunique().ne(8).any(): raise ValueError('Fixed8 video qualification failed')
    train,associations=canonical_training_pool(selected[selected.role.eq('train')])
    train.to_parquet(out/'train.parquet',index=False);associations.to_parquet(out/'training_associations.parquet',index=False)
    selected.to_parquet(out/'selected_associations.parquet',index=False)
    for role in ['dev','source_cal','source_test','target_full']:
        selected[selected.role.eq(role)].to_parquet(out/f'{role}.parquet',index=False)
    identity=selected.drop_duplicates('video_uid')[['dataset','domain','role','label','method_id','method_type','video_id','video_uid','group_id','source_ids','relation_evidence','group_basis']]
    identity.to_parquet(out/'video_identity.parquet',index=False)
    counts=selected.groupby(['role','method_id','label']).agg(rows=('image_path','size'),videos=('video_uid','nunique'),unique_RGB=('rgb_sha256','nunique'),groups=('group_id','nunique')).reset_index()
    counts.to_csv(out/'counts.csv',index=False)
    qualification=train.groupby('method_id',sort=True).head(2).copy()
    qualification.to_parquet(out/'qualification_source.parquet',index=False)
    cdf=cdf_proxy_split(selected,frames,old,out)
    masses=associations.groupby('method_id').sampling_contribution.sum().to_dict()
    exclusion_counts=exclusions.drop_duplicates('video_uid').groupby('exclusion_reason').size().to_dict()
    hashes={p.name:digest(p) for p in sorted(out.glob('*')) if p.is_file() and p.name not in ['freeze.json','audit_progress.json']}
    receipt={'status':'FROZEN','protocol':'D_FS9_FR12_full_association_rebuild_20261023_v2','utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
        'method_registry':{'FS9':FS,'FR12':FR},'method_count':21,'manifest_sha256':hashes,'original_indices_sha256':index_hashes,'official_split_sha256':split_hashes,
        'code_sha256':digest(__file__),'canonical_train_inputs':len(train),'training_association_rows':len(associations),'samples_per_epoch':math.ceil(len(train)/128)*128,
        'sampling_method_probability':masses,'source_pair_partition':{'train_pairs':306,'dev_pairs':54,'seed':SEED},
        'counts':counts.to_dict('records'),'exclusion_video_counts':exclusion_counts,'unresolved_source_index_videos':len(unknown),
        'cache_audit':audit,'cdf_calibration_scope':cdf,
        'independence_scope':'Known FF filename sources, official source pairs, and full candidate exact RGB components; unknown audio/driver ancestry remains unverified. CDF uses video/content and conservative identity-name proxies only.',
        'source_validation_and_test_disjoint':True,'target_scores_used':False,'original_data_modified':False,'elapsed_seconds':time.time()-start}
    save_json(out/'freeze.json',receipt)
    report_dir=output_root/'reports';report_dir.mkdir(parents=True,exist_ok=True)
    save_json(report_dir/'d_manifest_complete.json',{'status':'COMPLETE','freeze_json':str(out/'freeze.json'),'freeze_sha256':digest(out/'freeze.json'),'counts':counts.to_dict('records'),'exclusions':exclusion_counts,'elapsed_seconds':time.time()-start})
    print(json.dumps({'status':'FROZEN','canonical_train_inputs':len(train),'training_association_rows':len(associations),'source_counts':selected.groupby('role').video_uid.nunique().to_dict(),'cdf_scope':cdf,'seconds':time.time()-start}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--old-root',type=Path,default=OLD)
    args=parser.parse_args();build(args.output_root,args.old_root)
