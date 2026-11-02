"""D FS9/FR12 manifests, rebuilt from original indices and full associations.

No target scores are read. Raw data and checked-in split definitions are read-only.
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
import time

import numpy as np
import pandas as pd
from PIL import Image

FS = ['fsgan','faceswap','simswap','inswap','blendface','uniface','mobileswap','e4s','facedancer']
FR = ['fomm','facevid2vid','wav2lip','MRAA','one_shot_free','pirender','tpsm','lia','danet','sadtalker','mcnet','hyperreenact']
METHODS = FS + FR
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFINITION_ROOT = REPOSITORY_ROOT / 'configs/data_protocol'
DATA_ROOT = Path('/Dataset/deepfake_detection/DF40_all')
DEFAULT_OUTPUT = REPOSITORY_ROOT / 'outputs/journal_21methods'
SCHEMA = ['image_path','label','method_id','video_id','video_uid','group_id','domain','role','rgb_sha256','content_sha256','source_ids']
SEED = 20261023
CDF_DEVELOPMENT_POLICY = {
    'version': 'cdf_joint_relation_development_v1',
    'selection_seed': 20261102,
    'real_video_target': 20,
    'fake_videos_per_method_cap': 20,
    'calibration_video_target': 80,
    'seed_real_video_ids': [],
    'development_component_selection': 'smallest_feasible_fake_bearing_component_prefer_all_registered_methods',
    'development_scoring': 'score_capped_subset_quarantine_entire_touched_joint_component',
    'calibration_selection': 'whole_real_only_joint_components_historical_CAL_first_then_other_real_only',
    'test_selection': 'all_remaining_non_DEV_non_CAL_components_no_historical_TEST_restriction',
    'unusable_fake_development_fallback': 'real_only_development_and_explicit_not_usable_status',
}


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
    return attach_cdf_relations(out)


def cdf_name_tokens(value):
    """Conservative filename identity proxy; id1 must never match id10."""
    return sorted(set(re.findall(r'(?<![A-Za-z0-9])id\d+(?!\d)', str(value))))


def attach_cdf_relations(frames):
    """Joint closure of full known groups, explicit sources and CDF name proxies.

    The known group was formed using *all candidate RGB frames*, including frames
    never selected for fixed-eight scoring. New tokens encountered through any
    known-group edge participate in the same closure; there is no one-hop limit.
    This relation graph is separate from the authenticated-source/RGB group_id.
    """
    out = frames.copy()
    units = out.loc[out.domain.eq('cdf'), ['video_uid', 'video_id', 'group_id', 'source_ids']].drop_duplicates()
    if units.video_uid.duplicated().any():
        raise ValueError('Inconsistent CDF video relation metadata')
    comp = Components()
    token_map = {}
    for row in units.sort_values('video_uid').itertuples(index=False):
        node = 'video/' + row.video_uid
        comp.union(node, 'known_group/' + str(row.group_id))
        comp.union(node, 'cdf_name/' + row.video_id)
        tokens = cdf_name_tokens(row.video_id)
        token_map[row.video_uid] = json.dumps(tokens)
        for token in tokens:
            comp.union(node, 'cdf_identity_proxy/' + token)
        for source in json.loads(row.source_ids):
            comp.union(node, 'cdf_explicit_source/' + str(source))
    lookup = {uid: hashlib.sha256(comp.find('video/' + uid).encode()).hexdigest()
              for uid in units.video_uid}
    out['cdf_relation_group_id'] = out.video_uid.map(lookup).fillna('')
    out['cdf_name_tokens'] = out.video_uid.map(token_map).fillna('[]')
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


def resolve_path(original, method, role, label, data_root=DATA_ROOT):
    data_root = Path(data_root)
    parts = Path(original.replace('\\','/')).parts
    if parts and parts[0] == 'deepfakes_detection_datasets': parts = parts[1:]
    if not parts or '..' in parts or Path(original.replace('\\','/')).is_absolute(): raise ValueError('Unsafe index path')
    if not label: return data_root.joinpath(*parts)
    if parts[0] == 'DF40_train': return data_root.joinpath('DF40_train',method,*parts[2:])
    if parts[0] == 'DF40':
        if role == 'train':
            tail = parts[3:] if len(parts)>2 and parts[2]=='ff' else parts[2:]
            return data_root.joinpath('DF40_train',method,*tail)
        return data_root.joinpath('DF40_test',method,*parts[2:])
    raise ValueError('Unknown fake path prefix '+parts[0])


def load_definition(definition_root=DEFINITION_ROOT):
    """Read the small versioned experimental definitions, verifying every input."""
    root = Path(definition_root)
    definition = json.loads((root/'definition.json').read_text())
    required = {'train.json','val.json','test.json','source_splits_provenance.json',
                'indices_sha256.json','source_real_candidates.json','cdf_real_split.json'}
    if set(definition['files_sha256']) != required: raise ValueError('Incomplete data definition')
    for name, expected in definition['files_sha256'].items():
        if Path(name).name != name or digest(root/name) != expected:
            raise ValueError('Data definition hash mismatch: '+name)
    for name in ['indices_sha256','source_real_candidates','cdf_real_split']:
        definition[name] = json.loads((root/(name+'.json')).read_text())
    definition['splits'] = {name:json.loads((root/(name+'.json')).read_text()) for name in ['train','val','test']}
    definition['split_provenance'] = json.loads((root/'source_splits_provenance.json').read_text())
    policy_path = root/'cdf_development_policy.json'
    definition['cdf_development_policy'] = (json.loads(policy_path.read_text())
        if policy_path.exists() else dict(CDF_DEVELOPMENT_POLICY))
    for name, evidence in definition['split_provenance'].items():
        if digest(root/(name+'.json')) != evidence['sha256']: raise ValueError('Official FF split hash mismatch')
    if set(definition['indices_sha256']) != {f'{m}_{d}.json' for m in METHODS for d in ['ff','cdf']}:
        raise ValueError('Requires exactly 42 official indices')
    cdf = definition['cdf_real_split']
    if (len(set(cdf['calibration_pool'])) != 80 or len(set(cdf['heldout_test'])) != 97
            or set(cdf['calibration_pool']) & set(cdf['heldout_test'])):
        raise ValueError('Invalid fixed CDF real split')
    return definition


def rebuild_indices(data_root=DATA_ROOT, definition=None):
    definition = load_definition() if definition is None else definition
    data_root = Path(data_root)
    rows = []; hashes = {}; real_provenance = []; unknown = []
    for method in METHODS:
        for domain in ['ff','cdf']:
            path = data_root/'dataset_json_update'/f'{method}_{domain}.json'
            expected = definition['indices_sha256'][path.name]
            if digest(path) != expected: raise ValueError('Source index hash mismatch: '+str(path))
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
                            resolved = resolve_path(original,method,released,label,data_root); video = resolved.parent.name
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


def audit_images(frames, out, workers=4):
    """Fresh byte and decoded-RGB audit of every candidate, with bounded work queues."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if workers < 1: raise ValueError('Audit workers must be positive')
    paths = frames.image_path.drop_duplicates().tolist()
    checked = []; start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(paths), 4096):
            checked.extend(executor.map(inspect_image, paths[offset:offset+4096]))
            save_json(out/'audit_progress.json', {'phase':'fresh_full_byte_RGB_audit',
                'checked':len(checked),'total':len(paths),'pid':os.getpid()})
            if offset % 102400 == 0:
                print(f'D fresh audit {len(checked)}/{len(paths)} seconds={time.time()-start:.1f}',flush=True)
    checks = pd.DataFrame(checked)
    checks['hash_reuse_scope'] = 'fresh_decoded_RGB_and_byte_hash'
    checks.to_parquet(out/'audit_metadata.parquet',index=False)
    receipt = {'candidate_unique_paths':len(checks),'metadata_fresh_cached':0,'fresh_rehashed':len(checks),
        'hash_protocol':'SHA256(bytes); RGB=SHA256(ASCII(width x height :RGB:)+PIL.convert(RGB).tobytes())',
        'scope':'Every candidate is read and decoded again from raw data; no previous audit cache is used.',
        'elapsed_seconds':time.time()-start}
    save_json(out/'audit_receipt.json',receipt)
    return frames.merge(checks,on='image_path',validate='many_to_one'),receipt


def cdf_proxy_split(selected, full, definition, out):
    """Freeze score-blind DEV, ranking, CAL and TEST on the joint candidate graph.

    target_full remains the formal ranking filename for existing consumers. Its
    contents exclude the entire development closure. CAL/TEST are separate RQ4
    cohorts; ranking may contain the RQ4 calibration videos, as in the prior design.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    policy = {**CDF_DEVELOPMENT_POLICY, **definition.get('cdf_development_policy', {})}
    for key in ['real_video_target', 'fake_videos_per_method_cap', 'calibration_video_target']:
        if not isinstance(policy[key], int) or policy[key] < 0:
            raise ValueError('Invalid CDF development allocation budget: ' + key)
    split = definition['cdf_real_split']
    full = attach_cdf_relations(full) if 'cdf_relation_group_id' not in full else full
    candidate = full[full.domain.eq('cdf')].copy()
    relation_columns = ['cdf_relation_group_id', 'cdf_name_tokens']
    relation_map = candidate[['video_uid', *relation_columns]].drop_duplicates()
    target = selected[selected.role.eq('target_full')].drop(columns=relation_columns, errors='ignore')
    target = target.merge(relation_map, on='video_uid', validate='many_to_one')
    units = target.drop_duplicates('video_uid').copy()
    all_units = candidate.drop_duplicates('video_uid').copy()
    all_units['fixed8_qualified'] = all_units.video_uid.isin(set(units.video_uid))
    group_column = 'cdf_relation_group_id'
    fake_groups = set(all_units.loc[all_units.label.eq(1), group_column])
    historical_cal = set(split['calibration_pool'])
    historical_test = set(split['heldout_test'])
    seed = int(policy['selection_seed'])
    def order(value):
        return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    def ordered(frame):
        return frame.assign(_order=frame.video_uid.map(order)).sort_values(
            ['_order', 'video_uid'], kind='stable').drop(columns='_order')
    method_set = set(units.loc[units.label.eq(1), 'method_id'])
    group_units = {group: rows for group, rows in units.groupby(group_column, sort=True)}
    candidate_sizes = all_units.groupby(group_column).size().to_dict()
    dev_groups = set()
    seeded = set(policy.get('seed_real_video_ids', []))
    unknown_seeds = seeded - set(units.loc[units.label.eq(0), 'video_id'])
    if unknown_seeds:
        raise ValueError('Ineligible explicit CDF development seed real IDs: ' + str(sorted(unknown_seeds)))
    dev_groups.update(units.loc[units.label.eq(0) & units.video_id.isin(seeded), group_column])
    eligible_fake_components = []
    for group in sorted(fake_groups):
        component = group_units.get(group)
        if component is None or not component.label.eq(1).any():
            continue
        remaining = units[~units[group_column].isin(dev_groups | {group})]
        remaining_methods = set(remaining.loc[remaining.label.eq(1), 'method_id'])
        if remaining_methods != method_set or not remaining.label.eq(0).any():
            continue
        coverage = set(component.loc[component.label.eq(1), 'method_id'])
        eligible_fake_components.append((group, len(coverage), candidate_sizes[group]))
    if (policy['fake_videos_per_method_cap'] > 0
            and not units.loc[units[group_column].isin(dev_groups), 'label'].eq(1).any()
            and eligible_fake_components):
        # Prefer complete method coverage, then the smallest complete candidate
        # component. Eligibility is always recomputed after the full RGB audit.
        chosen = min(eligible_fake_components,
            key=lambda item: (item[1] != len(method_set), -item[1], item[2], order(item[0])))
        dev_groups.add(chosen[0])
    real_target = policy['real_video_target']
    real_only = [group for group in group_units if group not in fake_groups and group not in dev_groups]
    def development_priority(group):
        real = group_units[group]
        old_test = real.video_id.isin(historical_test).sum()
        old_cal = real.video_id.isin(historical_cal).sum()
        return (not bool(old_test), bool(old_cal), len(real), order(group))
    # A real-only component is either wholly reserved or wholly left available.
    # Never select one member merely to fill the real scoring quota.
    for group in sorted(real_only, key=development_priority):
        current = units[units[group_column].isin(dev_groups) & units.label.eq(0)].video_uid.nunique()
        count = group_units[group].video_uid.nunique()
        if current >= real_target:
            break
        if count > real_target - current:
            continue
        remaining = units[~units[group_column].isin(dev_groups | {group})]
        if not remaining.label.eq(0).any() or not remaining.label.eq(1).any():
            continue
        dev_groups.add(group)
    development_candidates = units[units[group_column].isin(dev_groups)]
    scored = list(ordered(development_candidates[development_candidates.label.eq(0)]).head(real_target).video_uid)
    for method, rows in development_candidates[development_candidates.label.eq(1)].groupby('method_id', sort=True):
        scored.extend(ordered(rows).head(policy['fake_videos_per_method_cap']).video_uid)
    scored = set(scored)
    dev = target[target.video_uid.isin(scored)].copy()
    dev['role'] = 'cdf_dev'
    ranking = target[~target[group_column].isin(dev_groups)].copy()
    ranking['role'] = 'target_full'
    remaining_units = units[~units[group_column].isin(dev_groups)]
    cal_groups = set(); cal_count = 0
    def calibration_priority(group):
        rows = group_units[group]
        return (-int(rows.video_id.isin(historical_cal).sum()), len(rows), order(group))
    cal_candidates = [group for group in group_units if group not in fake_groups and group not in dev_groups]
    for group in sorted(cal_candidates, key=calibration_priority):
        count = group_units[group].video_uid.nunique()
        if count + cal_count > policy['calibration_video_target']:
            continue
        would_test = remaining_units[~remaining_units[group_column].isin(cal_groups | {group})]
        if not would_test.label.eq(0).any() or not would_test.label.eq(1).any():
            continue
        cal_groups.add(group); cal_count += count
    cal = ranking[ranking[group_column].isin(cal_groups)].copy()
    test = ranking[~ranking[group_column].isin(cal_groups)].copy()
    cal['role'] = 'target_cal'; test['role'] = 'target_test'
    for left, right, label in [(dev, ranking, 'DEV/ranking'), (dev, cal, 'DEV/CAL'),
                               (dev, test, 'DEV/TEST'), (cal, test, 'CAL/TEST')]:
        for key in ['video_uid', 'group_id', 'rgb_sha256', group_column]:
            if set(left[key]) & set(right[key]):
                raise ValueError('CDF ' + label + ' overlap in ' + key)
    blocked_units = all_units[all_units[group_column].isin(dev_groups)].copy()
    blocked_units['development_group_id'] = blocked_units[group_column]
    group_scored = {group: json.dumps(sorted(dev.loc[dev[group_column].eq(group), 'video_uid'].unique()))
                    for group in dev_groups}
    blocked_units['development_scored_video_uids'] = blocked_units[group_column].map(group_scored)
    blocked_units['role'] = np.where(blocked_units.video_uid.isin(scored), 'cdf_dev', 'excluded')
    blocked_units['disposition'] = np.where(blocked_units.video_uid.isin(scored),
        'development_scoring_and_formal_exclusion', 'unscored_development_relation_quarantine')
    blocked_units['exclusion_reason'] = 'cdf_development_full_candidate_joint_relation_component'
    blocked_units['excluded_from'] = 'target_full,target_cal,target_test'
    cal_excluded = all_units[all_units[group_column].isin(cal_groups)
                             & ~all_units.video_uid.isin(set(cal.video_uid))].copy()
    cal_excluded['calibration_group_id'] = cal_excluded[group_column]
    cal_excluded['role'] = 'excluded'
    cal_excluded['exclusion_reason'] = 'target_calibration_full_candidate_joint_relation_component'
    cal_excluded['excluded_from'] = 'target_test'
    for name, frame in [('cdf_dev', dev), ('target_full', ranking), ('target_cal', cal), ('target_test', test),
                        ('cdf_development_exclusions', blocked_units), ('target_calibration_exclusions', cal_excluded)]:
        frame.to_parquet(out/(name + '.parquet'), index=False)
    ledger = pd.concat([blocked_units, cal_excluded], ignore_index=True)
    ledger.to_parquet(out/'cdf_role_exclusion_ledger.parquet', index=False)
    all_units['component_allocation'] = np.where(all_units[group_column].isin(dev_groups), 'development',
        np.where(all_units[group_column].isin(cal_groups), 'calibration', 'test'))
    all_units['scored_in_development'] = all_units.video_uid.isin(scored)
    all_units.to_parquet(out/'cdf_candidate_videos.parquet', index=False)
    all_units[['video_uid', 'video_id', 'group_id', group_column, 'source_ids', 'cdf_name_tokens']].to_parquet(
        out/'cdf_relation_membership.parquet', index=False)
    component_records = []
    for group, component in all_units.groupby(group_column, sort=True):
        qualified = component[component.fixed8_qualified]
        component_records.append({'cdf_relation_group_id': group,
            'candidate_videos': len(component), 'candidate_real_videos': int(component.label.eq(0).sum()),
            'candidate_fake_videos': int(component.label.eq(1).sum()), 'qualified_videos': len(qualified),
            'qualified_real_videos': int(qualified.label.eq(0).sum()),
            'qualified_fake_videos': int(qualified.label.eq(1).sum()),
            'fake_method_count': component.loc[component.label.eq(1), 'method_id'].nunique(),
            'fake_method_video_counts': json.dumps(component.loc[component.label.eq(1)].groupby('method_id').size().to_dict(), sort_keys=True),
            'historical_cal_real_videos': int((component.label.eq(0) & component.video_id.isin(historical_cal)).sum()),
            'allocation': 'development' if group in dev_groups else 'calibration' if group in cal_groups else 'test',
            'development_scored_videos': int(component.scored_in_development.sum())})
    pd.DataFrame(component_records).to_csv(out/'cdf_split_components.csv', index=False)
    dev_reals = dev.loc[dev.label.eq(0), 'video_uid'].nunique()
    dev_fakes = dev.loc[dev.label.eq(1), 'video_uid'].nunique()
    dev_methods = dev[dev.label.eq(1)].drop_duplicates('video_uid').groupby('method_id').size().to_dict()
    usable = bool(dev_reals and dev_fakes)
    scope = {'status': 'USABLE' if usable else 'NOT_USABLE', 'usable_for_epoch_selection': usable,
        'protocol': policy['version'], 'real_video_target': real_target, 'real_video_actual': dev_reals,
        'fake_videos_per_method_cap': policy['fake_videos_per_method_cap'], 'fake_video_actual': dev_fakes,
        'fake_method_video_counts': dev_methods, 'fake_method_coverage': sorted(dev_methods),
        'target_met': dev_reals == real_target and set(dev_methods) == set(METHODS),
        'development_joint_groups': sorted(dev_groups), 'quarantined_candidate_videos': len(blocked_units),
        'unscored_quarantined_candidate_videos': int((~blocked_units.video_uid.isin(scored)).sum()),
        'candidate_joint_groups': len(component_records), 'full_candidate_videos': len(all_units),
        'development_only': True, 'formal_cache_role': False, 'target_scores_used': False,
        'selection_rationale': 'Score-blind smallest feasible fake-bearing joint component, prefer complete method coverage; capped scoring subset, entire component quarantined; supplement whole real-only components, historical TEST preferred.',
        'fallback_reason': '' if usable else 'No feasible fake-bearing development component can be held out while preserving formal method coverage and two-class test; no target predictions may be used to choose epochs.',
        'policy': policy}
    save_json(out/'cdf_development_scope.json', scope)
    save_json(out/'cdf_development_policy.json', policy)
    budgets = [n for n in [10, 20, 40, 80] if n <= cal.video_uid.nunique()]
    evidence = {'status': 'PROXY_SCOPE_EMPIRICAL_ONLY', 'cal_real_videos': cal.video_uid.nunique(),
        'test_real_videos': test.loc[test.label.eq(0), 'video_uid'].nunique(),
        'test_fake_videos': test.loc[test.label.eq(1), 'video_uid'].nunique(),
        'ranking_real_videos': ranking.loc[ranking.label.eq(0), 'video_uid'].nunique(),
        'ranking_fake_videos': ranking.loc[ranking.label.eq(1), 'video_uid'].nunique(),
        'excluded_videos': len(cal_excluded), 'development_excluded_candidate_videos': len(blocked_units),
        'available_target_calibration_budgets': budgets,
        'unavailable_target_calibration_budgets': [n for n in [10, 20, 40, 80] if n not in budgets],
        'historic_split_sha256': split['source_split_sha256'], 'historic_protocol_sha256': split['source_protocol_sha256'],
        'historical_cal_candidates_retained': int(cal.drop_duplicates('video_uid').video_id.isin(historical_cal).sum()),
        'cal_candidate_pool': 'New score-blind whole real-only joint components; historical80 IDs preferred, remaining real-only components may replace linked historical candidates. Maximum80; reserve two-class TEST.',
        'test_candidate_pool': 'All qualified remaining non-development non-calibration joint components; previous97 restriction is superseded.',
        'ranking_cohort': 'CDF_formal_ranking_after_development_relation_exclusion',
        'isolation': 'Joint transitive closure of all-candidate exact RGB/known source groups, same CDF names across methods and conservative id-token proxies.',
        'ancestor_independence_certified': False, 'strict_NP_population_guarantee': False,
        'development': scope,
        'unresolved': 'Complete generated-parent, donor/driver, audio and YouTube identity ancestry is not available. A filename is a proxy, not authenticated identity.'}
    save_json(out/'cdf_calibration_scope.json', evidence)
    if set(test.label.unique()) != {0, 1} or set(ranking.label.unique()) != {0, 1}:
        save_json(out/'cdf_allocation_failure.json', {'status': 'NO_USABLE_TWO_CLASS_FORMAL_TEST', 'scope': evidence})
        raise ValueError('CDF joint relation allocation leaves no usable two-class formal TEST')
    return evidence


def build(output_root=DEFAULT_OUTPUT, data_root=DATA_ROOT, definition_root=DEFINITION_ROOT, workers=4):
    start=time.time(); output_root=Path(output_root);data_root=Path(data_root).resolve();out=output_root/'data/D'
    definition=load_definition(definition_root)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists(): raise ValueError('D is already frozen; refusing to overwrite scientific manifests')
    print('D build original42 indices',flush=True)
    frames,provenance,index_hashes,unknown=rebuild_indices(data_root,definition)
    provenance.to_parquet(out/'real_index_provenance.parquet',index=False)
    save_json(out/'unknown_source_index_audit.json',unknown)
    splits=definition['splits'];split_hashes={name:evidence['sha256'] for name,evidence in definition['split_provenance'].items()}
    roles,records=pair_partition(splits)
    save_json(out/'source_pair_partition.json',{'seed':SEED,'algorithm':'lexicographically sorted internally sorted official train pairs; numpy.default_rng(seed).permutation(360) first306 TRAIN last54 DEV','numpy_version':np.__version__,'pairs':records,'source_roles':roles,'official_split_sha256':split_hashes})
    candidates=[]
    for video in definition['source_real_candidates']['videos']:
        for name in video['frames']:
            relative=Path(video['directory'])/name
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe sourceCAL path')
            candidates.append({'image_path':str(data_root/relative),'source_id':video['source_id']})
    cal=pd.DataFrame(candidates)
    if cal.image_path.duplicated().any(): raise ValueError('Duplicate sourceCAL paths')
    if len(cal)!=definition['source_real_candidates']['candidate_paths']: raise ValueError('SourceCAL candidate count mismatch')
    cal=cal[['image_path','source_id']].copy();cal['dataset']='DF40';cal['domain']='ff';cal['released_role']='official_val_real';cal['label']=0;cal['method_id']='real';cal['method_type']='real';cal['evaluation_unit']='video_frame'
    cal['video_id']=cal.source_id;cal['video_uid']=cal.video_id.map(lambda v:video_uid('ff','real',v));cal['source_ids']=cal.source_id.map(lambda v:json.dumps([v]));cal['frame_number']=cal.image_path.map(lambda v:int(Path(v).stem))
    cal['index_path']=str(Path(definition_root).resolve()/'val.json');cal['original_image_path']=cal.image_path;cal['original_index_id']=cal.source_id;cal['driving_source_id']='unknown';cal['relation_evidence']='official_FF_val_real_source';cal['processing_version']='DF40_released_RGB_fixed8_numeric_v2'
    frames=pd.concat([frames,cal.drop(columns='source_id')],ignore_index=True)
    print(f'D rebuilt association rows={len(frames)} unique_paths={frames.image_path.nunique()}',flush=True)
    frames,audit=audit_images(frames,out,workers)
    frames=assign_roles(frames,roles)
    print('D full source/video/RGB graph',flush=True)
    frames=attach_components(frames,splits);frames=apply_dispositions(frames)
    eligible=frames[frames.role.ne('excluded')].copy();selected=select_eight(eligible)
    missing=set(eligible.video_uid)-set(selected.video_uid)
    exclusions=frames[frames.role.eq('excluded')].copy()
    insufficient=eligible[eligible.video_uid.isin(missing)].copy();insufficient['exclusion_reason']='fewer_than8_distinct_valid_RGB_frames';insufficient['role']='excluded'
    exclusions=pd.concat([exclusions,insufficient],ignore_index=True)
    for role in ['train','dev','source_test','target_full']:
        got=set(selected.loc[selected.role.eq(role)&selected.label.eq(1),'method_id'])
        if got!=set(METHODS): raise ValueError(f'Missing21method coverage in {role}: {set(METHODS)-got}')
    if selected.groupby('rgb_sha256').role.nunique().max()>1: raise ValueError('Selected RGB crosses roles')
    if selected.groupby('group_id').role.nunique().max()>1: raise ValueError('Selected full graph component crosses roles')
    if selected.groupby('video_uid').size().ne(8).any() or selected.groupby('video_uid').rgb_sha256.nunique().ne(8).any(): raise ValueError('Fixed8 video qualification failed')
    print('D score-blind CDF joint-relation DEV/CAL/TEST allocation',flush=True)
    cdf=cdf_proxy_split(selected,frames,definition,out)
    cdf_dev=pd.read_parquet(out/'cdf_dev.parquet')
    cdf_ranking=pd.read_parquet(out/'target_full.parquet')
    selected=pd.concat([selected[~selected.domain.eq('cdf')],cdf_dev,cdf_ranking],ignore_index=True)
    selected['experiment_role']=selected.role
    dev_groups=set(cdf['development']['development_joint_groups'])
    dev_scored_uids=set(cdf_dev.video_uid)
    blocked=frames.domain.eq('cdf') & frames.cdf_relation_group_id.isin(dev_groups)
    quarantine=blocked & ~frames.video_uid.isin(dev_scored_uids)
    frames.loc[blocked & frames.video_uid.isin(dev_scored_uids),'role']='cdf_dev'
    frames.loc[quarantine,'role']='excluded'
    frames.loc[quarantine,'exclusion_reason']=frames.loc[quarantine,'exclusion_reason'].map(
        lambda value:';'.join(filter(None,[value,'cdf_development_full_candidate_joint_relation_component'])))
    frames['experiment_role']=frames.role
    frames.to_parquet(out/'full_associations.parquet',index=False)
    development_exclusions=frames[quarantine].copy()
    exclusions=pd.concat([exclusions,development_exclusions],ignore_index=True).drop_duplicates(['video_uid','image_path'])
    exclusions.to_parquet(out/'exclusions.parquet',index=False)
    train,associations=canonical_training_pool(selected[selected.role.eq('train')])
    train.to_parquet(out/'train.parquet',index=False);associations.to_parquet(out/'training_associations.parquet',index=False)
    selected.to_parquet(out/'selected_associations.parquet',index=False)
    for role in ['dev','source_cal','source_test','target_full','cdf_dev']:
        selected[selected.role.eq(role)].to_parquet(out/f'{role}.parquet',index=False)
    identity=selected.drop_duplicates('video_uid')[['dataset','domain','role','label','method_id','method_type','video_id','video_uid','group_id','source_ids','relation_evidence','group_basis','cdf_relation_group_id','cdf_name_tokens']]
    identity.to_parquet(out/'video_identity.parquet',index=False)
    counts=selected.groupby(['role','method_id','label']).agg(rows=('image_path','size'),videos=('video_uid','nunique'),unique_RGB=('rgb_sha256','nunique'),groups=('group_id','nunique')).reset_index()
    rq4=pd.concat([pd.read_parquet(out/'target_cal.parquet'),pd.read_parquet(out/'target_test.parquet')],ignore_index=True)
    rq4_counts=rq4.groupby(['role','method_id','label']).agg(rows=('image_path','size'),videos=('video_uid','nunique'),unique_RGB=('rgb_sha256','nunique'),groups=('group_id','nunique')).reset_index()
    counts=pd.concat([counts,rq4_counts],ignore_index=True)
    counts.to_csv(out/'counts.csv',index=False)
    qualification=train.groupby('method_id',sort=True).head(2).copy()
    qualification.to_parquet(out/'qualification_source.parquet',index=False)
    masses=associations.groupby('method_id').sampling_contribution.sum().to_dict()
    exclusion_counts=exclusions.drop_duplicates('video_uid').groupby('exclusion_reason').size().to_dict()
    hashes={p.name:digest(p) for p in sorted(out.glob('*')) if p.is_file() and p.name not in ['freeze.json','audit_progress.json']}
    from .protocol import PROTOCOL
    receipt={'status':'FROZEN','protocol':PROTOCOL,'data_definition_sha256':digest(Path(definition_root)/'definition.json'),'raw_data_root':str(data_root),'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
        'method_registry':{'FS9':FS,'FR12':FR},'method_count':21,'manifest_sha256':hashes,'original_indices_sha256':index_hashes,'official_split_sha256':split_hashes,
        'code_sha256':digest(__file__),'canonical_train_inputs':len(train),'training_association_rows':len(associations),'samples_per_epoch':math.ceil(len(train)/128)*128,
        'sampling_method_probability':masses,'source_pair_partition':{'train_pairs':306,'dev_pairs':54,'seed':SEED},
        'counts':counts.to_dict('records'),'exclusion_video_counts':exclusion_counts,'unresolved_source_index_videos':len(unknown),
        'cache_audit':audit,'cdf_calibration_scope':cdf,'cdf_development_scope':cdf['development'],
        'cdf_development_policy_sha256':digest(out/'cdf_development_policy.json'),
        'cdf_ranking_cohort':'CDF_formal_ranking_after_development_relation_exclusion',
        'independence_scope':'Known FF filename sources, official source pairs, and full candidate exact RGB components; unknown audio/driver ancestry remains unverified. CDF uses video/content and conservative identity-name proxies only.',
        'source_validation_and_test_disjoint':True,'target_scores_used':False,'original_data_modified':False,'elapsed_seconds':time.time()-start}
    save_json(out/'freeze.json',receipt)
    report_dir=output_root/'reports';report_dir.mkdir(parents=True,exist_ok=True)
    save_json(report_dir/'d_manifest_complete.json',{'status':'COMPLETE','freeze_json':str(out/'freeze.json'),'freeze_sha256':digest(out/'freeze.json'),'counts':counts.to_dict('records'),'exclusions':exclusion_counts,'elapsed_seconds':time.time()-start})
    print(json.dumps({'status':'FROZEN','canonical_train_inputs':len(train),'training_association_rows':len(associations),'source_counts':selected.groupby('role').video_uid.nunique().to_dict(),'cdf_scope':cdf,'seconds':time.time()-start}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--definition-root',type=Path,default=DEFINITION_ROOT)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();build(args.output_root,args.data_root,args.definition_root,args.workers)
