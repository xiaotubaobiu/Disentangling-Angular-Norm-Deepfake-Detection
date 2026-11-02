"""CPU-only journal H Raw analysis: all-B deployment and matched common 2x2."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import analysis_d as io
from . import cache_h,scoring_d as score,stats_h,trainer

ROLES=cache_h.ROLES
SEEDS=cache_h.SEEDS
BACKBONES=tuple(cache_h.DIMENSIONS)
SOURCE_METHODS=('Deepfakes','Face2Face','FaceSwap','NeuralTextures')
TARGET_METHODS=('DeepFakeDetection',)
MOTHER_COLUMNS=('video_uid','mother_id','mother_rgb_sha256','label','method_id','group_id')


def _mother_rows(frame):
    if not set(MOTHER_COLUMNS).issubset(frame):raise ValueError('H logical mother identity is missing')
    q=frame[list(MOTHER_COLUMNS)].sort_values(['video_uid','mother_id']).reset_index(drop=True)
    if q[['video_uid','mother_id']].duplicated().any():raise ValueError('duplicate H logical mother')
    return q


def _same_mothers(a,b):
    if not _mother_rows(a).equals(_mother_rows(b)):raise ValueError('H matched common mothers differ')


def _same_inputs(a,b):
    _same_mothers(a,b)
    columns=list(MOTHER_COLUMNS)+['image_path','rgb_sha256','content_sha256','processing_version']
    if not set(columns).issubset(a) or not set(columns).issubset(b):raise ValueError('H frozen input provenance missing')
    keys=['video_uid','mother_id']
    if not a[columns].sort_values(keys).reset_index(drop=True).equals(b[columns].sort_values(keys).reset_index(drop=True)):
        raise ValueError('H frozen input bytes/paths differ between model histories')


def score_pools(cache_data,*,aggregation='mean_logit'):
    """Keep original FP32 heads, Raw FP64 margins and precise rank objects."""
    if aggregation not in ('mean_logit','mean_probability'):raise ValueError('unknown H aggregation')
    if set(cache_data)!={'B','AB'} or any(set(roles)!=set(ROLES) for roles in cache_data.values()):
        raise ValueError('H requires two histories and exactly seven cache roles')
    videos={history:{} for history in cache_data}
    for history,roles in cache_data.items():
        original=None
        for role,(frames,h,W,b) in roles.items():
            leaf=role.split('/')[-1];version='AB' if role.startswith('common/AB/') else 'B'
            domain='target' if leaf=='target_test' else 'source'
            if not frames.role.eq(leaf).all() or not frames.domain.eq(domain).all() or not frames.processing_version.eq(version).all():
                raise ValueError('H role/domain/processing version mismatch')
            if np.asarray(W).dtype!=np.float32 or np.asarray(b).dtype!=np.float32:raise ValueError('H retains original FP32 two-class head rows/bias')
            if original is None:original=(W.copy(),b.copy())
            elif not np.array_equal(W,original[0]) or not np.array_equal(b,original[1]):raise ValueError('H model roles contain different original heads')
            if leaf=='source_cal' and not frames.label.eq(0).all():raise ValueError('H calibration is real-only')
            margins=score.frame_margins(frames,h,W,b)
            videos[history][role]=score.aggregate_videos(margins,representation='raw',aggregation=aggregation)
        for leaf in ('source_test','target_test'):
            b=roles['common/B/'+leaf][0];ab=roles['common/AB/'+leaf][0];all_b=roles['B/'+leaf][0]
            _same_mothers(b,ab)
            uids=set(b.video_uid)
            if not uids.issubset(set(all_b.video_uid)):raise ValueError('H common B must be an all-B subset')
            _same_inputs(b,all_b[all_b.video_uid.isin(uids)])
    for role in ROLES:_same_inputs(cache_data['B'][role][0],cache_data['AB'][role][0])
    out={'source_cal':{history:videos[history]['B/source_cal'] for history in ('B','AB')}}
    for leaf in ('source_test','target_test'):
        out[leaf]=dict(all_B={'H00':videos['B']['B/'+leaf],'H10':videos['AB']['B/'+leaf]},
            common={'H00':videos['B']['common/B/'+leaf],'H01':videos['B']['common/AB/'+leaf],
                    'H10':videos['AB']['common/B/'+leaf],'H11':videos['AB']['common/AB/'+leaf]})
    return out


def analyze_cohort(cells,source_cal,*,fixed_methods,aggregation='mean_logit',repeats=2000,seed=20261023):
    result=stats_h.analyze_cohort(cells,source_cal,fixed_methods=fixed_methods,
        aggregation=aggregation,repeats=repeats,seed=seed)
    result['bootstrap']=score.enforce_bootstrap_group_support(result['bootstrap'],cells['H00'])
    return result


def verify_saved_metrics(cells,metrics,thresholds,*,fixed_methods):
    if len(metrics)!=2*len(cells) or metrics.duplicated(['cell','threshold_kind']).any():
        raise ValueError('H saved metric inventory is incomplete or duplicated')
    checks=[]
    for row in metrics.to_dict('records'):
        cell=row['cell']
        if cell not in cells or row['threshold_kind'] not in ('default','source_real'):raise ValueError('unknown H saved cell/threshold kind')
        q=cells[cell];history=stats_h.CELL_HISTORY[cell]
        expected=(0. if q.aggregation.iloc[0]=='mean_logit' else .5) if row['threshold_kind']=='default' else thresholds[history]
        io._equal(row['threshold'],expected,'H independent frozen source threshold')
        usable=bool(q.ranking_usable.all())
        auc=io.independent_auc(q.label,q.rank_score) if usable else float('nan')
        io._equal(auc,row['pooled_auc'],'H independent AUC')
        macro=[];tpr=[];real=q[q.label.eq(0)]
        for method in fixed_methods:
            fake=q[q.label.eq(1)&q.method_id.eq(method)]
            if fake.empty:macro=[];tpr=[];break
            part=pd.concat([real,fake],ignore_index=True)
            if usable:macro.append(io.independent_auc(part.label,part.rank_score))
            tpr.append(float(fake.score.ge(expected).mean()))
        io._equal(row['macro_auc'],float(np.mean(macro)) if macro and usable else float('nan'),'H independent macro AUC')
        op=io.independent_operating(q.label,q.score,expected)
        for key,value in op.items():io._equal(row[key],value,'H independent '+key)
        mt=float(np.mean(tpr)) if tpr else float('nan')
        io._equal(row['macro_tpr'],mt,'H independent macro TPR');io._equal(row['macro_ba'],(1-op['fpr']+mt)/2,'H independent macro BA')
        checks.append(dict(cell=cell,threshold_kind=row['threshold_kind'],status='PASSED' if usable else 'OPERATING_PASSED_RANKING_UNRESOLVED'))
    return checks


def coverage_report(root,freeze):
    """Read full frozen candidate denominators; never derive them from caches."""
    root=Path(root).resolve();metadata=freeze.get('metadata',{})
    coverage=metadata.get('coverage');inventory=metadata.get('inventory')
    if not isinstance(coverage,dict) or not isinstance(inventory,dict):raise ValueError('H frozen candidate coverage/inventory evidence is required')
    paths={}
    for name,item in (('coverage',coverage),('inventory',inventory)):
        path=Path(item['path'])
        if not path.is_absolute():path=root/path
        if not path.is_file() or trainer.digest(path)!=item['sha256']:raise ValueError('H frozen candidate '+name+' hash mismatch')
        paths[name]=path
    q=pd.read_csv(paths['coverage'])
    required={'video_uid','label','role','method_id','dataset','processed','decode_ok','A','B','AB','common','quarantined'}
    if not required.issubset(q) or q[list(required)].isna().any().any() or q.video_uid.duplicated().any():raise ValueError('H candidate coverage requires complete unique video rows')
    count=len(q)
    if count!=coverage.get('candidate_videos') or count!=inventory.get('candidate_videos'):raise ValueError('H frozen candidate denominator count mismatch')
    raw=json.loads(paths['inventory'].read_text());records=raw if isinstance(raw,list) else raw.get('records',raw.get('inventory'))
    if not isinstance(records,list):raise ValueError('H candidate inventory records are missing')
    source=pd.DataFrame(records);keys=['video_uid','label','role','method_id','dataset']
    if (not set(keys).issubset(source) or source.video_uid.duplicated().any()
            or not q[keys].sort_values('video_uid').reset_index(drop=True).equals(source[keys].sort_values('video_uid').reset_index(drop=True))):
        raise ValueError('H candidate coverage identities differ from the frozen original inventory')
    for key in ('processed','decode_ok','A','B','AB','common','quarantined'):
        values=q[key].astype(str).str.lower()
        if not values.isin(['true','false','1','0']).all():raise ValueError('H candidate coverage flag is not boolean')
        q[key]=values.isin(['true','1'])
    active=q.processed&q.decode_ok&~q.quarantined&~q.role.eq('source_unused')
    reports=[]
    for identity,part in q.groupby(['dataset','role','method_id','label'],sort=True):
        valid=active.loc[part.index];row=dict(zip(('dataset','role','method_id','label'),identity))
        row.update(candidate_videos=len(part),processed_videos=int(part.processed.sum()),
            decoded_videos=int(part.decode_ok.sum()),quarantined_videos=int(part.quarantined.sum()))
        for version in ('A','B','AB','common'):
            row['geometric_'+version+'_videos']=int(part[version].sum())
            row['active_'+version+'_videos']=int((part[version]&valid).sum())
            row['active_'+version+'_fraction']=float((part[version]&valid).sum()/len(part))
        reports.append(row)
    return dict(schema='JOURNAL_H_CANDIDATE_COVERAGE_REPORT_V1',candidate_videos=count,
        real_candidate_videos=int(q.label.eq(0).sum()),fake_candidate_videos=int(q.label.eq(1).sum()),
        denominator_scope=coverage.get('denominator_scope'),source_coverage_sha256=coverage['sha256'],
        source_inventory_sha256=inventory['sha256'],by_role_method=reports,
        real_counting='one unique candidate video row; real videos are never repeated per fake method',
        eligibility='Geometric success and active success after quarantine are reported separately')


def _code_identities():
    from . import bootstrap_fast
    return {Path(m.__file__).name:trainer.digest(m.__file__) for m in (cache_h,io,score,stats_h,bootstrap_fast)}|{'analysis_h.py':trainer.digest(__file__)}


def _pair_context(config_path,backbone,seed):
    if backbone not in BACKBONES or seed not in SEEDS:raise ValueError('unregistered H backbone/formal seed')
    path=Path(config_path).resolve();config=json.loads(path.read_text());root=Path(config['artifact_root']).resolve()
    tasks=cache_h._registry(config,backbone)
    runs={h:next(run for run,t in tasks.items() if t['seed']==seed and t['history']==h) for h in ('B','AB')}
    contexts={};markers={}
    for history,run in runs.items():
        base,_,_=cache_h._context(path,run);contexts[history]=base
        aggregate=root/'features/H'/run/'complete.json'
        if not aggregate.is_file():raise ValueError('H formal caches are incomplete')
        receipt=json.loads(aggregate.read_text())
        if receipt.get('status')!='CACHE_COMPLETE' or receipt.get('identity')!=base or set(receipt.get('roles',{}))!=set(ROLES):raise ValueError('H aggregate formal cache identity/roles mismatch')
        markers[history+'/complete']=trainer.digest(aggregate)
        for role in ROLES:
            folder=root/'features/H'/run/role
            cache_h.verify_cache(folder,expected_run_id=run,expected_role=role,config_path=path)
            sha=trainer.digest(folder/'complete.json')
            if receipt['roles'][role].get('complete_sha256')!=sha:raise ValueError('H role completion differs from aggregate cache')
            markers[history+'/'+role]=sha
    freeze=json.loads((root/'data/H/freeze.json').read_text())
    identity=dict(schema='JOURNAL_H_PAIR_ANALYSIS_IDENTITY_V1',backbone=backbone,seed=seed,
        run_ids=runs,cache_contexts=contexts,cache_completion_sha256=markers,
        config_sha256=trainer.digest(path),data_freeze_sha256=trainer.digest(root/'data/H/freeze.json'),
        code_sha256=_code_identities(),bootstrap_repeats=2000,resampling_seed=20261023,
        primary='DFD_all_B_Raw_mean_logit_H10_minus_H00',source_threshold='each_model_fixed_B_source_CAL_reused_for_B_and_AB_inputs',
        target_development_used=False,formal_trajectory_target_used=False)
    return root,config,freeze,runs,identity


def formal_cache_status(config_path):
    path=Path(config_path);config=json.loads(path.read_text());root=Path(config['artifact_root'])
    tasks={run:t for run,t in config.get('runs',{}).items() if t.get('route')=='H' and t.get('phase')=='formal' and t.get('recipe_frozen') is True}
    ready=[run for run in tasks if (root/'features/H'/run/'complete.json').is_file()]
    return dict(status='FORMAL_H_CACHE_MARKERS_PRESENT' if tasks and len(ready)==len(tasks) else 'WAITING_FORMAL_H_CACHES',
        formal_registered=len(tasks),cache_markers_present=len(ready),run_ids=sorted(ready),marker_only=True)


def _analysis_inventory():
    names={'identity.json','coverage.json','independent_saved_score_recompute.json','limitations.json'}
    for aggregation in ('mean_logit','mean_probability'):
        names.update(f'{aggregation}/source_cal_{h}.csv' for h in ('B','AB'))
        for role in ('source_test','target_test'):
            for cohort,cells in (('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))):
                prefix=f'{aggregation}/{role}/{cohort}'
                names.update(f'{prefix}/{cell}_scores.csv' for cell in cells)
                names.update(f'{prefix}/{name}' for name in ('metrics.csv','comparisons.csv','thresholds.json','conditional_bootstrap.json','independent_kernel_checks.json'))
    return names


def _verify_completed(output,identity):
    output=Path(output);path=output/'complete.json'
    if not path.is_file():return None
    receipt=json.loads(path.read_text())
    if (receipt.get('schema')!='JOURNAL_H_PAIR_ANALYSIS_COMPLETE_V1' or receipt.get('status') not in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
            or receipt.get('identity')!=identity or receipt.get('independent_recompute')!='PASSED_FOR_ALL_REPORTABLE_CELLS'
            or receipt.get('target_development_used') is not False):raise ValueError('H completed analysis identity/status/provenance mismatch')
    hashes=receipt.get('artifact_sha256',{});actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p!=path}
    if set(hashes)!=_analysis_inventory() or actual!=set(hashes):raise ValueError('H committed analysis artifact inventory changed')
    if any(not (output/name).resolve().is_relative_to(output.resolve()) or trainer.digest(output/name)!=sha for name,sha in hashes.items()):raise ValueError('H committed analysis artifact hash/path changed')
    return receipt


def analyze_pair(config_path,backbone,seed):
    root,config,freeze,runs,identity=_pair_context(config_path,backbone,seed)
    output=root/f'analysis/H/{backbone}/seed{seed}'
    if not output.resolve().is_relative_to(root):raise ValueError('unsafe H analysis output path')
    with trainer.exclusive_file(root/f'analysis/H/.locks/{backbone}_seed{seed}.lock'):
        prior=_verify_completed(output,identity)
        if prior:return prior
        if shutil.disk_usage(root).free<=50*1024**3:raise RuntimeError('H analysis requires 50GiB storage reserve')
        output.mkdir(parents=True,exist_ok=True);header=output/'identity.json'
        if header.exists() and json.loads(header.read_text())!=identity:raise ValueError('H partial analysis identity changed')
        io._save_json(header,identity)
        data={history:{role:cache_h.load_features(root/'features/H'/run/role,
            expected_run_id=run,expected_role=role,config_path=config_path) for role in ROLES} for history,run in runs.items()}
        independent=[];numerical=[];support=[]
        for aggregation in ('mean_logit','mean_probability'):
            pools=score_pools(data,aggregation=aggregation)
            for history,q in pools['source_cal'].items():io.write_scores(output/f'{aggregation}/source_cal_{history}.csv',q)
            cal={h:io.read_scores(output/f'{aggregation}/source_cal_{h}.csv') for h in runs}
            for role,methods in (('source_test',SOURCE_METHODS),('target_test',TARGET_METHODS)):
                for cohort,cells in pools[role].items():
                    folder=output/aggregation/role/cohort;folder.mkdir(parents=True,exist_ok=True)
                    for cell,q in cells.items():io.write_scores(folder/f'{cell}_scores.csv',q)
                    restored={cell:io.read_scores(folder/f'{cell}_scores.csv') for cell in cells}
                    result=analyze_cohort(restored,cal,fixed_methods=methods,aggregation=aggregation)
                    io._csv(folder/'metrics.csv',result['point_metrics']);io._csv(folder/'comparisons.csv',result['comparisons'])
                    io._save_json(folder/'thresholds.json',result['thresholds'])
                    checks=verify_saved_metrics(restored,pd.read_csv(folder/'metrics.csv',float_precision='round_trip'),
                        json.loads((folder/'thresholds.json').read_text()),fixed_methods=methods)
                    independent.append(dict(role=role,cohort=cohort,aggregation=aggregation,checks=checks))
                    numerical.extend(dict(role=role,cohort=cohort,aggregation=aggregation,**row) for row in checks if row['status']!='PASSED')
                    io._save_json(folder/'conditional_bootstrap.json',result['bootstrap']);io._save_json(folder/'independent_kernel_checks.json',result['independent_checks'])
                    support.append(dict(role=role,cohort=cohort,aggregation=aggregation,
                        **{key:result['bootstrap'][key] for key in ('n_real_groups','n_fake_groups','method_fake_group_counts','conditional_interval_stability')}))
        io._save_json(output/'coverage.json',coverage_report(root,freeze))
        io._save_json(output/'independent_saved_score_recompute.json',independent)
        io._save_json(output/'limitations.json',dict(primary='DFD all-B Raw mean-logit H10-H00; source FF++ reported separately',
            cohorts='all-B deployment and matched common 2x2 are never subtracted across cohorts',
            source_threshold='each model uses its fixed B SOURCE_CAL threshold for both B and AB test inputs',
            training_seed_count=1,three_seed_summary_pending=True,target_development_used=False,
            independent_group_support=support,numerical_NA=numerical,
            bootstrap='2000 shared conditional TEST group draws; sparse independent class/method groups make intervals NA',
            precision='original FP32 feature/head promoted before row subtraction; rank objects retain arbitrary precision; operating decisions explicitly use FP64 scores'))
        if _pair_context(config_path,backbone,seed)[-1]!=identity:raise ValueError('H analysis inputs changed during computation')
        hashes={str(p.relative_to(output)):trainer.digest(p) for p in output.rglob('*') if p.is_file() and p.name!='complete.json'}
        if set(hashes)!=_analysis_inventory():raise ValueError('H analysis did not produce its complete registered inventory')
        receipt=dict(schema='JOURNAL_H_PAIR_ANALYSIS_COMPLETE_V1',status='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' if numerical else 'ANALYSIS_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),backbone=backbone,seed=seed,identity=identity,
            artifact_sha256=hashes,independent_recompute='PASSED_FOR_ALL_REPORTABLE_CELLS',target_development_used=False,
            formal_trajectory_target_used=False,paired_epoch=identity['cache_contexts']['B']['best_epoch'])
        io._save_json(output/'complete.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('status','run'))
    parser.add_argument('--config',required=True);parser.add_argument('--backbone',choices=BACKBONES)
    parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.command=='run' and (args.backbone is None or args.seed is None):parser.error('run requires backbone and seed')
    result=formal_cache_status(args.config) if args.command=='status' else analyze_pair(args.config,args.backbone,args.seed)
    print(json.dumps(io._jsonable(result),indent=2,allow_nan=False))
