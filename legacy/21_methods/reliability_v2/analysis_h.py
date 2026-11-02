"""CPU H analysis: all-B deployment and a separate common-cohort 2x2.

Only effective frozen formal models, gated by every paired numerical ledger,
may supply the seven FP32 caches. This module never launches a GPU job, changes
cohorts, selects recipes or uses AB calibration thresholds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import analysis_d as io
from . import scoring_d as score
from .trainer import digest, exclusive_file

ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST',
       'common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST')
SEEDS=(7,42,123,2026,3407)
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SOURCE_METHODS=('Deepfakes','Face2Face','FaceSwap','NeuralTextures')
MOTHER_COLUMNS=('video_uid','mother_id','mother_rgb_sha256','label','method_id','group_id')


def _mother_rows(frame):
    if not set(MOTHER_COLUMNS) <= set(frame):raise ValueError('H logical mother identity is missing')
    q=frame[list(MOTHER_COLUMNS)].sort_values(['video_uid','mother_id']).reset_index(drop=True)
    if q[['video_uid','mother_id']].duplicated().any():raise ValueError('duplicate H logical mother')
    return q


def _same_mothers(a,b):
    if not _mother_rows(a).equals(_mother_rows(b)):raise ValueError('H matched common mothers differ')


def _same_inputs(a,b):
    _same_mothers(a,b)
    columns=list(MOTHER_COLUMNS)+['image_path','rgb_sha256','content_sha256','processing_version']
    if not set(columns)<=set(a) or not set(columns)<=set(b):raise ValueError('H frozen input provenance missing')
    key=['video_uid','mother_id']
    if not a[columns].sort_values(key).reset_index(drop=True).equals(b[columns].sort_values(key).reset_index(drop=True)):
        raise ValueError('H input bytes/paths differ between identical role caches')


def score_pools(cache_data, *, aggregation='mean_logit'):
    """Score already-provenanced FP32 cache arrays without any I/O or GPU.

    Checks both model histories consume exactly the same input for a role and
    that common B is an exact subset of all B. Logical mothers match B and AB.
    Returns source_real calibration separately from each TEST cohort.
    """
    if aggregation not in ('mean_logit','mean_probability'):raise ValueError('unregistered H aggregation')
    if set(cache_data)!={'B','AB'} or any(set(x)!=set(ROLES) for x in cache_data.values()):
        raise ValueError('exactly two histories and seven cache roles required')
    videos={history:{} for history in cache_data}
    for history,roles in cache_data.items():
        original=None
        for role,(frames,h,W,b) in roles.items():
            leaf=role.split('/')[-1];version='AB' if role.startswith('common/AB/') else 'B'
            domain='target' if leaf=='TARGET_TEST' else 'source'
            if (not frames.role.eq(leaf).all() or not frames.domain.eq(domain).all()
                or 'processing_version' not in frames or not frames.processing_version.eq(version).all()):
                raise ValueError('H role/domain/processing version mismatch')
            if np.asarray(W).dtype!=np.float32 or np.asarray(b).dtype!=np.float32:
                raise ValueError('H cache must retain original FP32 head rows/bias')
            if original is None:original=(W.copy(),b.copy())
            elif not np.array_equal(W,original[0]) or not np.array_equal(b,original[1]):
                raise ValueError('H model cache roles have different original heads')
            if leaf=='SOURCE_CAL' and not frames.label.eq(0).all():raise ValueError('H SOURCE_CAL must be real-only')
            margins=score.frame_margins(frames,h,W,b)
            videos[history][role]=score.aggregate_videos(margins,representation='raw',aggregation=aggregation)
        for leaf in ['SOURCE_TEST','TARGET_TEST']:
            commonB=roles['common/B/'+leaf][0];commonAB=roles['common/AB/'+leaf][0];allB=roles['B/'+leaf][0]
            _same_mothers(commonB,commonAB)
            uids=set(commonB.video_uid)
            if not uids<=set(allB.video_uid):raise ValueError('common B is not an all-B subset')
            _same_inputs(commonB,allB[allB.video_uid.isin(uids)])
    for role in ROLES:_same_inputs(cache_data['B'][role][0],cache_data['AB'][role][0])
    out={'source_cal':{history:videos[history]['B/SOURCE_CAL'] for history in ['B','AB']}}
    for leaf in ['SOURCE_TEST','TARGET_TEST']:
        out[leaf]=dict(all_B={'H00':videos['B']['B/'+leaf],'H10':videos['AB']['B/'+leaf]},
            common={'H00':videos['B']['common/B/'+leaf],'H01':videos['B']['common/AB/'+leaf],
                    'H10':videos['AB']['common/B/'+leaf],'H11':videos['AB']['common/AB/'+leaf]})
    return out


def verify_saved_metrics(cells, metrics, thresholds, *, fixed_methods):
    """Independent concordance/FP/TP recomputation from re-read score exports."""
    if len(metrics)!=2*len(cells) or metrics.duplicated(['cell','threshold_kind']).any():
        raise ValueError('independent metric inventory must contain both operating points for every cell')
    checks=[]
    for row in metrics.to_dict('records'):
        cell=row['cell'];q=cells[cell];history='B' if cell in ('H00','H01') else 'AB'
        expected=(0. if q.aggregation.iloc[0]=='mean_logit' else .5) if row['threshold_kind']=='default' else thresholds[history]
        if row['threshold_kind'] not in ('default','source_real'):raise ValueError('unregistered H threshold kind')
        io._equal(row['threshold'],expected,'independent frozen source/default threshold')
        usable=bool(q.ranking_usable.all()) if 'ranking_usable' in q else True
        auc=io.independent_auc(q.label,q.rank_score) if usable else float('nan')
        io._equal(row['pooled_auc'],auc,'independent AUC')
        macro=[];macro_tpr=[]
        real=q[q.label.eq(0)]
        for method in fixed_methods:
            fake=q[q.label.eq(1)&q.method_id.eq(method)]
            if fake.empty:macro=[];macro_tpr=[];break
            part=pd.concat([real,fake],ignore_index=True)
            if usable:macro.append(io.independent_auc(part.label,part.rank_score))
            macro_tpr.append(float((fake.score>=expected).mean()))
        io._equal(row['macro_auc'],float(np.mean(macro)) if macro and usable else float('nan'),'independent macro AUC')
        op=io.independent_operating(q.label,q.score,expected)
        for key,value in op.items():io._equal(row[key],value,'independent '+key)
        mt=float(np.mean(macro_tpr)) if macro_tpr else float('nan')
        io._equal(row['macro_tpr'],mt,'independent macro TPR')
        io._equal(row['macro_ba'],(1-op['fpr']+mt)/2,'independent macro BA')
        checks.append(dict(cell=cell,threshold_kind=row['threshold_kind'],status='PASS' if usable else 'PASS_OPERATING_ONLY_NUMERIC_AUC_NA'))
    if set(metrics.cell)!=set(cells):raise ValueError('independent metric cell inventory changed')
    return checks


def _code_identities():
    here=Path(__file__).parent
    names=('analysis_h.py','cache_h.py','stats_h.py','coverage_h.py','analysis_d.py','scoring_d.py','bootstrap_fast.py','cache_d.py')
    return {name:digest(here/name) for name in names}


def _pair_context(config_path,backbone,seed):
    from . import cache_h,data_h
    from .queue_h import formal_id
    if backbone not in BACKBONES or seed not in SEEDS:raise ValueError('unregistered H backbone/formal seed')
    path=Path(config_path).resolve();config=json.loads(path.read_text());root=Path(config['artifact_root']).resolve()
    if path!=root/'configs/H_execution.json' or config.get('schema')!='H_EXECUTION_V2':raise ValueError('dedicated frozen H configuration required')
    freeze=data_h.validate_h_freeze(root)
    registry_path=root/'H_run_registry.json';registry=json.loads(registry_path.read_text())
    if registry.get('status')!='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING' or registry.get('target_export_allowed') is not True:
        raise ValueError('all H formal pairs/effective precision must be frozen before analysis')
    effective=registry['effective_runs'][backbone];precision=effective['precision']
    if precision not in ('AMP','FP32'):raise ValueError('unregistered H effective precision')
    expected=[formal_id(backbone,h,s,precision) for s in SEEDS for h in ['B','AB']]
    if effective['run_ids']!=expected:raise ValueError('H effective registry must contain all five exact pairs')
    gate_path=root/f'qualification/H_target_export_{backbone}.json';gate=json.loads(gate_path.read_text())
    if (gate.get('status')!='PASS' or gate.get('all_five_pairs_checked') is not True or gate.get('effective_runs')!=expected
        or gate.get('precision')!=precision or gate.get('backbone')!=backbone or gate.get('target_performance_used') is not False):
        raise ValueError('H all-five-pair target gate/effective precision mismatch')
    ledger=Path(gate['paired_ledger_path']).resolve()
    if ledger!=root/f'reports/H_{backbone}_{precision}_paired_ledger.json' or digest(ledger)!=gate['paired_ledger_sha256']:
        raise ValueError('H paired numerical ledger binding changed')
    run_ids={h:formal_id(backbone,h,seed,precision) for h in ['B','AB']}
    markers={}
    for history,run in run_ids.items():
        for role in ROLES:
            folder=root/'features/H'/run/role
            cache_h.verify_cache(folder,expected_run_id=run,expected_role=role,config_path=path)
            markers[history+'/'+role]=digest(folder/'complete.json')
    inputs={str(p):digest(p) for p in [path,registry_path,gate_path,ledger,root/'data/H/freeze.json']}
    identity=dict(schema='H_PAIR_ANALYSIS_IDENTITY_V1',backbone=backbone,seed=seed,effective_precision=precision,
        run_ids=run_ids,inputs_sha256=inputs,cache_completion_sha256=markers,code_sha256=_code_identities(),
        bootstrap_repeats=2000,resampling_seed=20261023,source_threshold='each_model_allB_SOURCE_CAL_empiricalFPR5',
        primary='DFD_allB_Raw_mean_logit_H10_minus_H00',test_cohorts='allB_and_common_reported_separately')
    return root,config,freeze,run_ids,identity


def formal_cache_status(root):
    """Read-only availability inventory; absence does not create any artifact."""
    root=Path(root);path=root/'configs/H_execution.json';registry=root/'H_run_registry.json'
    if not path.is_file() or not registry.is_file():
        return dict(status='WAITING_H_FORMAL_CACHES',complete_pairs=0,expected_pairs=15,GPU_launch_attempted=False)
    config=json.loads(path.read_text());record=json.loads(registry.read_text());rows=[]
    from .queue_h import formal_id
    for backbone in BACKBONES:
        effective=record.get('effective_runs',{}).get(backbone)
        for seed in SEEDS:
            ready=False
            if effective:
                runs=[formal_id(backbone,h,seed,effective['precision']) for h in ['B','AB']]
                ready=all((root/'features/H'/run/role/'complete.json').is_file() for run in runs for role in ROLES)
            rows.append(dict(backbone=backbone,seed=seed,markers_present=ready))
    complete=sum(r['markers_present'] for r in rows)
    return dict(status='FORMAL_CACHE_MARKERS_PRESENT_VERIFICATION_REQUIRED' if complete==15 else 'WAITING_H_FORMAL_CACHES',
        complete_pairs=complete,expected_pairs=15,pairs=rows,GPU_launch_attempted=False,marker_presence_is_acceptance=False)


def _analysis_inventory():
    names=['identity.json','coverage.json','independent_saved_score_recompute.json','limitations.json']
    for aggregation in ['mean_logit','mean_probability']:
        names += [f'{aggregation}/source_cal_{h}.csv' for h in ['B','AB']]
        for leaf in ['SOURCE_TEST','TARGET_TEST']:
            for cohort,cells in [('all_B',['H00','H10']),('common',['H00','H01','H10','H11'])]:
                prefix=f'{aggregation}/{leaf}/{cohort}'
                names += [f'{prefix}/{cell}_scores.csv' for cell in cells]
                names += [f'{prefix}/{name}' for name in ['metrics.csv','comparisons.csv','thresholds.json','conditional_bootstrap.json','independent_kernel_checks.json']]
    return names


def _verify_completed(output,identity):
    path=output/'complete.json'
    if not path.is_file():return None
    receipt=json.loads(path.read_text())
    if (receipt.get('schema')!='H_PAIR_ANALYSIS_COMPLETE_V1' or receipt.get('status') not in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
        or receipt.get('independent_recompute')!='PASSED_FOR_ALL_REPORTABLE_CELLS'
        or any(receipt.get(key)!=identity[key] for key in ['backbone','seed','effective_precision'])):
        raise ValueError('H analysis completion schema/status/model identity is invalid')
    if receipt.get('identity')!=identity:raise ValueError('H existing analysis identity changed; register a separate revision')
    inventory=receipt.get('artifact_sha256',{})
    actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p!=path}
    if set(inventory)!=set(_analysis_inventory()) or set(inventory)!=actual:raise ValueError('H completed analysis artifact inventory changed')
    if any(digest(output/name)!=sha for name,sha in inventory.items()):raise ValueError('H committed analysis artifact hash changed')
    return receipt


def analyze_pair(config_path,backbone,seed):
    """Production pair analysis: 2000 common TEST-group draws; CPU only."""
    from . import cache_h,stats_h,coverage_h
    root,config,freeze,runs,identity=_pair_context(config_path,backbone,seed)
    output=root/f'analysis/H/{backbone}/seed{seed}'
    if shutil.disk_usage(root).free <=50*1024**3:raise RuntimeError('H analysis requires50GiB storage reserve; preserve existing artifacts')
    with exclusive_file(root/f'analysis/H/.locks/{backbone}_seed{seed}.lock'):
        complete=_verify_completed(output,identity)
        if complete:return complete
        output.mkdir(parents=True,exist_ok=True);identity_path=output/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise ValueError('H partial analysis identity changed; preserve and register a separate revision')
        io._save_json(identity_path,identity)
        data={history:{} for history in runs}
        for history,run in runs.items():
            for role in ROLES:
                data[history][role]=cache_h.load_features(root/'features/H'/run/role,
                    expected_run_id=run,expected_role=role,config_path=config_path)
        independent=[];numerical_na=[]
        for aggregation in ['mean_logit','mean_probability']:
            pools=score_pools(data,aggregation=aggregation)
            for history,q in pools['source_cal'].items():io.write_scores(output/f'{aggregation}/source_cal_{history}.csv',q)
            restored_cal={h:io.read_scores(output/f'{aggregation}/source_cal_{h}.csv') for h in runs}
            for leaf,methods in [('SOURCE_TEST',SOURCE_METHODS),('TARGET_TEST',('DeepFakeDetection',))]:
                for cohort,cells in pools[leaf].items():
                    folder=output/aggregation/leaf/cohort;folder.mkdir(parents=True,exist_ok=True)
                    for cell,q in cells.items():io.write_scores(folder/f'{cell}_scores.csv',q)
                    restored={cell:io.read_scores(folder/f'{cell}_scores.csv') for cell in cells}
                    result=stats_h.analyze_cohort(restored,restored_cal,fixed_methods=methods,aggregation=aggregation,
                        repeats=2000,seed=20261023)
                    io._csv(folder/'metrics.csv',result['point_metrics']);io._csv(folder/'comparisons.csv',result['comparisons'])
                    io._save_json(folder/'thresholds.json',result['thresholds'])
                    checks=verify_saved_metrics(restored,pd.read_csv(folder/'metrics.csv',float_precision='round_trip'),
                        json.loads((folder/'thresholds.json').read_text()),fixed_methods=methods)
                    independent.append(dict(role=leaf,cohort=cohort,aggregation=aggregation,checks=checks))
                    numerical_na += [dict(role=leaf,cohort=cohort,aggregation=aggregation,**r) for r in checks if r['status']!='PASS']
                    io._save_json(folder/'conditional_bootstrap.json',result['bootstrap'])
                    io._save_json(folder/'independent_kernel_checks.json',result['independent_checks'])
        io._save_json(output/'coverage.json',coverage_h.report_coverage(root))
        io._save_json(output/'independent_saved_score_recompute.json',independent)
        io._save_json(output/'limitations.json',dict(independence_scope=freeze['independence_scope'],
            primary='DFD all-B Raw mean-logit H10-H00; FF source validation',
            cohorts='All-B deployment and common 2x2 remain separate; no cross-cohort metric subtraction',
            source_threshold='each model allB SOURCE_CAL only; AB cells reuse B deployment threshold',
            bootstrap='2000 pointwise conditional shared-group TEST draws, not training replicates; proxy ancestry unverified',
            representation='Raw only; fixed original FP32 head promoted to FP64 before subtraction',numerical_NA=numerical_na,
            completed_model_pairs=1,training_seed_count=1,five_seed_summary_pending=True,formal_scientific_results=True))
        # Guard against input or code changes while calculating. It re-verifies
        # every cache, gate and source identity before endorsing the results.
        _,_,_,_,current=_pair_context(config_path,backbone,seed)
        if current!=identity:raise ValueError('H analysis inputs changed during execution')
        artifacts={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='complete.json'}
        receipt=dict(schema='H_PAIR_ANALYSIS_COMPLETE_V1',status='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' if numerical_na else 'ANALYSIS_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),identity=identity,artifact_sha256=artifacts,
            backbone=backbone,seed=seed,effective_precision=identity['effective_precision'],independent_recompute='PASSED_FOR_ALL_REPORTABLE_CELLS')
        if set(artifacts)!=set(_analysis_inventory()):raise ValueError('H analysis expected inventory was not produced')
        io._save_json(output/'complete.json',receipt)
        return receipt


def summarize_backbone(config_path,backbone):
    """Re-read all five accepted pair analyses; never pool backbones as IID."""
    from . import stats_h
    if backbone not in BACKBONES:raise ValueError('unregistered H backbone')
    roots=[];markers={};pair_identities={};metrics=[];comparisons=[]
    for seed in SEEDS:
        root,_,_,_,identity=_pair_context(config_path,backbone,seed)
        folder=root/f'analysis/H/{backbone}/seed{seed}'
        receipt=_verify_completed(folder,identity)
        if receipt is None:raise ValueError('incomplete H pair analyses; all five registered seeds required')
        roots.append(root);markers[str(seed)]=digest(folder/'complete.json');pair_identities[seed]=identity
        for aggregation in ['mean_logit','mean_probability']:
            for domain in ['SOURCE_TEST','TARGET_TEST']:
                for cohort in ['all_B','common']:
                    base=folder/aggregation/domain/cohort
                    for filename,destination in [('metrics.csv',metrics),('comparisons.csv',comparisons)]:
                        q=pd.read_csv(base/filename,float_precision='round_trip')
                        q=q.assign(seed=seed,backbone=backbone,domain=domain,cohort=cohort,aggregation=aggregation)
                        destination.append(q)
    if len(set(roots))!=1:raise ValueError('H five-seed artifact roots differ')
    root=roots[0];output=root/f'analysis/H/{backbone}/five_seed_summary'
    identity=dict(schema='H_FIVE_SEED_SUMMARY_IDENTITY_V1',backbone=backbone,seeds=list(SEEDS),
        pair_complete_sha256=markers,code_sha256=_code_identities())
    inventory={'identity.json','per_seed_metrics.csv','per_seed_comparisons.csv','five_seed_metrics.csv','five_seed_comparisons.csv','limitations.json'}
    with exclusive_file(root/f'analysis/H/.locks/{backbone}_five_seed_summary.lock'):
        complete=output/'complete.json'
        if complete.exists():
            receipt=json.loads(complete.read_text())
            if (receipt.get('schema')!='H_FIVE_SEED_SUMMARY_COMPLETE_V1' or receipt.get('status')!='SUMMARY_COMPLETE'
                or receipt.get('identity')!=identity or receipt.get('training_seed_count')!=5 or receipt.get('backbone')!=backbone):
                raise ValueError('H five-seed summary schema/identity changed')
            hashes=receipt.get('artifact_sha256',{})
            actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p!=complete}
            if set(hashes)!=inventory or actual!=inventory:raise ValueError('H five-seed summary artifact inventory changed')
            if any(digest(output/name)!=value for name,value in hashes.items()):raise ValueError('H five-seed summary artifact hash changed')
            return receipt
        if shutil.disk_usage(root).free<=50*1024**3:raise RuntimeError('H summary requires50GiB reserve')
        output.mkdir(parents=True,exist_ok=True)
        prior=output/'identity.json'
        if prior.exists() and json.loads(prior.read_text())!=identity:raise ValueError('H partial five-seed identity changed')
        io._save_json(prior,identity)
        m=pd.concat(metrics,ignore_index=True);c=pd.concat(comparisons,ignore_index=True)
        io._csv(output/'per_seed_metrics.csv',m);io._csv(output/'per_seed_comparisons.csv',c)
        cs=stats_h.summarize_seed_comparisons(c)
        melted=m.melt(id_vars=['seed','backbone','domain','cohort','aggregation','cell','threshold_kind'],
            value_vars=['pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba'],var_name='metric',value_name='estimate')
        melted['comparison']=melted.pop('cell')
        ms=stats_h.summarize_seed_comparisons(melted).rename(columns={'comparison':'cell'})
        io._csv(output/'five_seed_metrics.csv',ms);io._csv(output/'five_seed_comparisons.csv',cs)
        io._save_json(output/'limitations.json',dict(training_seed_count=5,sample_sd_ddof=1,backbone=backbone,
            model_count='five matched training-seed pairs; two histories per pair',
            inference='descriptive across registered five seeds; TEST bootstrap intervals remain per-seed conditional',
            cohorts='all-B and common never subtracted across cohorts',
            direction_counts='descriptive only, not significance tests',missing_estimates='five-seed mean/SD unavailable if any seed estimate missing'))
        for seed in SEEDS:
            _,_,_,_,current_identity=_pair_context(config_path,backbone,seed)
            if current_identity!=pair_identities[seed]:raise ValueError('H pair inputs changed during summary')
            _verify_completed(root/f'analysis/H/{backbone}/seed{seed}',current_identity)
            if digest(root/f'analysis/H/{backbone}/seed{seed}/complete.json')!=markers[str(seed)]:
                raise ValueError('H pair inputs changed during summary')
        if _code_identities()!=identity['code_sha256']:raise ValueError('H analysis code changed during summary')
        hashes={name:digest(output/name) for name in inventory}
        receipt=dict(schema='H_FIVE_SEED_SUMMARY_COMPLETE_V1',status='SUMMARY_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),identity=identity,artifact_sha256=hashes,backbone=backbone,training_seed_count=5)
        io._save_json(complete,receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('status','pair','summary'))
    parser.add_argument('--root');parser.add_argument('--config');parser.add_argument('--backbone',choices=BACKBONES);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action=='status':
        if not args.root:parser.error('status requires --root')
        result=formal_cache_status(args.root)
    elif args.action=='pair':
        if not args.config or not args.backbone or args.seed is None:parser.error('pair requires --config --backbone --seed')
        result=analyze_pair(args.config,args.backbone,args.seed)
    else:
        if not args.config or not args.backbone:parser.error('summary requires --config --backbone')
        result=summarize_backbone(args.config,args.backbone)
    print(json.dumps(io._jsonable(result),indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
