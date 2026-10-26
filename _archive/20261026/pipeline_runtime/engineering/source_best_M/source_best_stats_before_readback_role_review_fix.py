"""Selected-M four-core statistical orchestration, using original kernels.

Low-level CPU drivers consume already-qualified arrays and an explicit sink that
returns re-read values. They do not load artifacts, select models/threshold rules,
issue permits, dispatch jobs or declare production/scientific admission. The
compact persistent sink, cache-admission/CPU-slot/budget entry remain separate.
"""
import numpy as np
import pandas as pd
from . import analysis_d as io,analysis_h as original,scoring_d as score,bootstrap_fast as boot,stats_h,source_best_cache as cache

SEEDS=(7,42,123,2026,3407)
TEST_REPEATS=2000
CAL_REPEATS=200
RESAMPLING_SEED=20261023
AGGREGATIONS=('mean_logit','mean_probability')

def require(ok,why):
    if not ok:raise ValueError(why)

def selection(meta,route):
    require(meta.get('route')==route and meta.get('checkpoint_selection')=='source_best' and meta.get('source_best_bytes_tensors_head_verified') is True,'verified source-only source-best selection required')
    require(type(meta.get('seed')) is int and meta['seed'] in SEEDS,'all registered five seeds retained')
    best=meta['best'];require(type(best['epoch']) is int and 1<=best['epoch']<=meta['registered_terminal_epochs'],'selected source epoch outside registered formal budget')
    require(all(type(best[x]) is int and best[x]>=0 for x in ('attempts','successful_updates','AMP_skips')) and best['successful_updates']+best['AMP_skips']==best['attempts'],'selected prefix attempted/success/skip counting mismatch')
    return dict(seed=meta['seed'],epoch=best['epoch'],attempts=best['attempts'],successful_updates=best['successful_updates'],AMP_skips=best['AMP_skips'])

def _heads(data,expected=None):
    first=None
    for role,(frame,h,w,b) in data.items():
        require(np.asarray(h).ndim==2 and len(frame)==len(h) and np.asarray(h).dtype==np.float32,'raw FP32 feature rows required')
        cache.cache_d._validate_features(h,len(frame),h.shape[1]);head=cache._head_hash(w,b,h.shape[1])
        require(first is None or head==first,'one unchanged original head per model across roles');first=head
    require(expected is None or first==expected,'selected source-best original head mismatch')
    return first

def native_H_pools(data,*,aggregation='mean_logit'):
    require(aggregation in AGGREGATIONS and set(data)=={'B','AB'} and all(set(roles)==set(cache.ROLES['H']) for roles in data.values()),'exact selected H native-B roles and two trained histories required')
    videos={history:{} for history in data}
    for history,roles in data.items():
        _heads(roles)
        for role,(frame,h,w,b) in roles.items():
            leaf=role.split('/')[-1];domain='target' if leaf=='TARGET_TEST' else 'source'
            require(frame.role.eq(leaf).all() and frame.domain.eq(domain).all() and frame.processing_version.eq('B').all(),'H native role/domain/pixel version mismatch')
            if leaf=='SOURCE_CAL':require(frame.label.eq(0).all(),'H source calibration must be real-only')
            margins=score.frame_margins(frame,h,w,b)
            videos[history][role]=score.aggregate_videos(margins,representation='raw',aggregation=aggregation)
    for role in cache.ROLES['H']:original._same_inputs(data['B'][role][0],data['AB'][role][0])
    return dict(source_cal={h:videos[h]['B/SOURCE_CAL'] for h in ('B','AB')},
                SOURCE_TEST={'H00':videos['B']['B/SOURCE_TEST'],'H10':videos['AB']['B/SOURCE_TEST']},
                TARGET_TEST={'H00':videos['B']['B/TARGET_TEST'],'H10':videos['AB']['B/TARGET_TEST']})

def h_driver(data,selections,emit):
    require(set(selections)=={'B','AB'},'both original model history selections required')
    disclosed={h:selection(selections[h],'H') for h in selections}
    require(disclosed['B']['seed']==disclosed['AB']['seed'] and selections['B']['history']=='B' and selections['AB']['history']=='AB','registered matched seed/history metadata required')
    for h in selections:_heads(data[h],selections[h]['head_sha256'])
    for aggregation in AGGREGATIONS:
        pools=native_H_pools(data,aggregation=aggregation)
        cal={h:emit(f'{aggregation}/source_cal_{h}.csv','scores',q) for h,q in pools['source_cal'].items()}
        for domain,methods in [('SOURCE_TEST',original.SOURCE_METHODS),('TARGET_TEST',('DeepFakeDetection',))]:
            folder=f'{aggregation}/{domain}/all_B'
            cells={cell:emit(f'{folder}/{cell}_scores.csv','scores',q) for cell,q in pools[domain].items()}
            result=stats_h.analyze_cohort(cells,cal,fixed_methods=methods,aggregation=aggregation,repeats=TEST_REPEATS,seed=RESAMPLING_SEED)
            metrics=emit(f'{folder}/metrics.csv','csv',result['point_metrics'])
            emit(f'{folder}/comparisons.csv','csv',result['comparisons'])
            thresholds=emit(f'{folder}/thresholds.json','json',result['thresholds'])
            checks=original.verify_saved_metrics(cells,metrics,thresholds,fixed_methods=methods)
            emit(f'{folder}/independent_saved_score_checks.json','json',checks)
            emit(f'{folder}/conditional_bootstrap.json','json',result['bootstrap'])
            emit(f'{folder}/independent_kernel_checks.json','json',result['independent_checks'])
    descriptor=dict(checkpoint_selection='source_best',source_best_epochs={h:q['epoch'] for h,q in disclosed.items()},source_only_selected_prefix=disclosed,update_matched_primary_claim=False,scope='Descriptive checkpoint sensitivity of native-B H10-H00; no common/AB-test 2x2 sensitivity claim',test_bootstrap_repeats=TEST_REPEATS,resampling_seed=RESAMPLING_SEED,source_CAL_resampled=False,training_seeds_resampled=False,production_admission_provided_here=False)
    emit('limitations.json','json',descriptor);return descriptor

def d_driver(data,selected,freeze,emit):
    disclosed=selection(selected,'D')
    require(set(data)==set(cache.ROLES['D']),'exact selected D six roles required; no DEV/foreign inputs')
    require(freeze.get('status')=='FROZEN' and freeze.get('method_count')==21 and freeze.get('target_scores_used') is False,'frozen original D method/source-selection protocol required')
    registry=freeze['method_registry'];require(len(registry['FS9'])==9 and len(registry['FR12'])==12 and len(set(registry['FS9']+registry['FR12']))==21,'original FS9+FR12 registry required')
    methods=sorted(registry['FS9']+registry['FR12']);_heads(data,selected['head_sha256'])
    for role in ('source_cal','target_cal'):require(data[role][0].label.eq(0).all(),'real-only source/target calibration pools required')
    train,h,w,b=data['train'];reference=io.training_reference(train,h)
    emit('training_radii.npy','array',reference['radii']);emit('training_reference.json','json',{k:v for k,v in reference.items() if k not in ('radii','weights')})
    independent=[];unresolved=[]
    for role in ('source_test','target_full'):
        frame,h,weight,bias=data[role]
        result=io.analyze_rank_role(frame,h,w,b,training_radii=reference['radii'],training_weights=reference['weights'],fixed_methods=methods,seed=selected['seed'])
        emit(f'{role}/frame_margins.parquet','parquet',result['frames'])
        videos=emit(f'{role}/video_scores.csv','scores',result['videos']);metrics=emit(f'{role}/metrics.csv','csv',result['metrics'])
        emit(f'{role}/effects.csv','csv',result['effects']);checks=io.verify_saved_metrics(videos,metrics,methods);independent.append(dict(role=role,cells=checks))
        unresolved.extend(dict(role=role,**q) for q in checks if q['status']!='PASSED')
        controls=result['controls'];emit(f'{role}/control_video_scores.csv','scores',controls['controls'])
        for name in ('video_diagnostics','method_diagnostics','numerical_audits'):emit(f'{role}/{name}.csv','csv',controls[name])
        unresolved.extend(dict(role=role,analysis='numerical_control',**q) for q in controls['numerical_audits'].to_dict('records') if not q.get('ranking_usable',True))
        for keys,part in controls['controls'].groupby(['control','grouping','representation','aggregation','scale'],sort=False):
            if not bool(part.ranking_usable.all()):unresolved.append(dict(role=role,analysis='control_scores',**dict(zip(('control','grouping','representation','aggregation','scale'),keys))))
        emit(f'{role}/pair_transition_ledgers.json','json',controls['transition_ledgers'])
        emit(f'{role}/closure_bootstrap.json','json',boot.bootstrap_d_fast(result['frames'],fixed_methods=methods,repeats=TEST_REPEATS,seed=RESAMPLING_SEED))
        for representation in ('raw','l2'):
            for aggregation in AGGREGATIONS:
                q=result['videos'].query('grouping == "correct" and representation == @representation and aggregation == @aggregation')
                emit(f'{role}/normal_bootstrap_{representation}_{aggregation}.json','json',boot.bootstrap_video_metrics_fast(q,fixed_methods=methods,repeats=TEST_REPEATS,seed=RESAMPLING_SEED))
    margins={}
    for role in ('source_cal','target_cal','target_test'):
        frame,h,weight,bias=data[role];margins[role]=score.frame_margins(frame,h,w,b);emit(f'rq4/{role}_frame_margins.parquet','parquet',margins[role])
    emit('rq4/scope.json','json',freeze['cdf_calibration_scope'])
    for representation in ('raw','l2'):
        for aggregation in AGGREGATIONS:
            folder=f'rq4/{representation}_{aggregation}'
            pools={role:emit(f'{folder}/{role}_scores.csv','scores',score.aggregate_videos(q,representation=representation,aggregation=aggregation)) for role,q in margins.items()}
            rows=score.calibration_experiment(pools['source_cal'],pools['target_cal'],pools['target_test'],aggregation=aggregation,fixed_methods=methods,budgets=(10,20,40,80),repeats=CAL_REPEATS,seed=RESAMPLING_SEED)
            saved=emit(f'{folder}/calibration.csv','csv',rows)
            emit(f'{folder}/independent_calibration_recompute.json','json',io._independent_calibration(pools['target_test'],saved,methods))
            source_threshold=rows.loc[rows.calibration_kind.eq('source_real'),'threshold'].item();target_thresholds=rows.loc[rows.calibration_kind.eq('target_real') & rows.budget.eq(20) & rows.status.eq('ok'),'threshold'].tolist()
            primary=boot.bootstrap_operating_comparison(pools['target_test'],source_threshold=source_threshold,target_thresholds=target_thresholds,repeats=TEST_REPEATS,seed=RESAMPLING_SEED,calibration_budget=20) if target_thresholds else dict(status='UNAVAILABLE_CALIBRATION_BUDGET',calibration_budget=20,available_target_real_videos=len(pools['target_cal']),intervals=None)
            emit(f'{folder}/primary_n20_vs_source_bootstrap.json','json',primary)
    emit('independent_rank_recomputation.json','json',independent)
    descriptor=dict(checkpoint_selection='source_best',source_best_epoch=disclosed['epoch'],source_only_selected_prefix=disclosed,update_matched_primary_claim=False,scope='Four fixed core sensitivity comparisons; original D8cells/both aggregations/radius/numerical/order controls and CAL/TEST rules retained',target_scope=freeze['cdf_calibration_scope'],numerical_unresolved=unresolved,calibration_repeats=CAL_REPEATS,test_bootstrap_repeats=TEST_REPEATS,resampling_seed=RESAMPLING_SEED,production_admission_provided_here=False)
    emit('limitations.json','json',descriptor);return descriptor
