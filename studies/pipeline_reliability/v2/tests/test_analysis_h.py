import copy
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST',
       'common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST')


def api():
    assert importlib.util.find_spec('reliability_v2.analysis_h') is not None, 'H analysis implementation is missing'
    return importlib.import_module('reliability_v2.analysis_h')


def cache_fixture():
    data={history:{} for history in ['B','AB']}
    for history in data:
        for role in ROLES:
            leaf=role.split('/')[-1];version='AB' if role.startswith('common/AB/') else 'B'
            domain='target' if leaf=='TARGET_TEST' else 'source'
            ids=['c0','c1'] if leaf=='SOURCE_CAL' else ['r0','r1','f0','f1']
            if role.startswith('common/'):ids=['r0','f0']
            rows=[];features=[]
            for vid in ids:
                label=int(vid.startswith('f'));uid=f'{domain}/{leaf}/{vid}'
                for frame in range(8):
                    sample=f'{uid}/{frame}'
                    rows.append(dict(image_path=f'{sample}/{version}.png',label=label,method_id='DeepFakeDetection' if label and domain=='target' else 'Deepfakes' if label else 'real',
                        video_id=vid,video_uid=uid,group_id=f'{domain}/{vid}',domain=domain,role=leaf,rgb_sha256=f'{sample}/{version}',
                        mother_id=sample,sample_id=sample,mother_rgb_sha256=f'{sample}/mother',processing_version=version,content_sha256=f'{sample}/{version}'))
                    features.append([float(label)*2-1+(history=='AB')*.3+(version=='AB')*.2,1.])
            w=np.array([[0.,0.],[1.,0.]],dtype=np.float32);b=np.zeros(2,dtype=np.float32)
            data[history][role]=(pd.DataFrame(rows),np.array(features,dtype=np.float32),w,b)
    return data


def test_scores_keep_all_B_and_common_cohorts_separate_and_raw_only():
    result=api().score_pools(cache_fixture(),aggregation='mean_logit')
    assert set(result['source_cal'])=={'B','AB'}
    for role in ['SOURCE_TEST','TARGET_TEST']:
        assert set(result[role]['all_B'])=={'H00','H10'}
        assert set(result[role]['common'])=={'H00','H01','H10','H11'}
        assert len(result[role]['all_B']['H00'])==4 and len(result[role]['common']['H00'])==2
        for cohort in result[role].values():
            for videos in cohort.values():assert videos.representation.eq('raw').all()
    assert result['source_cal']['B'].label.eq(0).all()
    assert np.isclose(result['SOURCE_TEST']['common']['H00'].score.iloc[0],-1)
    assert np.isclose(result['SOURCE_TEST']['common']['H10'].score.iloc[0],-.7)
    assert np.isclose(result['SOURCE_TEST']['common']['H01'].score.iloc[0],-.8)


@pytest.mark.parametrize('mutation',['role','domain','history_version','head','cal_fake','mother','allB_subset','missing_role'])
def test_score_pool_provenance_and_common_mothers_fail_closed(mutation):
    data=cache_fixture();frame,h,w,b=data['AB']['common/AB/SOURCE_TEST']
    if mutation=='role':frame.loc[0,'role']='SOURCE_CAL'
    elif mutation=='domain':frame['domain']='target'
    elif mutation=='history_version':frame['processing_version']='B'
    elif mutation=='head':w[1,0]=2
    elif mutation=='cal_fake':data['B']['B/SOURCE_CAL'][0].loc[0,'label']=1
    elif mutation=='mother':frame.loc[0,'mother_id']='another_mother'
    elif mutation=='allB_subset':data['B']['common/B/SOURCE_TEST'][0].loc[0,'content_sha256']='changed_crop'
    else:del data['B']['B/SOURCE_CAL']
    with pytest.raises(ValueError):api().score_pools(data)


def test_one_original_head_per_model_across_all_roles():
    data=cache_fixture();data['B']['B/TARGET_TEST'][3][0]=.5
    with pytest.raises(ValueError,match='head'):api().score_pools(data)


def test_readiness_on_missing_H_execution_is_read_only(tmp_path):
    result=api().formal_cache_status(tmp_path)
    assert result['status']=='WAITING_H_FORMAL_CACHES' and result['complete_pairs']==0
    assert result['GPU_launch_attempted'] is False
    assert not list(tmp_path.iterdir())


def test_unregistered_pair_rejected_before_any_cache_read(tmp_path,monkeypatch):
    module=api()
    with pytest.raises((ValueError,FileNotFoundError)):
        module.analyze_pair(tmp_path/'configs/H_execution.json','clip_vit_b16',1007)
    assert not (tmp_path/'analysis').exists()


def test_precision_safe_saved_score_roundtrip_and_metric_tamper_detection(tmp_path):
    from reliability_v2 import analysis_d,scoring_d
    module=api();pools=module.score_pools(cache_fixture())
    cells=pools['TARGET_TEST']['common'];cal=pools['source_cal']
    from reliability_v2.stats_h import analyze_cohort
    result=analyze_cohort(cells,cal,fixed_methods=['DeepFakeDetection'],repeats=10,min_groups=1,max_group_fraction=1)
    output=tmp_path/'scores';output.mkdir()
    restored={}
    for cell,part in cells.items():
        path=output/f'{cell}.csv';analysis_d.write_scores(path,part);restored[cell]=analysis_d.read_scores(path)
    module.verify_saved_metrics(restored,result['point_metrics'],result['thresholds'],fixed_methods=['DeepFakeDetection'])
    bad=result['point_metrics'].copy();bad.loc[0,'pooled_auc']=.123
    with pytest.raises(ValueError,match='independent|metric|AUC'):
        module.verify_saved_metrics(restored,bad,result['thresholds'],fixed_methods=['DeepFakeDetection'])


def test_truncated_self_consistent_analysis_completion_is_rejected(tmp_path):
    module=api();identity={'backbone':'clip_vit_b16','seed':7,'effective_precision':'AMP'}
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    (tmp_path/'complete.json').write_text(json.dumps(dict(identity=identity,
        artifact_sha256={'identity.json':module.digest(tmp_path/'identity.json')})))
    with pytest.raises(ValueError,match='schema|status|inventory'):module._verify_completed(tmp_path,identity)


def driver_fixture(tmp_path,monkeypatch):
    module=api();from reliability_v2 import cache_h,coverage_h
    data=cache_fixture()
    # Include each registered FF fake method so macro metrics exercise the real
    # fixed registry. Source ancestry groups remain shared across manipulations.
    for history,roles in data.items():
        for role,(frame,h,w,b) in list(roles.items()):
            if not role.endswith('SOURCE_TEST'):continue
            source=frame[frame.video_id.eq('f0')];features=h[frame.video_id.eq('f0')]
            additions=[]
            for method in module.SOURCE_METHODS[1:]:
                q=source.copy();q['method_id']=method;q['video_uid']=q.video_uid+'/'+method
                q['mother_id']=q.mother_id+'/'+method;q['sample_id']=q.mother_id
                additions.append(q)
            roles[role]=(pd.concat([frame,*additions],ignore_index=True),np.concatenate([h,*[features]*len(additions)]),w,b)
    p=tmp_path/'configs/H_execution.json';p.parent.mkdir();p.write_text(json.dumps({'artifact_root':str(tmp_path)}))
    calls=[]
    def context(path,backbone,seed):
        if seed not in module.SEEDS:raise ValueError('unregistered')
        ids={hist:f'H_formal_{backbone}_{hist}_seed{seed}' for hist in ['B','AB']}
        identity=dict(backbone=backbone,seed=seed,effective_precision='AMP',synthetic=True)
        return tmp_path,{},dict(independence_scope='SYNTHETIC_TEST_ONLY'),ids,identity
    def load(folder,*,expected_run_id,expected_role,config_path):
        assert config_path==p
        assert folder==tmp_path/'features/H'/expected_run_id/expected_role
        history='AB' if '_AB_' in expected_run_id else 'B'
        calls.append((expected_run_id,expected_role));return data[history][expected_role]
    monkeypatch.setattr(module,'_pair_context',context);monkeypatch.setattr(cache_h,'load_features',load)
    monkeypatch.setattr(coverage_h,'report_coverage',lambda root:dict(scope='SYNTHETIC_TEST_ONLY'))
    return module,p,calls


def test_complete_driver_saved_scores_recompute_and_reuse_without_reload(tmp_path,monkeypatch):
    module,p,calls=driver_fixture(tmp_path,monkeypatch)
    first=module.analyze_pair(p,'clip_vit_b16',7)
    assert first['status']=='ANALYSIS_COMPLETE' and len(calls)==14
    assert set(first['artifact_sha256'])==set(module._analysis_inventory())
    second=module.analyze_pair(p,'clip_vit_b16',7)
    assert second==first and len(calls)==14
    out=tmp_path/'analysis/H/clip_vit_b16/seed7';metrics=out/'mean_logit/TARGET_TEST/all_B/metrics.csv'
    altered=pd.read_csv(metrics);altered.loc[0,'fpr']=.123;altered.to_csv(metrics,index=False)
    with pytest.raises(ValueError,match='artifact hash'):module.analyze_pair(p,'clip_vit_b16',7)


def test_five_seed_summary_requires_all_pairs_and_keeps_cohorts_separate(tmp_path,monkeypatch):
    module,p,calls=driver_fixture(tmp_path,monkeypatch)
    module.analyze_pair(p,'clip_vit_b16',7)
    with pytest.raises(ValueError,match='five|incomplete'):module.summarize_backbone(p,'clip_vit_b16')
    for seed in module.SEEDS[1:]:module.analyze_pair(p,'clip_vit_b16',seed)
    result=module.summarize_backbone(p,'clip_vit_b16')
    assert result['training_seed_count']==5 and result['backbone']=='clip_vit_b16'
    folder=tmp_path/'analysis/H/clip_vit_b16/five_seed_summary'
    table=pd.read_csv(folder/'five_seed_comparisons.csv')
    assert set(table.cohort)=={'all_B','common'} and set(table.domain)=={'SOURCE_TEST','TARGET_TEST'}
    assert table.n_seeds.eq(5).all() and table.estimate_sample_sd.eq(0).all()
    before=len(calls);assert module.summarize_backbone(p,'clip_vit_b16')==result and len(calls)==before
    # A file changing after the summary's first read must not be endorsed by
    # merely observing that its parent completion marker is unchanged.
    from reliability_v2 import stats_h
    (folder/'complete.json').unlink()
    original=stats_h.summarize_seed_comparisons;changed=[]
    def mutate_after_read(rows,**kwargs):
        value=original(rows,**kwargs)
        if not changed:
            path=tmp_path/'analysis/H/clip_vit_b16/seed7/mean_logit/TARGET_TEST/all_B/metrics.csv'
            q=pd.read_csv(path);q.loc[0,'fpr']=.123;q.to_csv(path,index=False);changed.append(True)
        return value
    monkeypatch.setattr(stats_h,'summarize_seed_comparisons',mutate_after_read)
    with pytest.raises(ValueError,match='artifact hash|changed during summary'):
        module.summarize_backbone(p,'clip_vit_b16')
    assert not(folder/'complete.json').exists()
