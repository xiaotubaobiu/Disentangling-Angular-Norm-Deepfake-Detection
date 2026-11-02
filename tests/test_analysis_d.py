import numpy as np
import pandas as pd
import pytest
from journal_detector.analysis_d import (independent_auc, independent_operating,
    training_reference, analyze_rank_role, write_scores, read_scores,
    verify_saved_metrics, formal_cache_status)


def fixture_frames(role='target_full', domain='cdf'):
    rows=[];features=[]
    for method,video,label,group,kind in [('real','r0',0,'g0',0),('real','r1',0,'g1',1),
        ('A','p0',1,'g0',2),('B','p0',1,'g1',3)]:
        uid=f'DF40/{domain}/{method}/{video}'
        for i in range(8):
            rows.append(dict(image_path=f'/{uid}/{i}.png',label=label,method_id=method,
                video_id=video,video_uid=uid,group_id=group,domain=domain,role=role,rgb_sha256=f'{uid}/{i}'))
            x=([-1-i*.01,0] if kind==0 else [1+i*.01,0] if kind==1 else
                [.5+i*.001,np.sqrt(1-(.5+i*.001)**2)] if kind==2 else [10+i*.01,0])
            features.append(x)
    return pd.DataFrame(rows),np.array(features,dtype=np.float32),np.array([[0.,0.],[1.,0.]],dtype=np.float32),np.zeros(2,dtype=np.float32)


def test_independent_integer_concordance_and_small_sample_operating_point():
    assert independent_auc([0,0,1,1],[0.,1.,1.,2.])==pytest.approx(.875)
    m=independent_operating([0]*97+[1]*2,[0.]*93+[1.]*4+[1.,0.],1.)
    assert m['false_positives']==4 and m['fpr']==pytest.approx(4/97)
    assert m['pooled_ba']==pytest.approx((93/97+.5)/2)


def test_training_reference_accepts_canonical_rows_without_eight_frame_groups():
    h=np.array([[3.,4.],[0.,2.],[1.,0.]],dtype=np.float32)
    train=pd.DataFrame(dict(label=[0,1,1],sampling_mass=[.5,.25,.25]))
    r=training_reference(train,h)
    assert r['weighted_radius_median']==2. and r['canonical_rows']==3


def test_eight_cells_roundtrip_and_independent_recomputation(tmp_path):
    f,h,w,b=fixture_frames()
    result=analyze_rank_role(f,h,w,b,training_radii=np.array([1.,2.]),training_weights=np.array([.5,.5]),fixed_methods=['A','B'],seed=7)
    assert len(result['metrics'])==8 and len(result['effects'])==2
    assert len(result['controls']['transition_ledgers'])==8
    path=tmp_path/'scores.csv';write_scores(path,result['videos'])
    restored=read_scores(path)
    verified=verify_saved_metrics(restored,result['metrics'],['A','B'])
    assert len(verified)==8 and max(r['pooled_auc_abs_error'] for r in verified)==0
    correct=restored.query('grouping=="correct" and representation=="raw" and aggregation=="mean_logit"')
    collision=restored.query('grouping=="collision" and representation=="raw" and aggregation=="mean_logit"')
    assert len(correct)==4 and len(collision)==3
    assert collision.loc[collision.label.eq(1),'n_frames'].item()==16
    changed=result['metrics'].copy();changed.loc[0,'pooled_auc']+=.01
    with pytest.raises(ValueError,match='independent AUC'):verify_saved_metrics(restored,changed,['A','B'])


def test_status_never_counts_development_as_formal_cache(tmp_path):
    import json
    config={'artifact_root':str(tmp_path),'runs':{'D_dev':{'phase':'development','recipe_frozen':False}}}
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    report=formal_cache_status(path)
    assert report['formal_registered']==0 and report['formal_caches_complete']==0
    assert report['expected_formal_runs']==15 and report['status']=='WAITING_FORMAL_CACHES'


def test_precise_score_roundtrip_and_oracle_attainable_vertices(tmp_path):
    import mpmath as mp
    from journal_detector.analysis_d import oracle_points
    from journal_detector import scoring_d
    with mp.workdps(320):
        values=[mp.mpf(0),mp.mpf(1)-mp.mpf('1e-200'),mp.mpf(1)-mp.mpf('1e-210'),mp.mpf(1)]
    q=pd.DataFrame(dict(video_uid=['r0','r1','f0','f1'],label=[0,0,1,1],
        method_id=['real','real','A','B'],score=[0.,1.,1.,1.],rank_score=values,
        ranking_usable=True))
    write_scores(tmp_path/'hp.csv',q);restored=read_scores(tmp_path/'hp.csv')
    assert independent_auc(restored.label,restored.rank_score)==1.
    a=oracle_points(restored,fixed_methods=['A','B'])
    b=scoring_d.oracle_operating_points(restored,fixed_methods=['A','B'])
    assert a.criterion.tolist()==b.criterion.tolist()
    for column in ['pooled_auc','macro_auc','fpr','tpr','pooled_ba','false_positives','n_actual_vertices']:
        assert a[column].tolist()==b[column].tolist()
    assert a.threshold.tolist()==b.threshold.tolist()


def test_sparse_mapping_export_and_unresolved_auc_remains_unavailable(tmp_path):
    from collections import UserDict
    from journal_detector.analysis_d import _save_json
    import json
    p=tmp_path/'draw.json';_save_json(p,{'multiplicity':UserDict({'g0':2,'g2':1}), 'unresolved':float('nan')})
    assert json.loads(p.read_text())=={'multiplicity':{'g0':2,'g2':1},'unresolved':None}
    f,h,w,b=fixture_frames()
    result=analyze_rank_role(f,h,w,b,training_radii=np.array([1.,2.]),training_weights=np.array([.5,.5]),fixed_methods=['A','B'],seed=7)
    mask=result['videos'].grouping.eq('correct') & result['videos'].representation.eq('raw') & result['videos'].aggregation.eq('mean_probability')
    result['videos'].loc[mask,'ranking_usable']=False
    m=result['metrics'];cell=m.grouping.eq('correct') & m.representation.eq('raw') & m.aggregation.eq('mean_probability')
    m.loc[cell,['pooled_auc','macro_auc']]=float('nan');m.loc[cell,'macro_valid']=False
    audit=verify_saved_metrics(result['videos'],m,['A','B'])
    assert sum(r['status']=='UNRESOLVED_NUMERIC_ORDERING' for r in audit)==1
    assert [r for r in audit if r['status']=='UNRESOLVED_NUMERIC_ORDERING'][0]['pooled_auc_abs_error'] is None


def test_full_driver_synthetic_caches_and_resume_integrity(tmp_path,monkeypatch):
    """Exercise all writes/reloads with synthetic caches, never a real model."""
    import json
    from pathlib import Path
    from journal_detector import analysis_d,cache_d
    root=tmp_path/'synthetic';data=root/'data/D';data.mkdir(parents=True)
    run_id='D_formal_SYNTHETIC_ONLY'
    cache=root/'features/D'/run_id;cache.mkdir(parents=True)
    (cache/'complete.json').write_text(json.dumps({'status':'CACHE_COMPLETE'}))
    roles={}
    for role in cache_d.ROLES:
        domain='ff' if role in ('train','dev','source_cal','source_test') else 'cdf'
        f,h,w,b=fixture_frames(role=role,domain=domain)
        if role in ('source_cal','target_cal'):
            mask=f.label.eq(0);h=h[mask];f=f[mask].copy()
            f['video_uid']=role+'/'+f.video_uid
        if role=='train':f['sampling_mass']=1/len(f)
        f.to_parquet(data/f'{role}.parquet',index=False)
        roles[role]=(f,h,w,b)
    freeze={'status':'FROZEN','method_count':21,
        'method_registry':{'FS9':['A','B']+[f'X{i}' for i in range(7)],'FR12':[f'Y{i}' for i in range(12)]},
        'manifest_sha256':{p.name:analysis_d._digest(p) for p in data.glob('*.parquet')},
        'cdf_calibration_scope':{'status':'SYNTHETIC_ONLY','ancestor_independence_certified':False}}
    (data/'freeze.json').write_text(json.dumps(freeze))
    task={'phase':'formal','recipe_frozen':True,'seed':7,'backbone':'clip_vit_b16'}
    config={'artifact_root':str(root),'runs':{run_id:task}}
    config_path=root/'config.json';config_path.write_text(json.dumps(config))
    base={'SYNTHETIC_ONLY':True,'artifact_root':str(root),'run_id':run_id}
    monkeypatch.setattr(cache_d,'_context',lambda path,key:(base,config,task))
    def check_identity(path,identity):
        role=Path(path).name
        assert identity['run_id']==run_id and identity['role']==role
        assert identity['source_manifest_sha256']==freeze['manifest_sha256'][f'{role}.parquet']
        return role
    def load(path,identity=None):
        role=check_identity(path,identity)
        return tuple(x.copy() for x in roles[role])
    monkeypatch.setattr(cache_d,'load_features',load)
    checked=[];monkeypatch.setattr(cache_d,'verify_cache',lambda path,identity=None:checked.append(check_identity(path,identity)))
    receipt=analysis_d.analyze_run(config_path,run_id)
    assert receipt['status']=='ANALYSIS_COMPLETE'
    out=root/'analysis/D'/run_id
    assert len(json.loads((out/'independent_rank_recomputation.json').read_text()))==2
    for rep in ('raw','l2'):
        for agg in ('mean_logit','mean_probability'):
            q=pd.read_csv(out/'rq4'/f'{rep}_{agg}'/'calibration.csv')
            assert len(q)==802 and q.status.eq('unavailable').sum()==800
            assert json.loads((out/'rq4'/f'{rep}_{agg}'/'primary_n20_vs_source_bootstrap.json').read_text())['status']=='UNAVAILABLE_CALIBRATION_BUDGET'
    assert checked==['dev'];checked.clear()
    assert analysis_d.analyze_run(config_path,run_id)==receipt
    assert sorted(checked)==sorted(cache_d.ROLES)
    (out/'source_test/metrics.csv').write_text('tampered\n')
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        analysis_d.analyze_run(config_path,run_id)
