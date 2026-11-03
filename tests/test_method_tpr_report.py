"""Independent operating-point counts and the two distinct replication levels."""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/report_method_tpr.py'
    spec = importlib.util.spec_from_file_location('method_tpr_report', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def videos():
    return pd.DataFrame(dict(video_uid=['r1','r2','a1','a2','a3','b1'],
        label=[0,0,1,1,1,1], method_id=['real','real','A','A','A','B'],
        score=[.1,.5,.2,.3,.5,.8]))


def operating(threshold=.5, **extra):
    # At .5: real .5 and fake A .5 are positive; A=1/3, B=1.
    if threshold == .5:
        values=dict(false_positives=1,true_positives=2,fpr=.5,tpr=.5,
                    pooled_ba=.5,macro_tpr=2/3,macro_ba=7/12)
    else:
        assert threshold == .8
        values=dict(false_positives=0,true_positives=1,fpr=0.,tpr=.25,
                    pooled_ba=.625,macro_tpr=.5,macro_ba=.75)
    return dict(calibration_kind='default',budget=0,repeat=-1,status='ok',
        threshold=threshold,n_real=2,n_fake=4,operating_score_field='score',**values,**extra)


def test_ties_and_unequal_method_sizes_reconstruct_pooled_and_macro():
    q=module().method_operating_rows(videos(),pd.DataFrame([operating()]),['A','B']).set_index('method_id')
    assert q.loc['A','true_positives']==1 and q.loc['B','true_positives']==1
    assert q.loc['A','n_fake']==3 and q.loc['B','n_fake']==1
    assert q.loc['A','tpr']==pytest.approx(1/3) and q.loc['B','tpr']==1
    assert q.loc['A','balanced_accuracy']==pytest.approx(5/12)
    assert q.loc['B','balanced_accuracy']==pytest.approx(.75)


def test_wrong_saved_macro_or_count_is_rejected():
    m=module()
    for key,value in [('true_positives',3),('macro_tpr',.5)]:
        row=operating();row[key]=value
        with pytest.raises(ValueError,match='reconstruct'):
            m.method_operating_rows(videos(),pd.DataFrame([row]),['A','B'])


def test_missing_fixed_method_is_not_silently_dropped():
    with pytest.raises(ValueError,match='method'):
        module().method_operating_rows(videos(),pd.DataFrame([operating()]),['A','B','C'])


def test_unavailable_budget_does_not_become_zero_tpr():
    row=dict(calibration_kind='target_real',budget=80,repeat=0,status='unavailable')
    q=module().method_operating_rows(videos(),pd.DataFrame([row]),['A','B'])
    assert len(q)==2 and q.tpr.isna().all() and q.true_positives.isna().all()
    assert q.n_fake.tolist()==[3,1]


def test_calibration_subset_spread_is_reduced_within_one_model():
    m=module();cal=[]
    for repeat in range(200):
        row=operating(.5 if repeat<100 else .8)
        row.update(calibration_kind='target_real',budget=20,repeat=repeat);cal.append(row)
    raw=m.method_operating_rows(videos(),pd.DataFrame(cal),['A','B'])
    q=m.subset_summary(raw).set_index('method_id')
    assert q.loc['A','n_subsets']==200 and q.loc['A','n_valid_subsets']==200
    assert q.loc['A','tpr_mean']==pytest.approx(1/6)
    assert q.loc['A','tpr_subset_sd']==pytest.approx(math.sqrt(200/199)/6)
    assert q.loc['B','tpr_mean']==1 and q.loc['B','tpr_subset_sd']==0


def seed_rows():
    return pd.DataFrame([dict(backbone='dinov2_vit_b14',representation='raw',aggregation='mean_logit',
        calibration_kind='target_real',budget=20,method_id='A',seed=seed,
        tpr_mean=value,ba_mean=(1+value)/2,tpr_subset_sd=10.,n_valid_subsets=200)
        for seed,value in zip([7,42,123,2026,3407],[0.,.25,.5,.75,1.])])


def test_five_seed_sd_uses_seed_means_not_two_hundred_subsets():
    q=module().five_seed_summary(seed_rows()).set_index('metric')
    assert q.loc['tpr','n_training_seeds']==5
    assert q.loc['tpr','estimate_mean']==.5
    assert q.loc['tpr','estimate_sample_sd']==pytest.approx(math.sqrt(.15625))
    assert q.loc['balanced_accuracy','estimate_mean']==.75


def test_missing_training_seed_cannot_produce_a_complete_mean():
    with pytest.raises(ValueError,match='five'):
        module().five_seed_summary(seed_rows().iloc[:-1])


def test_one_unavailable_seed_estimate_keeps_five_seed_mean_na():
    rows=seed_rows();rows.loc[0,['tpr_mean','ba_mean']]=np.nan
    q=module().five_seed_summary(rows)
    assert q.estimate_mean.isna().all() and q.n_valid_seeds.eq(4).all()


def test_saved_input_hash_binding_rejects_changed_bytes(tmp_path):
    m=module();path=tmp_path/'scores.csv';path.write_text('score\n0.5\n')
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    assert m.read_checked_csv(path,expected).score.tolist()==[.5]
    path.write_text('score\n0.8\n')
    with pytest.raises(ValueError,match='hash'):
        m.read_checked_csv(path,expected)
