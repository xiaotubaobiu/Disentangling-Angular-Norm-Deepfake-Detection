import numpy as np
import pandas as pd
from deepfake_detector import analysis as a


def test_method_macro_does_not_duplicate_shared_real_or_weight_large_methods():
    assert hasattr(a,'metrics'), 'unit-aware metrics not implemented'
    df=pd.DataFrame({'method_id':['real','real','a','b','b','b'],'label':[0,0,1,1,1,1],
                     'score':[0,2,3,0,0,0],'probability':[.1,.8,.9,.1,.1,.1]})
    m=a.metrics(df,threshold=1)
    assert m['real_videos']==2 and m['FPR']==.5
    assert m['macro_TPR']==.5 and m['macro_BA']==.5
    assert m['pooled_TPR']==.25
    assert m['method_macro_AUC']==.625


def test_image_rows_have_equal_video_mass_and_efs_equal_image_mass():
    assert hasattr(a,'unit_weights'), 'image weighting not implemented'
    df=pd.DataFrame({'domain':['ff']*5,'method_id':['real']*3+['e','e'],'video_id':['r','r','s','',''],
        'evaluation_unit':['video_frame']*3+['image','image']})
    w=a.unit_weights(df)
    assert w[0]==w[1] and w[0]+w[1]==w[2]
    assert w[3]==w[4]


def test_video_scoring_preserves_same_id_method_boundaries():
    assert hasattr(a,'score_features'), 'feature scoring not implemented'
    df=pd.DataFrame({'domain':['ff']*4,'method_id':['a','a','b','b'],'video_id':['v']*4,'label':[1]*4,
      'evaluation_unit':['video_frame']*4,'sample_id':['a0','a1','b0','b1'],'group_id':['g']*4})
    h=np.array([[1.,0],[3.,0],[-1.,0],[-3.,0]])
    tables=a.score_features(df,h,np.array([1.,0]),0.)
    assert len(tables['video_RL'])==2
    assert list(tables['video_RL'].score)==[2.,-2.]
    assert set(tables)=={'video_RL','video_RP','video_DL','video_DP','image_raw','image_direction'}


def test_unavailable_calibration_never_fits_threshold_from_test():
    assert hasattr(a,'calibrate'), 'calibration gate not implemented'
    try:a.calibrate(None)
    except ValueError:pass
    else:raise AssertionError('Must not substitute target/test scores')


def test_bootstrap_missing_method_is_invalid_not_smaller_macro():
    df=pd.DataFrame({'method_id':['real','a','b'],'label':[0,1,1],
        'score':[0.,1.,2.],'probability':[.5,.7,.9],
        'sample_id':['r','a','b'],'group_id':['r','a','b']})
    reps,metadata=a.test_bootstrap({'RL':df},{'RL':.5},100,42)
    pooled=reps['RL']['pooled_AUC'];macro=reps['RL']['method_macro_AUC']
    assert np.isnan(macro).sum()>np.isnan(pooled).sum()
    assert metadata['invalid_replicates']['RL']['method_macro_AUC']==int(np.isnan(macro).sum())


def test_calibration_bootstrap_returns_only_threshold_metrics_without_auc(monkeypatch):
    cal=pd.DataFrame({'method_id':['real']*20,'label':[0]*20,'score':np.arange(20.),
       'sample_id':[str(i) for i in range(20)],'group_id':[str(i) for i in range(20)]})
    target=pd.DataFrame({'method_id':['real','a','b'],'label':[0,1,1],
        'score':[0.,25.,-1.],'probability':[.5,.9,.2]})
    monkeypatch.setattr(a,'auc',lambda *args:(_ for _ in ()).throw(AssertionError('AUC must not be resorted for threshold resamples')))
    reps=a.calibration_bootstrap({'RL':cal},{'RL':{'ff':target}},10,7)
    assert len(reps['RL'])==10
    assert all(q['ff_macro_TPR']==.5 for q in reps['RL'])
