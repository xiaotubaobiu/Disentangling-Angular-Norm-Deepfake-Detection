import numpy as np
import pandas as pd
import pytest
from deepfake_detector import experiments as e


def table(role,real_count=3):
    rows=[];features=[]
    for method,label,count in [('real',0,real_count),('a',1,2),('b',1,2)]:
        for video in range(count):
            for frame in range(8):
                rows.append(dict(domain='ff' if role!='cdf_test' else 'cdf',experiment_role=role,method_id=method,
                  method_type='real' if not label else 'FS',label=label,evaluation_unit='video_frame',video_id=f'{method}{video}',
                  sample_id=f'{role}/{method}/{video}/{frame}',group_id=f'{role}/{method}/{video}'))
                # Mixed sign and unequal lengths make nonlinear pooling meaningful.
                features.append([(-1 if not label else 1)*(1+video)+(-2 if frame==0 else .1), .2+frame/10])
    return pd.DataFrame(rows),np.asarray(features,dtype=np.float32)


def test_length_control_uses_training_balanced_median_and_preserves_bias():
    assert hasattr(e,'training_median'), 'source-only length controls missing'
    train,h=table('train');c=e.training_median(train,h)
    assert c>0
    with pytest.raises(ValueError,match='training'):e.training_median(table('ff_test')[0],h)
    x=np.array([[3.,4.]]);w=np.array([1.,0.]);b=2.
    logits=e.length_logits(x,w,b,2.)
    assert logits['h_over_c'][0]==3.5 and logits['h_over_c'][0]!=(x@w+b)[0]/2
    assert logits['u'][0]==2.6 and logits['c_u'][0]==3.2


def test_real_only_fpr5_analysis_does_not_require_or_fabricate_tpr95(tmp_path):
    assert hasattr(e,'analyze_payload'), 'A–D analysis interface missing'
    payload={role:table(role,20 if role=='calibration' else 3) for role in ['train','calibration','ff_test','cdf_test']}
    cal,h=payload['calibration'];keep=cal.label.eq(0).to_numpy();payload['calibration']=(cal[keep].reset_index(drop=True),h[keep])
    config={'scales':[.25,.5,1.,2.,4.],'source_FPR_budget':.05,'test_bootstrap':10,'calibration_bootstrap':10,
      'matched_head_L2':1e-4,'matched_head_max_iter':2000}
    result=e.analyze_payload(payload,np.array([1.,0.]),.3,config,tmp_path,expected_video_methods=['a','b'],expected_image_methods=['a','b'])
    assert result['FPR5']['available']==True and result['TPR95']['available']==False
    assert len(result['matched_heads'])==2
    assert all(q['source_FPR']<=.05+1e-14 for q in result['FPR5']['thresholds'].values())
    assert {'RL','RP','DL','DP'}==set(result['A']['ff']['video'])
    assert result['A']['ff']['interaction']==pytest.approx(result['A']['ff']['delta_dir']-result['A']['ff']['delta_raw'])
    assert all(abs(q['ledger_delta']-q['direct_delta'])<1e-10 for q in result['B']['pair_transitions'])
    assert (tmp_path/'scores/ff/video_RL.parquet').exists()
    assert result['D']['test_resampling']['ff']['iterations']==10


def test_missing_calibration_preserves_ranking_and_marks_only_operating_points_unavailable(tmp_path):
    assert hasattr(e,'analyze_payload'), 'A–D analysis interface missing'
    config={'scales':[1.],'source_FPR_budget':.05,'test_bootstrap':5,'calibration_bootstrap':5,'matched_head_L2':1e-4,'matched_head_max_iter':2000}
    payload={role:table(role) for role in ['train','ff_test','cdf_test']}
    result=e.analyze_payload(payload,np.array([1.,0.]),.3,config,tmp_path,expected_video_methods=['a','b'],expected_image_methods=['a','b'])
    assert not result['FPR5']['available'] and not result['TPR95']['available']
    assert np.isfinite(result['A']['ff']['video']['RL']['method_macro_AUC'])
    assert result['D']['calibration_resampling']['available']==False
