# tests/study_pooling/test_video_eval.py
import numpy as np, pandas as pd
from study_pooling.video_eval import eval_aggregator

def _df():
    rows=[]
    for vid,base,lbl in [("f1",3.0,1),("f2",4.0,1),("r1",-2.0,0),("r2",-1.0,0)]:
        for fr in range(3):
            rows.append(dict(video_id=vid,label=lbl,s_raw=base+0.1*fr,s_dir=base*0.3+0.1*fr))
    return pd.DataFrame(rows)

def test_mean_logit_view_metrics_identity():
    df=_df()
    m = eval_aggregator(df, "mean_logit")
    # fakes score higher than reals -> auc=1 both; identity holds
    assert abs(m["auc_raw"]-1.0)<1e-9
    assert m["identity_error"] < 1e-9

def test_mean_prob_runs():
    df=_df(); m = eval_aggregator(df, "mean_prob")
    for k in ["auc_raw","auc_dir","delta_auc","H","B","H_minus_B","identity_error"]:
        assert k in m
