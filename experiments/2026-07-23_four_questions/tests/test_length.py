# tests/study/test_length.py
import numpy as np, pandas as pd
from study.length import orient_sign, micro_auc, equalized_auc

def test_orient_by_ffpp_val():
    # fake has smaller q -> AUC<0.5 -> flip
    qf=np.array([0.1,0.2]); qr=np.array([1.0,2.0])
    assert orient_sign(qf,qr)==-1

def test_micro_and_equalized():
    rng=np.random.default_rng(0)
    df=pd.DataFrame({"label":[1]*50+[0]*50,"method":(["fsgan"]*25+["lia"]*25+["real"]*50),
        "score":np.concatenate([rng.normal(1,1,50),rng.normal(0,1,50)]),
        "video_id":np.repeat(range(20),5)})
    assert micro_auc(df)>0.5
    m,lo,hi=equalized_auc(df,repeats=20,seed=0,frames_per_video=5)
    # tiny tol guards against float-repr ties when the mean sits exactly on a
    # percentile boundary (np.percentile can return 0.7840000000000001 vs 0.784)
    assert lo-1e-9 <= m <= hi+1e-9
