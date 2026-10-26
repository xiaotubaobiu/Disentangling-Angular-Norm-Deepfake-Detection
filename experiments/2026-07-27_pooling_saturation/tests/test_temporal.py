# tests/study_pooling/test_temporal.py
import numpy as np, pandas as pd
from study_pooling.temporal import temporal_stats, _frame_order

def test_frame_order_from_path():
    paths=["/x/000/310.png","/x/000/050.png","/x/000/200.png"]
    assert _frame_order(paths).tolist()==[1,2,0]  # sorted by numeric frame -> positions

def test_temporal_stats_keys():
    df=pd.DataFrame({"video_id":["v"]*5,"image_path":[f"/x/000/{i:03d}.png" for i in range(5)],
                     "s_raw":[0.,1.,0.5,-0.5,0.2]})
    st=temporal_stats(df,"s_raw")
    for k in ["std","mad","iqr","q90_q10","p_pos","adj_diff","outlier_ratio"]:
        assert k in st
    assert abs(st["adj_diff"]-np.mean(np.abs(np.diff([0.,1.,0.5,-0.5,0.2]))))<1e-9
