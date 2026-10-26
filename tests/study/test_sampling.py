# tests/study/test_sampling.py
import numpy as np, pandas as pd
from deepfake_detection.study.auc_pairs import (
    sample_pairs_native, sample_pairs_video_balanced, aggregate_video_scores)

def _frame_df():
    rows = []
    for vid, base in [("f1", 5.0), ("f2", 6.0)]:
        for fr in range(2):
            rows.append(dict(video_id=vid, label=1, score=base+fr,
                             source_video_id=vid, canonical_frame_id=f"{vid}|{fr}"))
    for vid, base in [("r1", 0.0), ("r2", 1.0)]:
        for fr in range(2):
            rows.append(dict(video_id=vid, label=0, score=base+fr,
                             source_video_id=vid, canonical_frame_id=f"{vid}|{fr}"))
    return pd.DataFrame(rows)

def test_native_uses_all_frames():
    df = _frame_df()
    ff, rr = sample_pairs_native(df, n_pairs=8, seed=1)
    assert len(ff) + len(rr) > 0

def test_video_aggregate_mean():
    df = _frame_df()
    v = aggregate_video_scores(df, how="mean")
    assert set(v["video_id"]) == {"f1","f2","r1","r2"}
    assert abs(v[v.video_id=="f1"]["video_score"].iloc[0] - 5.5) < 1e-9

def test_video_balanced_equal_videos():
    df = _frame_df()
    ff, rr = sample_pairs_video_balanced(df, frames_per_video=2, seed=1)
    assert len(ff) == len(rr)   # equal fake/real video count
