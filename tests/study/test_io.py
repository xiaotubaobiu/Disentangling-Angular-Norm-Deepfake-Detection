# tests/study/test_io.py
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from deepfake_detection.study.io import load_frame_subset, dedup_reals

def _write(tmp_path):
    df = pd.DataFrame({
        "checkpoint_id":["c","c","c","c"],"backbone":["clip"]*4,"seed":[42]*4,
        "train_mode":["raw"]*4,"infer_mode":["raw","direction","raw","direction"],
        "domain":["ffpp","ffpp","dfd","ffpp"],"method_id":["real","real","fsgan","real"],
        "label":[0,0,1,0],"video_id":["000","000","001","000"],
        "image_path":["/f/frames/000/1.png","/f/frames/000/1.png","/f/frames/001/2.png","/f/frames/000/1.png"],
        "feature_norm":[10.0,10.0,20.0,10.0],"signed_logit":[1.0,0.5,3.0,1.0],
        "real_logit":[0,0,0,0],"fake_logit":[1,0.5,3,1],"prob_fake":[.5,.5,.8,.5],
        "prediction_correct":[1,1,1,1],
    })
    pq.write_table(pa.Table.from_pandas(df), tmp_path/"f.parquet")

def test_load_filters_dfd(tmp_path):
    _write(tmp_path)
    got = load_frame_subset(str(tmp_path/"f.parquet"), train_mode="raw", infer_mode="raw",
                            domains=["ffpp","cdf"])
    assert (got["domain"]=="dfd").sum()==0

def test_dedup_reals_keeps_one_per_frame(tmp_path):
    _write(tmp_path)
    got = load_frame_subset(str(tmp_path/"f.parquet"), train_mode="raw", infer_mode="raw", domains=["ffpp"])
    got = dedup_reals(got)
    assert ((got["label"]==0)).sum()==1   # two identical real rows -> 1
