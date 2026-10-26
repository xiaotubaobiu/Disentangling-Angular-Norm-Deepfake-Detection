# tests/study/test_canonical.py
import pandas as pd
from study.canonical import add_canonical_keys

def _df(paths, domain="ffpp", method="real", video="000"):
    return pd.DataFrame({"image_path": paths, "domain": domain,
                         "method_id": method, "video_id": video, "label": 0})

def test_canonical_frame_id_from_path():
    p = "/Dataset/x/FaceForensics++/original_sequences/youtube/c23/frames/000/254.png"
    df = _df([p])
    out = add_canonical_keys(df)
    assert out.loc[0, "canonical_frame_id"] == "ffpp|000|254"
    assert out.loc[0, "source_video_id"] == "000"

def test_real_dedup_key_includes_domain():
    p = "/x/frames/000_003/010.png"
    df = _df([p], domain="cdf", method="fsgan", video="000_003")
    df.loc[0, "label"] = 1
    out = add_canonical_keys(df)
    assert out.loc[0, "source_video_id"] == "000"
