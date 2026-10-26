# src/study/canonical.py
"""Construct canonical_frame_id and source_video_id.

Priority (per spec §3): manifest domain+video+frame > normalized image_path >
path parse. We implement path parse (always available) and validate via human
spot-check in 00_inspect. Dedup key MUST include domain.
"""
from __future__ import annotations
import os
import re
import pandas as pd

_FRAME_RE = re.compile(r"/frames/(?P<vid>[^/]+)/(?P<frame>[^/]+)\.png$")

def _parse(image_path: str):
    m = _FRAME_RE.search(image_path or "")
    if m:
        return m.group("vid"), m.group("frame")
    base = os.path.basename(image_path or "")
    return os.path.basename(os.path.dirname(image_path or "")), os.path.splitext(base)[0]

def _source_from_video(video_id: str) -> str:
    v = str(video_id)
    if "_" in v:
        return v.split("_", 1)[0]
    return v

def add_canonical_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vids, frames = zip(*[_parse(p) for p in df["image_path"]])
    use_vid = [str(v) if (str(v) and str(v) != "nan") else vids[i]
               for i, v in enumerate(df.get("video_id", [""] * len(df)))]
    df["canonical_frame_id"] = [f"{d}|{v}|{f}" for d, v, f in
                                zip(df["domain"], use_vid, frames)]
    df["source_video_id"] = [_source_from_video(v) for v in use_vid]
    return df
