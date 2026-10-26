# scripts/00_inspect_project.py
"""Spec §3: schema report + canonical-key validation + FF++ split audit."""
from __future__ import annotations
import sys, os
import pandas as pd
from pathlib import Path
PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
ROOT = PHASE.parents[1]                              # repository root
sys.path.insert(0, str(ROOT / "src"))
from study.config import load_config
from study.io import load_frame_subset, dedup_reals
from study.canonical import add_canonical_keys

CFG = load_config(PHASE / "configs" / "study_config.yaml")
REP = PHASE / CFG.paths["reports"]; REP.mkdir(parents=True, exist_ok=True)

def main():
    fpath = str(ROOT / CFG.raw["parquet"]["frame"])
    lines = ["# Schema Report\n", f"## frame parquet: `{fpath}`\n"]
    import duckdb
    cols = duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{fpath}')").df()
    lines.append("### columns\n"); lines.append(cols[["column_name","column_type"]].to_markdown(index=False)); lines.append("")
    n = duckdb.query(f"SELECT COUNT(*) n FROM read_parquet('{fpath}')").df()["n"][0]
    lines.append(f"total rows: {n}\n")
    for col in ["backbone","seed","train_mode","infer_mode","domain","method_id","label"]:
        b = duckdb.query(f"SELECT {col}, COUNT(*) n FROM read_parquet('{fpath}') WHERE domain!='dfd' GROUP BY {col} ORDER BY n DESC").df()
        lines.append(f"### counts by {col}\n"); lines.append(b.to_markdown(index=False)); lines.append("")
    sub = load_frame_subset(fpath, train_mode="raw", infer_mode="direction",
                            backbones=["clip"], seeds=[42], domains=["ffpp"])
    sub = add_canonical_keys(sub)
    dup = sub[sub.label==0].duplicated(subset=["domain","source_video_id","canonical_frame_id"]).sum()
    lines.append(f"### real duplication (clip/seed42/raw/direction/ffpp)\n")
    lines.append(f"real rows: {(sub.label==0).sum()}, duplicates dropped by dedup: {dup}\n")
    nv = sub.groupby(["method_id"])["video_id"].nunique()
    lines.append("### unique video_id per method (clip/seed42/ffpp)\n")
    lines.append(nv.to_frame("n_videos").to_markdown()); lines.append("")
    lines += audit_ffpp_split(sub, fpath)
    (REP/"schema_report.md").write_text("\n".join(lines))
    print("wrote", REP/"schema_report.md")

def audit_ffpp_split(sub, fpath):
    import duckdb
    lines = ["\n## FF++ split audit (必查项 A)\n"]
    paths = sub["image_path"].drop_duplicates().head(20).tolist()
    lines.append("sample image_paths:")
    for p in paths[:10]: lines.append(f"- `{p}`")
    has_split_token = any(tok in (paths[0] if paths else "")
                          for tok in ["train","val","test"])
    lines.append(f"\npath encodes split token? {has_split_token}")
    # How many distinct ffpp video_ids, and whether method dirs encode split
    ff = duckdb.query(f"SELECT COUNT(DISTINCT video_id) nv, COUNT(DISTINCT image_path) nf "
                      f"FROM read_parquet('{fpath}') WHERE domain='ffpp' AND train_mode='raw'").df()
    lines.append(f"\ndistinct ffpp video_ids (raw): {ff['nv'][0]}, distinct frames: {ff['nf'][0]}")
    lines.append("\n**Decision rule:** if FF++ manifest (comp/list) can be located, tag each "
                 "ffpp video_id with split; else fall back to FF++ **train** for length "
                 "orientation and label Q3 accordingly. See spec §3, §6.1.\n")
    # Look for split-like path segments under ffpp
    segs = set()
    for p in paths: segs.update([s for s in p.split('/') if s])
    lines.append(f"path segments observed: {sorted(segs)[:25]}\n")
    return lines

if __name__ == "__main__":
    main()
