#!/usr/bin/env python3
"""Stage 00 — audit every input this study depends on, before touching anything.

Inputs : configs/threshold_study.yaml -> phase-2 frame parquet, phase-1 checkpoint
         root, phase-4 FF++ validation cache (all read-only).
Outputs: results/tables/00_input_audit.csv, reports/input_audit.md
GPU    : no.
Target-domain labels: no (schema and row counts only, no scores are read).

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/00_audit_inputs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from study_threshold.io import val_cache_path  # noqa: E402
from study_threshold.reporting import md_table, phase_paths, write_table  # noqa: E402


def load_cfg(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PHASE / "configs/threshold_study.yaml").read_text())


def audit(cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    frame_pq = ROOT / cfg["inputs"]["frame_scores"]
    ckpt_root = ROOT / cfg["inputs"]["checkpoint_root"]
    val_dir = ROOT / cfg["inputs"]["source_val_cache"]

    rows.append({"item": "frame_scores parquet", "path": cfg["inputs"]["frame_scores"],
                 "exists": frame_pq.is_file(),
                 "detail": f"{frame_pq.stat().st_size / 1e9:.2f} GB" if frame_pq.is_file() else "MISSING"})
    rows.append({"item": "checkpoint root", "path": cfg["inputs"]["checkpoint_root"],
                 "exists": ckpt_root.is_dir(),
                 "detail": f"{len(list(ckpt_root.glob('*/*/seed_*')))} checkpoint dirs"
                 if ckpt_root.is_dir() else "MISSING"})
    rows.append({"item": "FF++ val cache dir", "path": cfg["inputs"]["source_val_cache"],
                 "exists": val_dir.is_dir(),
                 "detail": f"{len(list(val_dir.glob('*__val.parquet')))} parquet files"
                 if val_dir.is_dir() else "MISSING"})

    # --- source cache coverage, per (backbone, seed)
    for backbone in cfg["dims"]["backbones"]:
        for seed in cfg["dims"]["seeds"]:
            ckpt_id = f"{backbone}_{cfg['dims']['train_mode']}_s{seed}"
            path = val_cache_path(val_dir, ckpt_id)
            if path.is_file():
                n, nv, nm, npos = duckdb.query(
                    f"select count(*), count(distinct video_id), count(distinct method_id), "
                    f"sum(label) from read_parquet('{path}')").fetchone()
                detail = f"{n} frames / {nv} videos / {nm} methods / {int(npos)} fake frames"
            else:
                detail = "MISSING"
            rows.append({"item": f"val cache {ckpt_id}",
                         "path": str(path.relative_to(ROOT)), "exists": path.is_file(),
                         "detail": detail})

    # --- target parquet coverage (counts only)
    if frame_pq.is_file():
        target = cfg["domains"]["target"]
        cov = duckdb.query(f"""
            select backbone, seed, count(*) n_frames,
                   count(distinct video_id) n_videos,
                   count(distinct method_id) n_methods
            from read_parquet('{frame_pq}')
            where domain = '{target}' and train_mode = '{cfg["dims"]["train_mode"]}'
              and infer_mode in ('raw','direction')
            group by 1,2 order by 1,2
        """).df()
        for _, r in cov.iterrows():
            rows.append({"item": f"target {target} {r.backbone} s{int(r.seed)}",
                         "path": cfg["inputs"]["frame_scores"], "exists": True,
                         "detail": f"{int(r.n_frames)} frames (raw+direction) / "
                                   f"{int(r.n_videos)} videos / {int(r.n_methods)} methods"})
        excluded = duckdb.query(
            f"select count(*) from read_parquet('{frame_pq}') "
            f"where domain in ('" + "','".join(cfg["domains"]["exclude"]) + "')").fetchone()[0]
        rows.append({"item": "excluded domains present in parquet",
                     "path": ",".join(cfg["domains"]["exclude"]), "exists": True,
                     "detail": f"{excluded} rows exist but are never read by this phase"})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config) if args.config else None)
    paths = phase_paths(PHASE, cfg)
    df = audit(cfg)
    write_table(df, paths["tables"] / "00_input_audit.csv", float_format=None)

    missing = df[~df["exists"]]
    report = [
        "# Threshold-transfer input audit",
        "",
        f"- config: `{(PHASE / 'configs/threshold_study.yaml').relative_to(ROOT)}`",
        f"- source domain: FF++ **validation** (`{cfg['inputs']['source_val_cache']}`)",
        f"- target domain: `{cfg['domains']['target']}` from the phase-2 frame parquet",
        f"- excluded: {', '.join(cfg['domains']['exclude'])}",
        "",
        f"**{len(df) - len(missing)}/{len(df)} inputs present.**",
        "",
        md_table(df),
    ]
    if len(missing):
        report += ["", "## Missing", "", md_table(missing)]
    out = paths["reports"] / "input_audit.md"
    out.write_text("\n".join(report) + "\n")
    print(f"[write] {out}")
    print(f"[audit] {len(df) - len(missing)}/{len(df)} inputs present")
    return 1 if len(missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
