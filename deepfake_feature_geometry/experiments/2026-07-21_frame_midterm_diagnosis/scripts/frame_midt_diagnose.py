"""Frame-MIDT diagnostic pipeline — main entrypoint.

Iterates over retained checkpoints, extracts raw features per (checkpoint,
domain), derives inference-mode scores, builds per-frame / per-video / per-
method / aggregate artifacts, computes bootstrap CIs and hypothesis summary,
and writes 6 diagnostic figures.

Read-only w.r.t. checkpoints and source files. No training, no new losses,
no MIDT implementation.

Usage examples:

  # Smoke test (1 ckpt × 1 domain):
  PYTHONPATH=src python experiments/2026-07-21_frame_midterm_diagnosis/scripts/frame_midt_diagnose.py \\
      --backbones efficientnet --modes raw --seeds 42 \\
      --domains ffpp --limit-ckpt 1

  # Full run (default: all 30 ckpts × 3 domains):
  PYTHONPATH=src python experiments/2026-07-21_frame_midterm_diagnosis/scripts/frame_midt_diagnose.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
ROOT = PHASE.parents[1]                              # repository root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.builders import build_eval_loader
from diagnosis.aggregation import (
    aggregate_frames_to_videos,
    rows_to_frame_dataframe,
)
from diagnosis.bootstrap import bootstrap_all_pairs
from diagnosis.checkpoint_index import (
    CheckpointInfo,
    discover_checkpoints,
)
from diagnosis.diagnostics import (
    aggregate_diagnostics_over_seeds,
    delta_auc_per_method,
    feature_norm_diagnostics,
    feature_norm_quartile_buckets,
)
from diagnosis.feature_extractor import extract_raw_features
from diagnosis.hypothesis_summary import build_hypothesis_summary
from diagnosis.infer_modes import (
    modes_for_checkpoint,
    score_normtrain_mode,
    score_rawtrain_modes,
)
from diagnosis.method_metrics import evaluate_all_methods
from diagnosis.plotting import generate_all_figures
from models.factory import build_model

# src/config_io.py handles `_base_` config inheritance.
from config_io import load_config


DEFAULT_CKPT_ROOT = str(ROOT / "experiments/2026-05-05_paper_feature_geometry/results/checkpoints")
DEFAULT_OUTPUT_DIR = str(PHASE / "results")
ALL_DOMAINS = ("ffpp", "cdf", "dfd")
ALL_BACKBONES = ("clip", "dinov2", "efficientnet")
ALL_MODES = ("raw", "normtrain")
ALL_SEEDS = (7, 42, 123, 999, 2048)


def parse_csv_arg(value: str | None, default):
    if value is None or value == "":
        return list(default)
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items


def parse_seed_arg(value: str | None, default):
    if value is None or value == "":
        return list(default)
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def build_model_from_checkpoint(info: CheckpointInfo, device: torch.device) -> torch.nn.Module:
    cfg = load_config(info.config_path)
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(info.ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def score_checkpoint_domain(
    *,
    model: torch.nn.Module,
    cfg: dict,
    info: CheckpointInfo,
    domain: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Run one (checkpoint, domain) and return its per-frame DataFrame.

    A rawtrain checkpoint produces 3 rows per frame (raw/direction/length);
    a normtrain checkpoint produces 1 row per frame.
    """
    cfg = dict(cfg)
    cfg.setdefault("train", {})
    cfg["train"]["per_gpu_batch"] = batch_size
    cfg["train"]["num_workers"] = num_workers
    cfg["train"]["prefetch_factor"] = 2

    loader = build_eval_loader(cfg, domain=domain, distributed=False)

    feats = extract_raw_features(model, loader, device)

    weight = model.classifier.weight.detach().cpu().float().numpy()
    bias = model.classifier.bias.detach().cpu().float().numpy()

    if info.mode == "raw":
        modes_scores = score_rawtrain_modes(feats.features, feats.feature_norms, weight, bias)
    else:
        modes_scores = score_normtrain_mode(feats.features, weight, bias)

    frame_df = rows_to_frame_dataframe(
        frame_features=feats,
        mode_scores_per_mode=modes_scores,
        checkpoint_id=info.checkpoint_id,
        backbone=info.backbone,
        seed=info.seed,
        train_mode=info.mode,
        domain=domain,
    )
    return frame_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbones", default=None,
                        help=f"Comma list. Default: {','.join(ALL_BACKBONES)}")
    parser.add_argument("--modes", default=None,
                        help="Comma list of train modes: raw,normtrain")
    parser.add_argument("--seeds", default=None,
                        help="Comma list of seeds")
    parser.add_argument("--domains", default=None,
                        help=f"Comma list. Default: {','.join(ALL_DOMAINS)}")
    parser.add_argument("--limit-ckpt", type=int, default=None,
                        help="Only run first N matching checkpoints (smoke test)")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Paired bootstrap iterations (lower for smoke test)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--resume-dir", default=None,
                        help="Resume into an existing output dir; skip checkpoints "
                             "whose per_ckpt/<id>.parquet already exists.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run checkpoints even if per_ckpt shard exists.")
    parser.add_argument("--stage", default="all",
                        choices=["extract", "aggregate", "all"],
                        help="extract = only run per-checkpoint feature extraction "
                             "(parallel-safe, writes per_ckpt/*.parquet); "
                             "aggregate = only concat shards + compute metrics + figures; "
                             "all = extract then aggregate (single-process).")
    parser.add_argument("--only-ckpt", default=None,
                        help="Extract only this single checkpoint_id (used by the "
                             "multi-GPU launcher to dispatch one ckpt per GPU).")
    args = parser.parse_args()

    backbones = parse_csv_arg(args.backbones, ALL_BACKBONES)
    modes = parse_csv_arg(args.modes, ALL_MODES)
    seeds = parse_seed_arg(args.seeds, ALL_SEEDS)
    domains = parse_csv_arg(args.domains, ALL_DOMAINS)
    for d in domains:
        if d not in ALL_DOMAINS:
            raise ValueError(f"Unknown domain: {d}. Allowed: {ALL_DOMAINS}")

    infos = discover_checkpoints(
        ckpt_root=args.ckpt_root,
        backbones=backbones,
        modes=modes,
        seeds=seeds,
        limit=args.limit_ckpt,
    )
    if args.only_ckpt:
        infos = [i for i in infos if i.checkpoint_id == args.only_ckpt]
        if not infos:
            raise RuntimeError(f"--only-ckpt={args.only_ckpt} matched no checkpoint")
    if not infos:
        raise RuntimeError(
            f"No checkpoints found under {args.ckpt_root} matching "
            f"backbones={backbones}, modes={modes}, seeds={seeds}"
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Resume support: if --resume-dir is given, reuse that directory and skip
    # checkpoints whose per_ckpt/<id>.parquet already exists. This protects
    # against the run being killed mid-way (e.g. session teardown, OOM).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"--resume-dir not found: {output_dir}")
    else:
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    per_ckpt_dir = output_dir / "per_ckpt"
    per_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] output dir: {output_dir}", flush=True)
    print(f"[diag] device: {device}", flush=True)
    print(f"[diag] stage={args.stage}, {len(infos)} checkpoint(s) × {len(domains)} domain(s) "
          f"= {len(infos) * len(domains)} (ckpt, domain) pairs", flush=True)

    # ============== STAGE: extract (parallel-safe) ======================
    if args.stage in ("extract", "all"):
        run_extract_stage(
            infos=infos,
            domains=domains,
            per_ckpt_dir=per_ckpt_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )

    if args.stage == "extract":
        return

    # ============== STAGE: aggregate (single-process) ===================
    run_aggregate_stage(
        output_dir=output_dir,
        per_ckpt_dir=per_ckpt_dir,
        domains=domains,
        backbones=backbones,
        modes=modes,
        seeds=seeds,
        infos=infos,
        timestamp=timestamp,
        n_bootstrap=args.n_bootstrap,
        skip_bootstrap=args.skip_bootstrap,
        skip_figures=args.skip_figures,
    )


def run_extract_stage(
    *,
    infos,
    domains,
    per_ckpt_dir: Path,
    device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> None:
    """Per-checkpoint feature extraction. Parallel-safe: each (GPU, ckpt)
    invocation writes its own per_ckpt/<id>.parquet and touches nothing else.
    """
    for i, info in enumerate(infos, 1):
        ckpt_out = per_ckpt_dir / f"{info.checkpoint_id}.parquet"
        if ckpt_out.exists() and not overwrite:
            print(f"[diag] ({i}/{len(infos)}) {info.checkpoint_id}: SKIP "
                  f"(exists: {ckpt_out.name})", flush=True)
            continue

        t0 = time.time()
        print(f"\n[diag] ({i}/{len(infos)}) {info.checkpoint_id}", flush=True)
        try:
            model, cfg = build_model_from_checkpoint(info, device)
        except Exception as e:
            print(f"[diag]   FAILED to load model: {e}", flush=True)
            continue

        ckpt_frame_dfs: list[pd.DataFrame] = []
        for domain in domains:
            td = time.time()
            try:
                frame_df = score_checkpoint_domain(
                    model=model, cfg=cfg, info=info, domain=domain,
                    device=device, batch_size=batch_size,
                    num_workers=num_workers,
                )
            except Exception as e:
                print(f"[diag]   {domain}: FAILED ({e})", flush=True)
                import traceback
                traceback.print_exc()
                continue
            ckpt_frame_dfs.append(frame_df)
            n_frames = len(frame_df)
            n_modes = frame_df["infer_mode"].nunique()
            print(
                f"[diag]   {domain}: {n_frames} frame-rows "
                f"({n_modes} infer_mode(s)) in {time.time() - td:.1f}s",
                flush=True,
            )

        # Free model + cached CUDA state before next checkpoint.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not ckpt_frame_dfs:
            print(f"[diag]   no domains succeeded; not writing", flush=True)
            continue

        # Incremental save: write this checkpoint's frames immediately to a
        # temp file then atomic rename, so a partial write is never visible.
        ckpt_all = pd.concat(ckpt_frame_dfs, ignore_index=True)
        tmp_out = ckpt_out.with_suffix(".parquet.tmp")
        ckpt_all.to_parquet(tmp_out, index=False)
        tmp_out.rename(ckpt_out)
        print(
            f"[diag]   total {time.time() - t0:.1f}s; wrote {ckpt_out.name} "
            f"({len(ckpt_all)} rows)", flush=True,
        )


def run_aggregate_stage(
    *,
    output_dir: Path,
    per_ckpt_dir: Path,
    domains,
    backbones,
    modes,
    seeds,
    infos,
    timestamp: str,
    n_bootstrap: int,
    skip_bootstrap: bool,
    skip_figures: bool,
) -> None:
    """Concat per_ckpt shards, then compute metrics / bootstrap / figures."""

    # ---------------- Concatenate all per_ckpt shards --------------
    shard_paths = sorted(per_ckpt_dir.glob("*.parquet"))
    if not shard_paths:
        raise RuntimeError("No per-frame data was produced; aborting.")
    print(f"\n[diag] concatenating {len(shard_paths)} per_ckpt shards", flush=True)
    frame_dfs = [pd.read_parquet(p) for p in shard_paths]
    all_frames = pd.concat(frame_dfs, ignore_index=True)
    print(f"[diag] total per-frame rows: {len(all_frames)}", flush=True)

    # ---------------- Write per-frame / per-video Parquet ----------------
    per_frame_path = output_dir / "per_frame_scores.parquet"
    all_frames.to_parquet(per_frame_path, index=False)
    print(f"[diag] wrote {per_frame_path}", flush=True)

    all_videos = aggregate_frames_to_videos(all_frames)
    per_video_path = output_dir / "per_video_scores.parquet"
    all_videos.to_parquet(per_video_path, index=False)
    print(f"[diag] wrote {per_video_path}  ({len(all_videos)} video rows)", flush=True)

    # ---------------- Per-method metrics ----------------------------------
    per_method_frames: list[pd.DataFrame] = []
    for domain in domains:
        per_method_frames.append(evaluate_all_methods(all_videos, domain))
    per_method_df = pd.concat(
        [f for f in per_method_frames if not f.empty], ignore_index=True
    )
    per_method_path = output_dir / "per_method_metrics.csv"
    per_method_df.to_csv(per_method_path, index=False)
    print(f"[diag] wrote {per_method_path}  ({len(per_method_df)} rows)", flush=True)

    # ---------------- Aggregate diagnostics + norm diagnostics ------------
    infer_modes_present = sorted(per_method_df["infer_mode"].unique())
    aggregate_df = aggregate_diagnostics_over_seeds(per_method_df, infer_modes_present)
    aggregate_path = output_dir / "aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"[diag] wrote {aggregate_path}", flush=True)

    delta_auc_df = delta_auc_per_method(per_method_df, "cdf")
    delta_auc_path = output_dir / "delta_auc_per_method_cdf.csv"
    delta_auc_df.to_csv(delta_auc_path, index=False)

    norm_diagnostics = feature_norm_diagnostics(all_frames)
    # Norm quartile buckets (rawtrain only — length mode has no AUC meaning).
    norm_bucket_frames: list[pd.DataFrame] = []
    for ckpt in sorted(all_frames["checkpoint_id"].unique()):
        for mode in ("raw", "direction", "normtrain_norm"):
            bdf = feature_norm_quartile_buckets(all_frames, ckpt, mode)
            if not bdf.empty:
                norm_bucket_frames.append(bdf)
    norm_bucket_df = (
        pd.concat(norm_bucket_frames, ignore_index=True)
        if norm_bucket_frames else pd.DataFrame()
    )
    norm_bucket_path = output_dir / "norm_quartile_buckets.csv"
    norm_bucket_df.to_csv(norm_bucket_path, index=False)

    # ---------------- Bootstrap -------------------------------------------
    bootstrap_df = pd.DataFrame()
    if not skip_bootstrap:
        # Only bootstrap rawtrain ckpts (direction vs raw); normtrain has no
        # comparable paired structure across modes within the same checkpoint.
        rawtrain_backbones = sorted(
            per_method_df[per_method_df["train_mode"] == "raw"]["backbone"].unique()
        )
        rawtrain_seeds = sorted(
            int(s) for s in per_method_df[per_method_df["train_mode"] == "raw"]["seed"].unique()
        )
        bootstrap_df = bootstrap_all_pairs(
            video_df=all_videos,
            backbones=rawtrain_backbones,
            seeds=rawtrain_seeds,
            domains=[d for d in domains if d != "dfd"],  # paired direction/raw only on ffpp/cdf
            mode_pairs=[("direction", "raw")],
            n_bootstrap=n_bootstrap,
        )
        bootstrap_path = output_dir / "bootstrap_direction_vs_raw.csv"
        bootstrap_df.to_csv(bootstrap_path, index=False)
        print(f"[diag] wrote {bootstrap_path}", flush=True)

    # ---------------- Hypothesis summary ----------------------------------
    hypothesis = build_hypothesis_summary(
        aggregate_df=aggregate_df,
        method_df=per_method_df,
        delta_auc_df=delta_auc_df,
        norm_diagnostics=norm_diagnostics,
    )
    summary_payload = {
        "generated_at": timestamp,
        "n_checkpoints": len(infos),
        "n_domains": len(domains),
        "backbones": backbones,
        "modes": modes,
        "seeds": seeds,
        "domains": domains,
        "hypotheses": hypothesis,
    }
    summary_path = output_dir / "hypothesis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"[diag] wrote {summary_path}", flush=True)

    # ---------------- Figures ---------------------------------------------
    if not skip_figures:
        written = generate_all_figures(
            output_dir=output_dir,
            method_df=per_method_df,
            aggregate_df=aggregate_df,
            delta_auc_df=delta_auc_df,
            norm_bucket_df=norm_bucket_df,
        )
        for p in written:
            print(f"[diag] wrote figure {p}", flush=True)

    # ---------------- Console summary -------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("Hypothesis verdicts:", flush=True)
    for h_id, h in hypothesis.items():
        print(f"  {h_id} [{h['verdict']}]: {h['question']}", flush=True)
        for k, v in h["evidence"].items():
            if isinstance(v, float):
                print(f"      {k}: {v:.4f}", flush=True)
            else:
                print(f"      {k}: {v}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
