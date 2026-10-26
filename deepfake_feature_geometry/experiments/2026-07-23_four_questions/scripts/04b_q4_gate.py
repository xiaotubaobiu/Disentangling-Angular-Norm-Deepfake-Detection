#!/usr/bin/env python
"""Q4 feature-extraction consistency gate (spec §7.1 step 2).

Re-extracts raw features ``z`` for ONE retained checkpoint (clip/raw/seed_42)
over ~1000 FF++ frames, then checks that:

    ||z||                    matches stored feature_norm
    z . w_signed + b_signed  matches stored raw signed_logit
    (z/||z||) . w + b        matches stored direction signed_logit

where w_signed = w_fake - w_real, b_signed = b_fake - b_real (the two-logit
head projected to signed-logit space, exactly as infer_modes.py uses them).

PASS (tol=1e-3) means the extraction wiring is correct and Q4's full
extraction can proceed. FAIL is itself valuable — the actual max errors are
reported so the controller can diagnose (wrong feature layer, preprocessing
mismatch, or checkpoint-load issue per spec §7.1 step2 / §10).

Report:  reports/Q4_consistency_gate.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
ROOT = PHASE.parents[1]                              # repository root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diagnosis.checkpoint_index import discover_checkpoints
from study.config import load_config
from study.q4_extract import (
    consistency_check,
    extract_checkpoint,
    signed_head_from_model,
)

CFG = load_config(PHASE / "configs" / "study_config.yaml")
PARQUET_PATH = ROOT / CFG.parquet["frame"]
REPORT_PATH = PHASE / "reports" / "Q4_consistency_gate.md"
CKPT_BACKBONE = "clip"
CKPT_MODE = "raw"
CKPT_SEED = 42
DOMAIN = "ffpp"
N_FRAMES = 1000
TOL = 1e-3


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Locate the checkpoint -------------------------------------------
    infos = discover_checkpoints(
        backbones=[CKPT_BACKBONE], modes=[CKPT_MODE], seeds=[CKPT_SEED],
    )
    if not infos:
        raise RuntimeError(
            f"No checkpoint found for {CKPT_BACKBONE}/{CKPT_MODE}/seed_{CKPT_SEED}"
        )
    info = infos[0]
    checkpoint_id = info.checkpoint_id
    print(f"[gate] checkpoint: {checkpoint_id}", flush=True)
    print(f"[gate]   ckpt:  {info.ckpt_path}", flush=True)
    print(f"[gate]   cfg:   {info.config_path}", flush=True)

    # ---- 2. Load stored per-frame rows for this (ckpt, domain) --------------
    # We need raw + direction signed_logit + feature_norm + image_path.
    cols = [
        "infer_mode", "image_path", "feature_norm",
        "signed_logit", "checkpoint_id", "domain",
    ]
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=cols,
        filters=[
            ("checkpoint_id", "==", checkpoint_id),
            ("domain", "==", DOMAIN),
            ("infer_mode", "in", ["raw", "direction"]),
        ],
    )
    raw_df = df[df["infer_mode"] == "raw"].drop_duplicates("image_path").set_index("image_path")
    dir_df = df[df["infer_mode"] == "direction"].drop_duplicates("image_path").set_index("image_path")
    # image_paths present in BOTH raw and direction modes (they share frames
    # for a rawtrain checkpoint, so this should be all of them).
    common_paths = sorted(set(raw_df.index) & set(dir_df.index))
    if len(common_paths) < N_FRAMES:
        raise RuntimeError(
            f"Only {len(common_paths)} frames available in parquet for "
            f"{checkpoint_id}/{DOMAIN}; need {N_FRAMES}."
        )
    rng = np.random.default_rng(0)
    chosen_idx = rng.choice(len(common_paths), size=N_FRAMES, replace=False)
    chosen_paths = [common_paths[i] for i in chosen_idx]

    stored_norm = raw_df.loc[chosen_paths, "feature_norm"].to_numpy(float)
    stored_raw = raw_df.loc[chosen_paths, "signed_logit"].to_numpy(float)
    stored_dir = dir_df.loc[chosen_paths, "signed_logit"].to_numpy(float)
    print(f"[gate] sampled {N_FRAMES} ffpp frames (of {len(common_paths)} available)", flush=True)

    # ---- 3. Re-extract raw features for exactly those frames ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gate] device: {device}", flush=True)
    t0 = time.time()
    feats = extract_checkpoint(info, DOMAIN, chosen_paths, device,
                               batch_size=64, num_workers=8)
    print(f"[gate] extracted {feats.features.shape[0]} frames in "
          f"{time.time() - t0:.1f}s", flush=True)

    # ---- 4. Align re-extracted features to the stored order -----------------
    # extract_checkpoint returns features ordered to match `chosen_paths`
    # (one record per requested frame_path), so they align positionally.
    ret_paths = list(feats.image_paths)
    chosen_lookup = {p: i for i, p in enumerate(chosen_paths)}
    ret_to_stored = np.array([chosen_lookup[p] for p in ret_paths])
    z = feats.features
    z_norms_re = np.linalg.norm(z, axis=1)
    stored_norm = stored_norm[ret_to_stored]
    stored_raw = stored_raw[ret_to_stored]
    stored_dir = stored_dir[ret_to_stored]
    print(f"[gate] aligned {z.shape[0]} re-extracted frames to stored rows", flush=True)

    # ---- 5. Compute signed head and check consistency -----------------------
    # Rebuild the model briefly just to read classifier weight/bias (cheap; we
    # already freed it inside extract_checkpoint). To avoid a second load, we
    # could return the head from extract_checkpoint, but a fresh load is simpler
    # and keeps the lib API clean. Use the same factory path.
    from study.q4_extract import build_model_from_checkpoint
    model, _ = build_model_from_checkpoint(info, device)
    head = signed_head_from_model(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w_signed, b_signed = head.w, head.b
    raw_re = z @ w_signed + b_signed
    dir_re = (z / z_norms_re[:, None]) @ w_signed + b_signed

    err_norm_rel = np.max(np.abs(z_norms_re - stored_norm) / np.maximum(stored_norm, 1e-6))
    err_raw = np.max(np.abs(raw_re - stored_raw))
    err_raw_rel = err_raw / np.maximum(1.0, np.max(np.abs(stored_raw)))
    err_dir = np.max(np.abs(dir_re - stored_dir))
    err_dir_rel = err_dir / np.maximum(1.0, np.max(np.abs(stored_dir)))

    passed = consistency_check(
        z, stored_norm, stored_raw, stored_dir, w_signed, b_signed, tol=TOL,
    )
    verdict = "PASS" if passed else "FAIL"

    # ---- 6. Write report ----------------------------------------------------
    lines = [
        "# Q4 Feature-Extraction Consistency Gate",
        "",
        f"- Spec: §7.1 step 2 (de-risking before full Q4 extraction).",
        f"- Checkpoint: `{checkpoint_id}`  (`{CKPT_BACKBONE}/{CKPT_MODE}/seed_{CKPT_SEED}`)",
        f"- Domain: `{DOMAIN}`",
        f"- Frames checked: **{N_FRAMES}**",
        f"- Tolerance (tol): `{TOL}`",
        "",
        "## Verdict",
        "",
        f"### **{verdict}**",
        "",
        "## Max errors",
        "",
        "| Quantity | max abs err | max rel err |",
        "|---|---|---|",
        f"| feature_norm (\\|\\|z\\|\\|) | {np.max(np.abs(z_norms_re - stored_norm)):.6g} | {err_norm_rel:.6g} |",
        f"| raw signed_logit (z.w+b) | {err_raw:.6g} | {err_raw_rel:.6g} |",
        f"| direction signed_logit (z/\\|\\|z\\|\\|.w+b) | {err_dir:.6g} | {err_dir_rel:.6g} |",
        "",
        "## Method",
        "",
        "Re-extraction mirrors `experiments/2026-07-21_frame_midterm_diagnosis/"
        "scripts/frame_midt_diagnose.py` exactly:",
        "`build_model(cfg)` → `load_state_dict(best_model.pth, strict=True)` → `.eval()`,",
        "`build_eval_loader(cfg, domain, distributed=False)` (shuffle=False), then",
        "`extract_raw_features`. The signed head is `w_signed = w_fake - w_real`,",
        "`b_signed = b_fake - b_real` (real_logit = logits[:,0], fake_logit = logits[:,1]).",
        "",
        f"- Re-extracted feature dim D = `{z.shape[1]}`",
        f"- b_signed = `{b_signed:.6g}`",
        "",
        "## Interpretation",
        "",
        ("- All quantities match within tol; the extraction wiring (model factory, "
         "checkpoint load, eval loader, feature layer) is correct. Q4's full "
         "extraction can proceed.") if passed else
        ("- A quantity exceeds tol. Per spec §7.1 step2 / §10 this indicates one of: "
         "wrong feature layer, preprocessing/normalization mismatch, or a "
         "checkpoint-load issue. Do NOT proceed to full Q4 extraction until resolved."),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"[gate] wrote {REPORT_PATH}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"Q4 consistency gate: {verdict}", flush=True)
    print(f"  err_norm_rel  = {err_norm_rel:.6g}", flush=True)
    print(f"  err_raw_rel   = {err_raw_rel:.6g}", flush=True)
    print(f"  err_dir_rel   = {err_dir_rel:.6g}", flush=True)
    print("=" * 60, flush=True)

    # Non-zero exit on FAIL so the controller can detect it programmatically,
    # but we still print the report above either way.
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
