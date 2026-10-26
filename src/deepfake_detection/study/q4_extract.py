"""Q4 feature-extraction consistency gate (spec §7.1 step 2).

Validates that *re-extracted* raw features ``z`` for a retained checkpoint
match the ``feature_norm`` and head scores already stored in the diagnosis
parquet, BEFORE any full Q4 extraction. This is the de-risking step for Q4:
a mismatch would indicate a wrong feature layer, a preprocessing mismatch,
or a checkpoint-load issue — any of which would silently corrupt Q4.

Extraction mirrors ``scripts/frame_midt_diagnose.py`` EXACTLY:

    build_model(cfg["model"]) -> load_state_dict(best_model.pth) -> .eval()
    build_eval_loader(cfg, domain, distributed=False)   # shuffle=False
    extract_raw_features(model, loader, device)         # FrameFeatures

so the raw ``z`` matches the stored per-frame scores. The only addition is
that we may restrict the loader to a subset of frame paths (1000-frame gate).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from deepfake_detection.data.builders import build_eval_loader, build_external_eval_loader
from deepfake_detection.diagnosis.checkpoint_index import CheckpointInfo
from deepfake_detection.diagnosis.feature_extractor import (
    FrameFeatures,
    extract_raw_features,
)
from deepfake_detection.models.factory import build_model


def consistency_check(z_new, stored_norm, stored_raw, stored_dir, w, b, tol=1e-3):
    """Compare re-extracted features ``z_new`` against stored diagnostics.

    Args:
        z_new: (N, D) re-extracted raw features.
        stored_norm: (N,) stored feature_norm = ||z||.
        stored_raw: (N,) stored raw signed_logit = z @ w_signed + b_signed
                    (signed = fake_logit - real_logit).
        stored_dir: (N,) stored direction signed_logit computed on L2-normalized z.
        w: (D,) signed classifier weight = w_fake - w_real.
        b: scalar signed bias = b_fake - b_real.
        tol: relative tolerance for the norm, absolute-relative for the logits.

    Returns:
        True iff all three quantities match within ``tol``.
    """
    z = np.asarray(z_new, float)
    w = np.asarray(w, float)
    n = np.linalg.norm(z, axis=1)
    if np.max(np.abs(n - stored_norm) / np.maximum(stored_norm, 1e-6)) > tol:
        return False
    raw = z @ w + b
    if np.max(np.abs(raw - stored_raw)) > tol * np.maximum(1, np.max(np.abs(stored_raw))):
        return False
    direction = (z / n[:, None]) @ w + b
    if np.max(np.abs(direction - stored_dir)) > tol * np.maximum(1, np.max(np.abs(stored_dir))):
        return False
    return True


@dataclass
class SignedHead:
    """Two-logit head projected to signed-logit space (fake - real)."""

    w: np.ndarray   # (D,) = w_fake - w_real
    b: float        #     = b_fake - b_real


def signed_head_from_model(model: torch.nn.Module) -> SignedHead:
    """Extract the signed (fake - real) weight/bias from a two-logit classifier.

    Matches infer_modes.py: real_logit = logits[:,0], fake_logit = logits[:,1],
    so signed_logit = fake - real = (w[1] - w[0]) . z + (b[1] - b[0]).
    """
    raw = model.module if hasattr(model, "module") else model
    weight = raw.classifier.weight.detach().cpu().float().numpy()  # (2, D)
    bias = raw.classifier.bias.detach().cpu().float().numpy()      # (2,)
    if weight.shape[0] != 2:
        raise ValueError(f"Expected 2-logit classifier, got shape {weight.shape}")
    w_signed = weight[1] - weight[0]   # fake - real
    b_signed = float(bias[1] - bias[0])
    return SignedHead(w=w_signed, b=b_signed)


def build_model_from_checkpoint(info: CheckpointInfo, device: torch.device):
    """Mirror scripts/frame_midt_diagnose.build_model_from_checkpoint exactly."""
    import sys

    root = Path(__file__).resolve().parents[3]  # repo root
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from train import load_config  # train.py at repo root; handles _base_ inheritance

    cfg = load_config(info.config_path)
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(info.ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def extract_checkpoint(
    ckpt_info: CheckpointInfo,
    domain: str,
    frame_paths: Iterable[str] | None,
    device,
    batch_size: int = 64,
    num_workers: int = 8,
) -> FrameFeatures:
    """Re-extract raw features for one (checkpoint, domain), reusing the
    project's model factory + eval loader.

    Args:
        ckpt_info: CheckpointInfo (config + ckpt paths, backbone/mode/seed).
        domain: ffpp | cdf | dfd.
        frame_paths: if given, restrict extraction to exactly these frames
            (matched by ``frame_path``). If None or empty, extract over the
            whole (checkpoint, domain) eval set.
        device: torch.device.
        batch_size / num_workers: loader knobs.

    The loader is built with shuffle=False and distributed=False so records
    align with features. When ``frame_paths`` is given we build the full eval
    loader once to obtain its ``.dataset.records``, filter to the requested
    frames, then rebuild a restricted loader via ``build_external_eval_loader``
    so only those frames run through the backbone.

    Returns:
        FrameFeatures whose ``image_paths`` are aligned to the (possibly
        restricted) records.
    """
    device = torch.device(device)
    model, cfg = build_model_from_checkpoint(ckpt_info, device)

    cfg = dict(cfg)
    cfg.setdefault("train", {})
    cfg["train"]["per_gpu_batch"] = batch_size
    cfg["train"]["num_workers"] = num_workers
    cfg["train"]["prefetch_factor"] = 2

    # Build the full eval loader once to get its records (cheap; no forward pass).
    full_loader = build_eval_loader(cfg, domain=domain, distributed=False)
    full_records = list(full_loader.dataset.records)

    if frame_paths:
        # Some eval sets contain duplicate frame_paths (the same physical frame
        # appears under multiple method/pair_id contexts). Keep exactly one
        # record per requested frame_path — first occurrence wins — so the
        # extracted features are 1:1 with ``frame_paths`` (modulo frames absent
        # from this eval set).
        wanted = list(dict.fromkeys(str(p) for p in frame_paths))
        wanted_set = set(wanted)
        seen: set[str] = set()
        restricted = []
        for r in full_records:
            p = str(r.frame_path)
            if p in wanted_set and p not in seen:
                seen.add(p)
                restricted.append(r)
        if not restricted:
            raise RuntimeError(
                "None of the requested frame_paths matched the (checkpoint, "
                f"domain={domain}) eval records. First requested: {wanted[0]!r}; "
                f"first record frame_path: {full_records[0].frame_path!r}"
            )
        # Reorder restricted to match the caller's requested order so the
        # returned FrameFeatures aligns positionally with frame_paths.
        by_path = {str(r.frame_path): r for r in restricted}
        ordered = [by_path[p] for p in wanted if p in by_path]
        loader = build_external_eval_loader(
            cfg, records=ordered, distributed=False,
            batch_size=batch_size, num_workers=num_workers,
        )
    else:
        loader = full_loader

    feats = extract_raw_features(model, loader, device)

    # Free model + CUDA cache (caller may run several checkpoints).
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return feats
