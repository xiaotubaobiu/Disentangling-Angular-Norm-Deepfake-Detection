# MIDT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MIDT (Method-Invariant Direction Training) — a dual-head training method with inter-method direction alignment for cross-domain deepfake detection.

**Architecture:** Wrap existing backbone with two classifier heads (raw + direction) and an EMA-based direction centroid tracker. Add `midt_step` to the training dispatch. Dataset returns method labels alongside images. Loss = L_raw + alpha * L_dir + beta * L_align.

**Tech Stack:** PyTorch DDP, mixed precision (AMP), YAML config inheritance, existing EfficientNet/DINOv2/CLIP backbones.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/deepfake_detection/models/midt.py` | MIDT wrapper: dual heads + EMA centroids + alignment loss |
| Modify | `src/deepfake_detection/models/factory.py` | Register `midt_*` model names |
| Modify | `src/deepfake_detection/data/datasets.py` | Add `method` field to `FrameClassificationDataset.__getitem__` |
| Modify | `src/deepfake_detection/engine/trainers.py` | Add `midt_step` + dispatch in `run_train_epoch` |
| Modify | `src/deepfake_detection/engine/trainers.py` | Update `run_sample_eval_epoch` for dual-head eval |
| Modify | `train.py` | Pass centroid state, save/load centroids, hyperparams |
| Create | `configs/midt_efficientnet.yaml` | Phase 1 config |
| Create | `configs/midt_dinov2.yaml` | Phase 2 config |
| Create | `configs/midt_clip.yaml` | Phase 3 config |

---

### Task 1: Create MIDT Model Wrapper

**Files:**
- Create: `src/deepfake_detection/models/midt.py`

- [ ] **Step 1: Write the MIDT model class**

```python
# src/deepfake_detection/models/midt.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIDTWrapper(nn.Module):
    """Method-Invariant Direction Training wrapper.

    Wraps any backbone with:
    - raw_head: classifier on raw features f
    - dir_head: classifier on L2-normalized features f_hat
    - EMA direction centroids per (method, class) for alignment loss
    """

    def __init__(self, backbone, feature_dim: int, num_classes: int = 2,
                 centroid_momentum: float = 0.9, num_methods: int = 16):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.normalize_features = False  # MIDT handles normalization internally

        self.raw_head = nn.Linear(feature_dim, num_classes)
        self.dir_head = nn.Linear(feature_dim, num_classes)

        self.centroid_momentum = centroid_momentum
        self.num_methods = num_methods

        # EMA centroids: shape (num_classes, num_methods, feature_dim)
        # Index: centroids[class_idx][method_idx] = direction centroid
        self.register_buffer(
            "centroid_initialized",
            torch.zeros(2, num_methods, dtype=torch.bool),
        )
        self.register_buffer(
            "centroids",
            torch.zeros(2, num_methods, feature_dim),
        )
        # Per-method sample counts for centroid averaging
        self.register_buffer(
            "centroid_counts",
            torch.zeros(2, num_methods, dtype=torch.float),
        )

    def extract_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        """Standard forward for backward compatibility. Returns raw logits."""
        features = self.extract_features(images)
        return self.raw_head(features)

    def forward_dual(self, images):
        """Returns (raw_logits, dir_logits, features)."""
        features = self.extract_features(images)
        raw_logits = self.raw_head(features)
        f_hat = F.normalize(features, dim=-1)
        dir_logits = self.dir_head(f_hat)
        return raw_logits, dir_logits, features

    @torch.no_grad()
    def update_centroids(self, f_hat, labels, method_ids):
        """Update EMA direction centroids from current batch.

        Args:
            f_hat: L2-normalized features, shape (B, D)
            labels: class labels, shape (B,)
            method_ids: integer method indices, shape (B,)
        """
        for c in range(2):
            for m in range(self.num_methods):
                mask = (labels == c) & (method_ids == m)
                n = mask.sum()
                if n == 0:
                    continue
                batch_centroid = f_hat[mask].mean(dim=0)
                if not self.centroid_initialized[c, m]:
                    self.centroids[c, m] = batch_centroid
                    self.centroid_initialized[c, m] = True
                    self.centroid_counts[c, m] = n.float()
                else:
                    mom = self.centroid_momentum
                    self.centroids[c, m] = mom * self.centroids[c, m] + (1 - mom) * batch_centroid
                    self.centroid_counts[c, m] += n.float()

    def compute_alignment_loss(self):
        """L_align: penalize dispersion of per-class direction centroids across methods.

        L_align = sum_c (1/M^2) sum_{m!=m'} [1 - cos(mu_hat_c^m, mu_hat_c^{m'})]
        """
        loss = torch.tensor(0.0, device=self.centroids.device)
        count = 0
        for c in range(2):
            initialized = self.centroid_initialized[c]  # (num_methods,)
            active_indices = torch.where(initialized)[0]
            M = len(active_indices)
            if M < 2:
                continue
            active_centroids = self.centroids[c, active_indices]  # (M, D)
            # Pairwise cosine similarity: (M, M)
            active_centroids_norm = F.normalize(active_centroids, dim=-1)
            sim_matrix = active_centroids_norm @ active_centroids_norm.t()
            # Sum of (1 - cos_sim) for all pairs m != m'
            mask = ~torch.eye(M, dtype=torch.bool, device=sim_matrix.device)
            loss = loss + sim_matrix[mask].sub(1.0).neg().mean()
            count += 1
        if count > 0:
            loss = loss / count
        return loss

    def centroid_state_dict(self):
        return {
            "centroids": self.centroids.clone(),
            "centroid_initialized": self.centroid_initialized.clone(),
            "centroid_counts": self.centroid_counts.clone(),
        }

    def load_centroid_state(self, state):
        self.centroids.copy_(state["centroids"])
        self.centroid_initialized.copy_(state["centroid_initialized"])
        self.centroid_counts.copy_(state["centroid_counts"])
```

- [ ] **Step 2: Verify file creates without syntax errors**

Run: `cd /home/z/project/deepfake_detection && python -c "from src.deepfake_detection.models.midt import MIDTWrapper; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/models/midt.py
git commit -m "feat: add MIDT model wrapper with dual heads and EMA centroids"
```

---

### Task 2: Add Method Label to Dataset Batch

**Files:**
- Modify: `src/deepfake_detection/data/datasets.py:60-74`
- Modify: `src/deepfake_detection/data/constants.py:1-11`

- [ ] **Step 1: Add method-to-index mapping to constants.py**

Append to `src/deepfake_detection/data/constants.py` after line 11:

```python
METHOD_TO_INDEX = {name: i for i, name in enumerate(ALL_METHODS)}
METHOD_TO_INDEX["real"] = len(ALL_METHODS)  # real gets its own index
```

- [ ] **Step 2: Add `method_id` to FrameClassificationDataset.__getitem__**

In `src/deepfake_detection/data/datasets.py`, add the import at line 10:

```python
from deepfake_detection.data.constants import ALL_METHODS, IMG_SIZE, REAL_LABEL, FAKE_LABEL, METHOD_TO_INDEX
```

Modify `__getitem__` (lines 60-74) to return `method_id`:

```python
    def __getitem__(self, idx):
        rec = self.records[idx]
        img = _load_frame(rec.frame_path)
        img = cv2_resize(img, self.img_size)
        img = self.transform(image=img)["image"]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        video_id = rec.video_id if hasattr(rec, "video_id") else rec.pair_id
        sample_id = rec.sample_id if hasattr(rec, "sample_id") else f"{rec.frame_path}::{rec.label}::{video_id}"
        method_id = METHOD_TO_INDEX.get(rec.method, len(ALL_METHODS))
        return {
            "image": tensor,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "video_id": video_id,
            "image_path": rec.frame_path,
            "sample_id": sample_id,
            "method_id": torch.tensor(method_id, dtype=torch.long),
        }
```

- [ ] **Step 3: Verify dataset import works**

Run: `cd /home/z/project/deepfake_detection && python -c "from src.deepfake_detection.data.constants import METHOD_TO_INDEX; print(METHOD_TO_INDEX)"`
Expected: dict with 17 entries (16 methods + real)

- [ ] **Step 4: Commit**

```bash
git add src/deepfake_detection/data/datasets.py src/deepfake_detection/data/constants.py
git commit -m "feat: add method_id to FrameClassificationDataset batch"
```

---

### Task 3: Register MIDT in Model Factory

**Files:**
- Modify: `src/deepfake_detection/models/factory.py`

- [ ] **Step 1: Add MIDT imports and factory branches**

In `src/deepfake_detection/models/factory.py`, add import at line 8:

```python
from deepfake_detection.models.midt import MIDTWrapper
from deepfake_detection.data.constants import ALL_METHODS
```

Add before the `raise ValueError` on line 48:

```python
    if name.startswith("midt_"):
        backbone_name = name.replace("midt_", "")
        backbone_cfg = {**model_cfg, "name": backbone_name}
        backbone = build_model(backbone_cfg)
        raw_backbone = backbone.backbone if hasattr(backbone, "backbone") else backbone
        feature_dim = backbone.feature_dim if hasattr(backbone, "feature_dim") else raw_backbone.feature_dim
        centroid_momentum = model_cfg.get("centroid_momentum", 0.9)
        return MIDTWrapper(
            backbone=raw_backbone,
            feature_dim=feature_dim,
            centroid_momentum=centroid_momentum,
            num_methods=len(ALL_METHODS) + 1,  # +1 for "real"
        )
```

- [ ] **Step 2: Verify factory import**

Run: `cd /home/z/project/deepfake_detection && python -c "from src.deepfake_detection.models.factory import build_model; m = build_model({'name': 'midt_efficientnet_b0'}); print(type(m), m.feature_dim)"`
Expected: `<class 'deepfake_detection.models.midt.MIDTWrapper'> 1280`

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/models/factory.py
git commit -m "feat: register MIDT models in factory"
```

---

### Task 4: Add MIDT Training Step

**Files:**
- Modify: `src/deepfake_detection/engine/trainers.py`

- [ ] **Step 1: Add midt_step function**

Add after `bgface_contrast_step` (after line 85) in `src/deepfake_detection/engine/trainers.py`:

```python
def midt_step(model, batch, device, alpha=1.0, beta=0.1):
    """MIDT training step: L_raw + alpha * L_dir + beta * L_align."""
    raw_model = model.module if hasattr(model, "module") else model
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    method_ids = batch["method_id"].to(device)

    raw_logits, dir_logits, features = raw_model.forward_dual(images)
    loss_raw = F.cross_entropy(raw_logits, labels)
    loss_dir = F.cross_entropy(dir_logits, labels)

    f_hat = F.normalize(features, dim=-1)
    raw_model.update_centroids(f_hat.detach(), labels, method_ids)
    loss_align = raw_model.compute_alignment_loss()

    total_loss = loss_raw + alpha * loss_dir + beta * loss_align
    return total_loss, raw_logits
```

- [ ] **Step 2: Add midt dispatch in run_train_epoch**

In `run_train_epoch` (line 88), add midt loss config extraction after line 100:

```python
    is_midt = loss_name == "midt"
    midt_alpha = cfg.get("loss", {}).get("midt_alpha", 1.0)
    midt_beta = cfg.get("loss", {}).get("midt_beta", 0.1)
```

Add dispatch branch in the `if/elif` chain (before the `else` on line 112):

```python
            elif is_midt:
                loss, _ = midt_step(model, batch, device, midt_alpha, midt_beta)
```

The chain becomes:

```python
            if is_contrastive:
                loss, _ = contrastive_step(model, batch, device, lambda_align, temperature)
            elif is_prompt_itc:
                loss, _ = prompt_itc_step(model, batch, device, lambda_itc, temperature)
            elif is_prompt:
                loss, _, _ = prompt_contrast_step(model, batch, device, beta)
            elif is_bgface:
                loss, _ = bgface_contrast_step(model, batch, device, lambda_align, temperature)
            elif is_midt:
                loss, _ = midt_step(model, batch, device, midt_alpha, midt_beta)
            else:
                loss, _ = classification_step(model, batch, device)
```

- [ ] **Step 3: Verify trainer imports**

Run: `cd /home/z/project/deepfake_detection && python -c "from src.deepfake_detection.engine.trainers import midt_step; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/deepfake_detection/engine/trainers.py
git commit -m "feat: add midt_step and dispatch in training loop"
```

---

### Task 5: Update Evaluation for Dual-Head Inference

**Files:**
- Modify: `src/deepfake_detection/engine/trainers.py:126-148`

- [ ] **Step 1: Update run_sample_eval_epoch to support MIDT dual-head**

Replace `run_sample_eval_epoch` (lines 126-148) with:

```python
@torch.no_grad()
def run_sample_eval_epoch(model, dataloader, device, cfg, output_dir: str | None = None, split_name: str = "eval"):
    model.eval()
    rows = []
    raw_model = model.module if hasattr(model, "module") else model
    is_midt = isinstance(raw_model, MIDTWrapper) if hasattr(__import__("deepfake_detection.models.midt", fromlist=["MIDTWrapper"]), "MIDTWrapper") else False

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            features = raw_model.extract_features(images.float()).float()

            if is_midt:
                # Evaluate using direction head (primary deployment mode)
                f_hat = F.normalize(features, dim=-1)
                logits = raw_model.dir_head(f_hat.float())
            else:
                classifier_features = F.normalize(features, dim=-1) if getattr(raw_model, "normalize_features", False) else features
                logits = raw_model.classifier(classifier_features.float())

            prob_fake = torch.softmax(logits, dim=1)[:, 1]
            feature_norm = features.norm(dim=-1)
        rows.extend(build_sample_rows(batch, logits, prob_fake, feature_norm))

    rows = gather_rows_across_ranks(rows)
    summary = summarize_sample_rows(rows)
    metrics_05 = video_metrics_from_rows(rows, threshold=0.5)
    result = {**metrics_05, **summary, "loss": 0.0}
    if output_dir is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        write_sample_rows_csv(rows, f"{output_dir}/{split_name}_sample_rows.csv")
        write_eval_summary(summary, metrics_05, f"{output_dir}/{split_name}_summary.json")
    return result
```

Wait — the `isinstance` check with dynamic import is fragile. Use a cleaner approach: check `hasattr(raw_model, "dir_head")`.

```python
@torch.no_grad()
def run_sample_eval_epoch(model, dataloader, device, cfg, output_dir: str | None = None, split_name: str = "eval"):
    model.eval()
    rows = []
    raw_model = model.module if hasattr(model, "module") else model
    is_midt = hasattr(raw_model, "dir_head")

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            features = raw_model.extract_features(images.float()).float()

            if is_midt:
                f_hat = F.normalize(features, dim=-1)
                logits = raw_model.dir_head(f_hat.float())
            else:
                classifier_features = F.normalize(features, dim=-1) if getattr(raw_model, "normalize_features", False) else features
                logits = raw_model.classifier(classifier_features.float())

            prob_fake = torch.softmax(logits, dim=1)[:, 1]
            feature_norm = features.norm(dim=-1)
        rows.extend(build_sample_rows(batch, logits, prob_fake, feature_norm))

    rows = gather_rows_across_ranks(rows)
    summary = summarize_sample_rows(rows)
    metrics_05 = video_metrics_from_rows(rows, threshold=0.5)
    result = {**metrics_05, **summary, "loss": 0.0}
    if output_dir is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        write_sample_rows_csv(rows, f"{output_dir}/{split_name}_sample_rows.csv")
        write_eval_summary(summary, metrics_05, f"{output_dir}/{split_name}_summary.json")
    return result
```

- [ ] **Step 2: Verify eval function compiles**

Run: `cd /home/z/project/deepfake_detection && python -c "from src.deepfake_detection.engine.trainers import run_sample_eval_epoch; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/deepfake_detection/engine/trainers.py
git commit -m "feat: update eval to use direction head for MIDT models"
```

---

### Task 6: Update train.py for MIDT

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Add MIDT centroid save/load**

In `train.py`, after the model save block (around line 257), add centroid state saving:

Find this block:
```python
                torch.save({"epoch": epoch, "model_state_dict": state, "auc": best_auc,
                            "eer": val_metrics["eer"], "acc": val_metrics["acc"]},
                           os.path.join(save_dir, "output", "best_model.pth"))
```

Replace with:
```python
                ckpt_data = {"epoch": epoch, "model_state_dict": state, "auc": best_auc,
                             "eer": val_metrics["eer"], "acc": val_metrics["acc"]}
                raw_model_ref = model.module if hasattr(model, "module") else model
                if hasattr(raw_model_ref, "centroid_state_dict"):
                    ckpt_data["centroid_state"] = raw_model_ref.centroid_state_dict()
                torch.save(ckpt_data, os.path.join(save_dir, "output", "best_model.pth"))
```

- [ ] **Step 2: Add centroid state restore at final eval**

After loading best model (around line 278):

Find:
```python
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
```

Replace with:
```python
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if "centroid_state" in ckpt and hasattr(raw_model, "load_centroid_state"):
        raw_model.load_centroid_state(ckpt["centroid_state"])
```

- [ ] **Step 3: Add DDP find_unused_parameters for MIDT**

In the DDP wrapping block (line 200), update the condition:

Find:
```python
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=(
            loss_name == "cross_entropy_plus_prompt" or "dinov2" in model_name))
```

Replace with:
```python
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=(
            loss_name == "cross_entropy_plus_prompt" or "dinov2" in model_name or loss_name == "midt"))
```

MIDT needs `find_unused_parameters=True` because `dir_head` parameters may not receive gradients in every backward pass depending on the loss configuration.

- [ ] **Step 4: Verify train.py parses correctly**

Run: `cd /home/z/project/deepfake_detection && python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: save/load centroid state and DDP compat for MIDT"
```

---

### Task 7: Create MIDT Configs

**Files:**
- Create: `configs/midt_efficientnet.yaml`
- Create: `configs/midt_dinov2.yaml`
- Create: `configs/midt_clip.yaml`

- [ ] **Step 1: Create EfficientNet MIDT config**

```yaml
# configs/midt_efficientnet.yaml
_base_: configs/efficientnet_base.yaml

model:
  name: midt_efficientnet_b0
  centroid_momentum: 0.9

loss:
  name: midt
  midt_alpha: 1.0
  midt_beta: 0.1
```

- [ ] **Step 2: Create DINOv2 MIDT config**

```yaml
# configs/midt_dinov2.yaml
_base_: configs/dinov2_base.yaml

model:
  name: midt_dinov2_vitb14
  centroid_momentum: 0.9

loss:
  name: midt
  midt_alpha: 1.0
  midt_beta: 0.1
```

- [ ] **Step 3: Create CLIP MIDT config**

```yaml
# configs/midt_clip.yaml
_base_: configs/clip_base.yaml

model:
  name: midt_clip_finetune
  centroid_momentum: 0.9

loss:
  name: midt
  midt_alpha: 1.0
  midt_beta: 0.1
```

- [ ] **Step 4: Verify configs load and merge correctly**

Run: `cd /home/z/project/deepfake_detection && python -c "from train import load_config; cfg = load_config('configs/midt_efficientnet.yaml'); print(cfg['model']['name'], cfg['loss']['name'], cfg['loss']['midt_alpha'], cfg['loss']['midt_beta'])"`
Expected: `midt_efficientnet_b0 midt 1.0 0.1`

- [ ] **Step 5: Commit**

```bash
git add configs/midt_efficientnet.yaml configs/midt_dinov2.yaml configs/midt_clip.yaml
git commit -m "feat: add MIDT config files for all three backbones"
```

---

### Task 8: Dry-Run Smoke Test (Single GPU, 2 Epochs)

**Files:**
- None (validation only)

- [ ] **Step 1: Run a 2-epoch dry test on single GPU**

This validates the full pipeline: data loading → model creation → forward/backward → EMA centroid update → eval → save checkpoint.

```bash
cd /home/z/project/deepfake_detection && torchrun --nproc_per_node=1 train.py \
  --config configs/midt_efficientnet.yaml \
  train.epochs=2 \
  train.seed=42 \
  train.output_dir=outputs/midt_smoke_test \
  experiment_name=midt_smoke_test_effnet_seed42
```

Expected: Training completes 2 epochs without error. Log shows loss values, val AUC, and checkpoint saved.

- [ ] **Step 2: Verify checkpoint contains centroid_state**

```bash
python -c "import torch; ckpt = torch.load('outputs/midt_smoke_test/midt_smoke_test_effnet_seed42/output/best_model.pth', map_location='cpu'); print('Keys:', list(ckpt.keys())); print('Centroid initialized:', ckpt.get('centroid_state', {}).get('centroid_initialized', 'N/A'))"
```

Expected: Keys include `centroid_state`, `centroid_initialized` has some `True` entries.

- [ ] **Step 3: Clean up smoke test output**

```bash
rm -rf outputs/midt_smoke_test
```

---

### Task 9: Phase 1 Hyperparameter Grid — EfficientNet-B0

**Files:**
- Create: `scripts/run_midt_phase1.sh`

- [ ] **Step 1: Create the Phase 1 launch script**

This script runs the beta grid (0, 0.01, 0.1, 1.0) with alpha=1.0 across 5 seeds, then alpha grid (0.5, 0.1) with best beta.

```bash
#!/bin/bash
# scripts/run_midt_phase1.sh
# Phase 1: EfficientNet-B0 MIDT ablation on 6xV100

set -euo pipefail
GPUS=6
SEEDS=(7 42 123 999 2048)
BETAS=(0.0 0.01 0.1 1.0)
ALPHA=1.0
EPOCHS=15
BATCH=128

# Beta grid: alpha=1.0, vary beta
for beta in "${BETAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp_name="midt_effnet_a${ALPHA}_b${beta}_s${seed}"
        echo "=== Launching $exp_name ==="
        torchrun --nproc_per_node=$GPUS train.py \
            --config configs/midt_efficientnet.yaml \
            train.epochs=$EPOCHS \
            train.seed=$seed \
            train.per_gpu_batch=$BATCH \
            loss.midt_alpha=$ALPHA \
            loss.midt_beta=$beta \
            train.output_dir=outputs/midt_phase1 \
            experiment_name=$exp_name
    done
done

# Alpha grid: best_beta (set manually after beta results), vary alpha
# Uncomment and set BEST_BETA after beta grid completes:
# BEST_BETA=0.1
# for alpha in 0.5 0.1; do
#     for seed in "${SEEDS[@]}"; do
#         exp_name="midt_effnet_a${alpha}_b${BEST_BETA}_s${seed}"
#         echo "=== Launching $exp_name ==="
#         torchrun --nproc_per_node=$GPUS train.py \
#             --config configs/midt_efficientnet.yaml \
#             train.epochs=$EPOCHS \
#             train.seed=$seed \
#             train.per_gpu_batch=$BATCH \
#             loss.midt_alpha=$alpha \
#             loss.midt_beta=$BEST_BETA \
#             train.output_dir=outputs/midt_phase1 \
#             experiment_name=$exp_name
#     done
# done
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/run_midt_phase1.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_midt_phase1.sh
git commit -m "feat: add Phase 1 MIDT launch script for EfficientNet ablation"
```

---

### Task 10: Evaluation Script for MIDT Results

**Files:**
- Create: `scripts/eval_midt.py`

- [ ] **Step 1: Create evaluation script that produces comparable CSV format**

This script loads a trained MIDT checkpoint and evaluates with both raw_head and dir_head, producing results in the same format as `auc_video_level_60rows.csv` and `dfd_balanced_thresholds_45rows.csv`.

```python
#!/usr/bin/env python
"""Evaluate MIDT models with both raw and direction heads.

Usage:
    python scripts/eval_midt.py --checkpoint <path>/best_model.pth --config configs/midt_efficientnet.yaml
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_eval_loader
from deepfake_detection.engine.metrics import (
    update_video_aggregate,
    video_aggregates_to_predictions,
    compute_auc,
)
from deepfake_detection.engine.ddp import init_ddp
from train import load_config


@torch.no_grad()
def evaluate_midt(model, loader, device):
    """Evaluate MIDT model, returning scores from both heads."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_scores = {}
    dir_scores = {}
    labels = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = raw_model.extract_features(images.float()).float()

        # Raw head
        raw_logits = raw_model.raw_head(features)
        raw_prob = torch.softmax(raw_logits, dim=1)[:, 1].cpu().numpy()

        # Direction head
        f_hat = F.normalize(features, dim=-1)
        dir_logits = raw_model.dir_head(f_hat)
        dir_prob = torch.softmax(dir_logits, dim=1)[:, 1].cpu().numpy()

        for i in range(len(batch["video_id"])):
            vid = batch["video_id"][i]
            label = batch["label"][i].item()
            raw_scores.setdefault(vid, []).append(float(raw_prob[i]))
            dir_scores.setdefault(vid, []).append(float(dir_prob[i]))
            labels[vid] = label

    return raw_scores, dir_scores, labels


def video_auc(scores, labels):
    vids = sorted(scores.keys())
    y_true = np.array([labels[v] for v in vids])
    y_score = np.array([np.mean(scores[v]) for v in vids])
    return roc_auc_score(y_true, y_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "centroid_state" in ckpt and hasattr(model, "load_centroid_state"):
        model.load_centroid_state(ckpt["centroid_state"])
    model.eval()

    domains = {"ffpp": "ffpp", "cdf": "cdf", "dfd": "dfd"}
    results = []

    for domain_name, domain_key in domains.items():
        loader = build_eval_loader(cfg, domain=domain_key, distributed=False)
        raw_scores, dir_scores, labels = evaluate_midt(model, loader, device)

        raw_auc = video_auc(raw_scores, labels)
        dir_auc = video_auc(dir_scores, labels)
        results.append({"domain": domain_name, "raw_auc": raw_auc, "dir_auc": dir_auc})
        print(f"{domain_name}: raw_auc={raw_auc:.4f}, dir_auc={dir_auc:.4f}")

    print("\nSummary:")
    for r in results:
        print(f"  {r['domain']}: raw={r['raw_auc']:.4f} dir={r['dir_auc']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/eval_midt.py
git commit -m "feat: add MIDT evaluation script with dual-head scoring"
```

---

## Self-Review

**1. Spec coverage:**
- Dual-head architecture (raw + dir): Task 1 ✓
- L_align with EMA centroids: Task 1 ✓
- Method labels in batch: Task 2 ✓
- Factory registration: Task 3 ✓
- Training step dispatch: Task 4 ✓
- Evaluation with dir head: Task 5 ✓
- Centroid save/load: Task 6 ✓
- Configs for 3 backbones: Task 7 ✓
- Smoke test: Task 8 ✓
- Phase 1 launch script: Task 9 ✓
- Eval script: Task 10 ✓

**2. Placeholder scan:** No TBD/TODO found. All code blocks contain complete implementations.

**3. Type consistency:**
- `method_id` is `torch.long` tensor from dataset → used as index in `update_centroids` ✓
- `forward_dual` returns `(raw_logits, dir_logits, features)` → consumed by `midt_step` ✓
- `centroid_state_dict()` / `load_centroid_state()` signatures match save/load in train.py ✓
- `num_methods = len(ALL_METHODS) + 1` = 17, matching `METHOD_TO_INDEX` range ✓
