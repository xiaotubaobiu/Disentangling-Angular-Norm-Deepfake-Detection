# MIDT: Method-Invariant Direction Training for Cross-Domain Deepfake Detection

## 1. Paper Narrative

### Target Venue
TIFS (IEEE Transactions on Information Forensics and Security) or Pattern Recognition.

### Core Story

**Title candidate**: "Method-Invariant Direction Training for Generalizable Deepfake Detection" or "Learning Method-Invariant Feature Directions for Zero-Calibration Cross-Domain Deepfake Detection"

**Story arc**:

1. **Empirical Discovery** (Section 3): Through systematic feature geometry decomposition (direction/norm) across 60 trained models (3 backbones × 5 seeds × 2 training modes), we discover that feature direction is the primary carrier of cross-domain generalization. Test-time L2 normalization (direction) consistently outperforms raw features on cross-domain AUC (15/15 models agree). Moreover, direction inference yields a decision threshold locked at ~0.48 across all backbones and seeds — near zero-calibration deployment.

2. **Analysis of the Gap** (Section 3): Despite direction's superiority on cross-domain AUC and threshold stability, it lags behind normtrain on DFD balanced accuracy by ~2%. Root cause: raw training's direction is only *implicitly* method-invariant — it happens to generalize because the backbone compensates for norm shortcuts, but nothing guarantees cross-method consistency. Normtrain explicitly trains direction but sacrifices backbone richness (training loss ~0.23 vs ~0.01 for raw), producing less discriminative features.

3. **Method** (Section 4): We propose MIDT — Method-Invariant Direction Training. Three components:
   - L_raw on raw features: backbone sees full information, learns rich representations
   - L_dir on L2-normalized features: direction classifier is naturally well-calibrated
   - L_align: explicit inter-method direction alignment — penalizes dispersion of per-method, per-class direction centroids
   The key insight: L_align makes the direction space a true "method-orthogonal intersection" — the shared forgery signal across all methods, which is exactly what transfers to unseen domains.

4. **Validation** (Section 5): Experiments on 3 backbones × 5 seeds show MIDT's direction head achieves: CDF AUC ≥ raw-train direction, DFD balanced accuracy approaching or exceeding normtrain, with zero-calibration deployment (threshold ~0.48).

### Key Contributions

1. A systematic feature geometry analysis framework for understanding deepfake detection generalization (direction vs norm decomposition)
2. The empirical finding that test-time L2 normalization is a free lunch for cross-domain detection with stable thresholds
3. MIDT: a training method that explicitly learns method-invariant feature directions while preserving backbone richness
4. A zero-calibration cross-domain detector: fixed threshold ~0.5, no target domain data needed

---

## 2. Method Design: MIDT

### 2.1 Architecture

```
Image → Backbone(f) → f ∈ R^D
                          ├── f̂ = L2_normalize(f)
                          ├── W_raw · f + b_raw      → L_raw (CE)
                          ├── W_dir · f̂ + b_dir      → L_dir (CE)
                          └── L_align (direction alignment)
```

### 2.2 Loss Function

```
L = L_raw + α · L_dir + β · L_align
```

Where:
- L_raw = CE(W_raw · f + b_raw, y)
- L_dir = CE(W_dir · f̂ + b_dir, y)
- L_align = Σ_{c ∈ {real,fake}} (1/M²) Σ_{m≠m'} [1 - cos(μ̂_c^m, μ̂_c^{m'})]
  - μ̂_c^m = EMA of mean(f̂[label==c, method==m]) per method m and class c
  - M = number of forgery methods in the batch (or total if using global EMA)

### 2.3 EMA Direction Centroids

Maintained across training for stability (avoid per-batch noise):

```python
# Per method m, class c:
if n_m_c > 0:  # batch has samples from method m, class c
    batch_centroid = f_hat[label==c, method==m].mean(dim=0)
    centroid_ema[c][m] = momentum * centroid_ema[c][m] + (1-momentum) * batch_centroid
```

momentum = 0.9 default.

L_align computed on EMA centroids, not per-batch centroids. This means:
- The loss is smooth and stable
- Methods not present in current batch still contribute to the loss
- The alignment is global, not batch-local

### 2.4 Inference

**Primary deployment**: Use direction head only.

```python
f = backbone(image)
f_hat = L2_normalize(f)
prob = sigmoid(W_dir · f_hat + b_dir)
prediction = (prob > 0.5)  # fixed threshold, no calibration needed
```

**Optional**: For within-domain scenarios where target domain data is available for calibration, raw head can provide marginal AUC improvement.

### 2.5 Implementation Changes

Files to modify:
- `src/deepfake_detection/models/`: Add MIDT wrapper that produces dual logits
- `src/deepfake_detection/engine/trainers.py`: Add L_dir and L_align to training step
- `src/deepfake_detection/data/datasets.py`: Ensure method labels are available in batch
- `train.py`: Add α, β hyperparameters, EMA centroid state
- `evaluate.py`: Support evaluating with direction head

The method label is already available in the data pipeline — FrameClassificationDataset returns video_id from which method can be extracted. Need to add explicit method field to batch.

---

## 3. Experimental Plan

### 3.1 Phase 1: Single-Backbone Validation (EfficientNet-B0)

**Goal**: Validate MIDT works before scaling up.

| Experiment | Config | Count |
|-----------|--------|-------|
| Baseline raw | EfficientNet, raw, 5 seeds | 5 (existing) |
| Baseline normtrain | EfficientNet, normtrain, 5 seeds | 5 (existing) |
| MIDT α=1 β=0 | EfficientNet, MIDT, 5 seeds | 5 (new) |
| MIDT α=1 β=0.01 | EfficientNet, MIDT, 5 seeds | 5 (new) |
| MIDT α=1 β=0.1 | EfficientNet, MIDT, 5 seeds | 5 (new) |
| MIDT α=1 β=1.0 | EfficientNet, MIDT, 5 seeds | 5 (new) |
| MIDT α=0.5 β=best | EfficientNet, MIDT, 5 seeds | 5 (new) |
| MIDT α=0.1 β=best | EfficientNet, MIDT, 5 seeds | 5 (new) |

Phase 1 total: ~30 new models on 6×V100. EfficientNet-B0 trains ~6 min/epoch × 15 epochs = ~1.5h per model. 30 models × 1.5h / 6 GPUs = ~7.5h wall time.

**Success criteria**:
- MIDT direction head CDF AUC ≥ raw-train direction (0.765 for EffNet)
- MIDT direction head DFD bal_ACC ≥ raw-train direction (0.614 for EffNet)
- MIDT direction head DFD bal_ACC approaches or exceeds normtrain (0.638 for EffNet)

### 3.2 Phase 2: Scale to DINOv2

If Phase 1 succeeds, extend to DINOv2-ViT-B with best (α, β).

| Experiment | Config | Count |
|-----------|--------|-------|
| MIDT best config | DINOv2, MIDT, 5 seeds | 5 (new) |

Phase 2 total: 5 new models. DINOv2 trains ~6 min/epoch × 15 epochs = ~1.5h per model. 5 models × 1.5h / 6 GPUs = ~1.3h wall time.

### 3.3 Phase 3: Scale to CLIP (Optional)

If results are strong, extend to CLIP ViT-B/16.

| Experiment | Config | Count |
|-----------|--------|-------|
| MIDT best config | CLIP, MIDT, 5 seeds | 5 (new) |

Phase 3 total: 5 new models. Similar training time.

### 3.4 Evaluation Protocol

For every trained model, evaluate on all 3 domains:
- FF++ test: AUC (within-domain, sanity check)
- CDF test: AUC (cross-domain ranking)
- DFD test: AUC, balanced accuracy at best threshold, threshold value, real_recall, fake_recall

Both inference modes for MIDT:
- Direction head: W_dir · f̂ + b_dir
- Raw head: W_raw · f + b_raw

### 3.5 Ablation Study

| Config | L_raw | L_dir | L_align | Purpose |
|--------|-------|-------|---------|---------|
| raw only | ✓ | | | Baseline |
| raw + dir (DDHT) | ✓ | ✓ | | Isolate alignment contribution |
| raw + align | ✓ | | ✓ | Isolate direction head contribution |
| MIDT (full) | ✓ | ✓ | ✓ | Full method |
| normtrain | | | | Reference (normtrain) |

This ablation naturally decomposes the contribution of each component.

### 3.6 Analysis Experiments

1. **Inter-method cosine similarity matrix**: Before and after alignment, compute pairwise cosine similarity of per-method direction centroids. Show L_align pushes methods together.

2. **Gradient angle analysis**: Track angle between gradients from L_raw and L_dir during training. Show they provide complementary signals.

3. **Per-method AUC on FF++**: Ensure alignment doesn't hurt individual method discrimination within training domain.

4. **t-SNE colored by method**: Show methods converge in direction space after alignment.

5. **Threshold curve comparison**: Show MIDT direction head maintains ~0.48 threshold stability across seeds and domains.

---

## 4. Paper Structure

### Abstract (~200 words)
Feature direction carries cross-domain generalization signal; test-time L2 normalization is a free lunch for cross-domain AUC with stable thresholds (~0.48). But raw training's direction lacks explicit method invariance, causing a gap on heterogeneous datasets. MIDT explicitly aligns feature directions across forgery methods while preserving backbone richness via raw classification. Zero-calibration deployment, state-of-the-art cross-domain performance.

### 1. Introduction (1.5 pages)
- Deepfake detection generalization problem
- Existing approaches: domain adaptation, augmentation, ensemble
- Our approach: understand feature geometry first, then design targeted training
- Contributions list

### 2. Related Work (1 page)
- Cross-domain deepfake detection
- Feature geometry in representation learning (norm/direction analysis)
- Method-invariant representation learning (IRM, CORAL, etc.)

### 3. Feature Geometry Analysis (2.5 pages)
- 3.1 Experimental setup (3 backbones, 5 seeds, FF++/CDF/DFD)
- 3.2 Direction vs Norm decomposition
- 3.3 Cross-domain transfer analysis (CDF: direction wins)
- 3.4 Within-domain heterogeneous analysis (DFD: normtrain wins on bal_ACC)
- 3.5 Threshold stability analysis (direction ~0.48, raw wildly unstable)
- 3.6 Summary of the tension and its root cause

### 4. Method: MIDT (2 pages)
- 4.1 Motivation: method-invariant direction as orthogonal intersection
- 4.2 Architecture and loss function
- 4.3 EMA direction centroids
- 4.4 Inference: zero-calibration deployment

### 5. Experiments (3 pages)
- 5.1 Setup and baselines
- 5.2 Main results (table: raw / normtrain / DDHT / MIDT across all metrics)
- 5.3 Ablation study (table: each loss component)
- 5.4 Analysis experiments (inter-method similarity, gradient angles, t-SNE)
- 5.5 Threshold stability analysis

### 6. Conclusion (0.5 pages)

---

## 5. Compute Budget

| Phase | Models | GPU-hours | Wall time (6×V100) |
|-------|--------|-----------|-------------------|
| Phase 1: EfficientNet ablation | 30 | ~45h | ~7.5h |
| Phase 2: DINOv2 | 5 | ~7.5h | ~1.3h |
| Phase 3: CLIP (optional) | 5 | ~7.5h | ~1.3h |
| Feature extraction + eval | — | ~5h | ~1h |
| **Total** | **40** | **~65h** | **~11h** |

Fits within 6×V100 32GB. EfficientNet-B0 and DINOv2-ViT-B fit with batch_size=128. CLIP ViT-B/16 may need batch_size=64.

---

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| L_align too strong → feature collapse | Medium | Start with small β (0.01), monitor inter-method similarity |
| Direction head doesn't beat normtrain on DFD bal_ACC | Medium | The dual-loss (L_raw + L_dir) provides richer gradients than normtrain's single loss. If gap remains, emphasize threshold stability advantage |
| MIDT worse than raw-train direction on CDF | Low | L_align should help (more method-invariant), not hurt. If it hurts, reduce β |
| Method labels not clean in data pipeline | Low | Method is extractable from video_id naming convention. Need to add to batch dict |
