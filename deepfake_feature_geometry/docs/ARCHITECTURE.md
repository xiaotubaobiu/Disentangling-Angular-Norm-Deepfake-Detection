# Architecture & Module Guide

This document explains **what every part of the repository is and what it is for**.
The project studies whether deepfake detection transfers across datasets through
feature **direction**, feature **length** (L2 norm), or both, by evaluating the
same raw checkpoints under three inference modes: `raw`, `feature_norm`
(direction only), and `feature_length` (norm only).

Read `README.md` first for the scientific summary and CLI usage. This file is the
code map.

---

## Entrypoints

Scripts are not at the repository root: each one lives in the dated experiment
phase that used it (`experiments/<date>_<topic>/scripts/`). `src/` holds only
code shared across phases. See `experiments/README.md` for the phase index.

| File | What it is | What it's for |
|------|-----------|---------------|
| `experiments/2026-05-05_paper_feature_geometry/scripts/train.py` | Training entrypoint (DDP-aware). | Builds a model + loaders from a YAML config, runs warmup+cosine schedule with AMP, optional EMA, early-stops on val AUC, then evaluates the best checkpoint on FF++/CDF/DFD and writes `results_summary.txt` / `meta.json`. |
| `experiments/2026-05-05_paper_feature_geometry/scripts/evaluate.py` | Standard evaluation entrypoint. | Loads a checkpoint and reports FF++/CDF/DFD video-level AUC/EER/ACC via the same eval loaders. |
| `src/config_io.py` | Shared config loader. | `load_config` with `_base_` inheritance, used by the training entrypoint and by every script that rebuilds a model from a checkpoint config. |

---

## `src/data/` — datasets & indexing

| Module | What it does | Used by |
|--------|--------------|---------|
| `constants.py` | Central taxonomy: `ALL_METHODS` (16 DF40 forgery methods), image size, real/fake label ints, prompt texts. | Everywhere method lists / labels are needed. |
| `index_ffpp.py` | Reads DF40/DeepfakeBench-style JSON indexes (`dataset_json_update/{method}_{domain}.json`) into `FrameRecord`s; also DFD JSON loader and aligned-triplet builder. | `builders.py`, diagnosis feature extraction. |
| `index_external.py` | Generic binary real/fake external dataset indexer with on-disk cache (seeded frame sampling). | `evaluate_norm_correction.py` external splits. |
| `sampling.py` | `sample_uniform_frame_indices` (evenly spaced frame picks) and real-video balancing. | Loader frame subsampling. |
| `builders.py` | **The loader factory.** `build_train_loader`, `build_val_loader`, `build_eval_loader(domain=ffpp/cdf/dfd)`, `build_bgface_train_loader`, `build_external_eval_loader`. Handles per-method record collection, real dedup, normalization mean/std selection (CLIP vs ImageNet), DDP samplers. | `train.py`, `evaluate.py`, diagnosis, figure scripts. |
| `datasets.py` | Torch `Dataset`s: `FrameClassificationDataset` (main train/eval), `AlignedTripletDataset` (shared-augmentation real/fake pairs), `BgFaceTripletDataset` (background/face contrastive). | Built by `builders.py`. |
| `transforms.py` | Albumentations pipelines: `build_rgb_augment` (aug **+ Normalize**), `build_eval_transform` (Normalize only), `build_shared_spatial_augment` / `apply_shared_transform_pair` (spatial aug **without** Normalize, applied identically to a real/fake pair). | Datasets. |
| `crops.py` | `expand_box` (context-scale a face box, clamped to image) and `crop_region` (safe crop+resize, guards degenerate boxes). | Triplet datasets. |
| `bg_patches.py` | Samples background patches that avoid the face box. | `BgFaceTripletDataset`. |
| `frequency.py` | `rgb_to_frequency_map` (DCT-based frequency features). | Optional frequency input in datasets. |

Normalization contract: datasets apply `A.Normalize` and convert to tensors
**without** dividing by 255. Dividing after Normalize is a double-normalization
bug (fixed during audit).

---

## `src/models/` — backbones & heads

`factory.build_model(cfg)` dispatches on `model.name`. Every classifier exposes
`extract_features(images)` (raw pooled backbone features), a `.classifier`
`nn.Linear(feature_dim, 2)` head, and a `normalize_features` flag (L2-normalize
before the head when true — this is how "normtrain" models are trained).

| Module | Model | Notes |
|--------|-------|-------|
| `clip_classifier.py` | `CLIPFineTuneBinaryClassifier` (`clip_finetune`) | CLIP ViT-B/16 visual tower + linear head. The main paper backbone. Exposes `.visual`. |
| `dinov2.py` | `DINOv2BinaryClassifier` (`dinov2_vitb14`) | DINOv2 ViT-B/14 via torch.hub. Exposes `.backbone`. |
| `efficientnet.py` | `EfficientNetBinaryClassifier` (`efficientnet_b0`) | torchvision EfficientNet-B0, classifier replaced. |
| `xception.py` | `XceptionBinaryClassifier` (`xception`) | timm legacy_xception. **Legacy**: kept for loading old checkpoints; training configs archived in `.deleted_configs_backup_20260729.tar.gz` (2026-07-29), so `launch_training.py` refuses this backbone. Supported training backbones are CLIP / DINOv2 / EfficientNet. |
| `clip_prompt.py` | `CLIPPromptBinaryClassifier` (`clip_prompt`) | CLIP + fixed text-prompt logits + projections for image-text contrastive. |
| `clip_bgcontrast.py` | `CLIPBgFaceContrastModel` (`clip_prompt_bgcontrast`) | CLIP with background/face contrastive heads. |
| `factory.py` | `build_model` | The only place model names are resolved. |

---

## `src/losses/`

| Function (`contrastive.py`) | What it computes |
|------|------|
| `same_frame_contrastive_loss` | 2-way (real vs fake face) contrastive against a background anchor. |
| `infonce_bg_face_loss` | InfoNCE with K background patches per sample; positive = (bg, own real face), negatives = own fake + cross-sample real/fake. Self-pair masking fixed during audit (was an `IndexError`). |
| `image_text_contrastive_loss` | CLIP-style image↔text alignment by label. |

---

## `src/engine/`

| Module | What it does |
|--------|--------------|
| `trainers.py` | Per-step loss functions (`classification_step`, `contrastive_step`, `prompt_*_step`, `bgface_contrast_step`), the AMP train loop `run_train_epoch`, and sample-based eval `run_sample_eval_epoch`/`run_eval_epoch`. |
| `metrics.py` | `compute_auc`/`compute_eer`/`compute_acc`, streaming video-score aggregation helpers (`update_video_aggregate`/`merge_video_aggregates`/`video_aggregates_to_predictions`). |
| `ddp.py` | Minimal DDP helpers: `init_ddp`, `is_main_process`, `barrier`. |

---

## `src/evaluation/`

| Module | What it does |
|--------|--------------|
| `norm_correction.py` | The norm-manipulation primitives used by the paper eval: `apply_partial_norm(features, alpha)` (α=0 raw, α=1 direction), `apply_mean_scale_norm`, `fuse_scores`, `best_accuracy_threshold`. **Single source of truth** (the duplicate `norm_modes.py` was deleted). |
| `sample_eval.py` | Per-frame sample-row collection, cross-rank gathering, and **video aggregation** (`sample_rows_to_video_predictions` groups by `video_id`). Note: this groups by video_id only; the diagnosis pipeline uses a stricter (method_id + video_id) key — see below. |

---

## `src/diagnosis/` — the frame-level analysis pipeline

Read-only pipeline over the 30 retained checkpoints. It extracts raw features
once per (checkpoint, domain), derives inference-mode scores, aggregates to
videos, computes per-method metrics + bootstrap CIs, and emits hypothesis
verdicts and figures.

| Module | What it does |
|--------|--------------|
| `feature_extractor.py` | Runs a backbone once over an eval loader, returns aligned raw features + per-frame method/label/video_id (read from `loader.dataset.records`, **not** by video_id lookup, to avoid losing methods that share a pair_id). |
| `infer_modes.py` | Derives `raw` / `direction` / `length` (rawtrain) and `normtrain_norm` scores from raw features + classifier weight/bias. `length` prob_fake is a non-saturating logistic of the standardized norm (the naive `softmax([-‖h‖,+‖h‖])` saturates to 1.0 — fixed during audit). |
| `method_map.py` | Normalizes `FrameRecord.method` into the diagnostic taxonomy (`real`, `dfd_native`, or verbatim DF40 method). |
| `aggregation.py` | Strict video aggregation `P_video = mean_t(softmax(logits_t)[fake])`, grouping by **(checkpoint, backbone, seed, train_mode, infer_mode, domain, method_id, video_id)**. Including `method_id` is essential: in FF++/CDF one pair_id is reused across methods, so grouping by video_id alone would merge distinct fake videos. |
| `method_metrics.py` | Per-(fake method vs same-domain reals) binary metrics: AUC, balanced acc at 0.5, optimal-threshold balanced acc, threshold regret, score distributions; plus an `all_fake` pooled row. |
| `diagnostics.py` | Domain-shift (`D_domain`), method dispersion (`D_method`, `T_method`), feature-norm correlations/quartile buckets, and `delta_auc_per_method` (direction−raw AUC paired **within backbone** — cross-backbone cartesian bug fixed during audit). |
| `bootstrap.py` | Video-level paired bootstrap of AUC(mode_a)−AUC(mode_b) with the same resampled video indices for both modes. |
| `hypothesis_summary.py` | Turns the above into H1–H5 verdicts (does direction reduce domain shift, is the 0.48 threshold stable, are method distributions distinct, does normtrain trade error for dispersion, is the norm a confidence-polarizer). |
| `plotting.py` | The 6 diagnostic figures (delta-AUC heatmap, score-shift, threshold distributions, norm-quartile-vs-error, etc.). |
| `checkpoint_index.py` | Discovers retained checkpoints under `checkpoints/<backbone>/<mode>/seed_*` and resolves their config paths. |

---

## `src/study_threshold/` — cross-domain threshold transfer

| Module | What it's for |
|---|---|
| `io.py` | source (FF++ val cache) / target (phase-2 parquet) loading, per-method real de-duplication, wide raw+direction frame table, `(method_id, video_id)` video aggregation |
| `calibration.py` | temperature (frame level) and video-level Platt fitted **on the source domain only**, plus ECE / NLL / Brier for the three probability views |
| `thresholds.py` | `t_BA` / `t_equal` / `t_FPR1` / `t_FPR5` / calibration-implied thresholds, vectorised rate curves, confusion metrics, oracle BA |
| `transfer.py` | apply frozen thresholds to the target domain, transfer regret, fixed-FPR error, mis-specified `mean_prob@0.5` baseline |
| `score_shift.py` | class-conditional distribution stats, common vs differential shift, scale, Wasserstein / KS, failure-cause label |
| `bootstrap.py` | video-cluster bootstrap wrapping `study.cluster_bootstrap`, with a fast path and paired raw/direction resampling |
| `reporting.py` | phase path resolution, CSV/markdown table and figure helpers |

## Phase scripts

Paths below are relative to the owning phase's `scripts/` directory:
`2026-05-05_paper_feature_geometry` (training / paper evaluation / figures / data
prep), `2026-07-21_frame_midterm_diagnosis` (frame-level diagnosis),
`2026-07-23_four_questions` (`00`–`05`, `run_all.py`),
`2026-07-27_pooling_saturation` (`06`–`09`),
`2026-07-30_threshold_transfer` (`00`–`06`, `run_all.py`).

| Script | What it's for |
|--------|---------------|
| `evaluate_norm_correction.py` | The paper's raw/direction/length (and partial-norm / fusion) evaluation over FF++/CDF/DFD and external datasets. |
| `summarize_paper_results.py` | Prints the grouped AUC table from a retained results CSV. |
| `frame_midt_diagnose.py` | Main entrypoint of the `diagnosis/` pipeline (extract → aggregate stages, parallel-safe per-checkpoint shards). |
| `cdf_balanced_dir_raw.py` | Same as above but downsamples fakes to the real count (178) with a **shared paired subsample** for raw and direction (rng-reuse pairing bug fixed during audit). |
| `generate_bw_figures.py` | Grayscale-safe thesis figures (norm histograms, t-SNE, cosine heatmaps, calibration, threshold curves). Visualization only — not the source of paper AUC. |
| `prepare_dfd_df40.py` | Prepares DFD into DF40-style JSON/frame layout (face alignment, frame sampling). One-time data utility. |
| `download_faceforensics.py` | FaceForensics++ download helper. One-time data utility. |
| `launch_training.py` | Generates per-(backbone, variant, seed) training configs and optionally launches `torchrun`. |
| `run_diag_parallel.sh`, `wait_and_run_diag.sh`, `wait_and_verify.sh` | Shell orchestration for multi-GPU diagnosis runs. |

---

## `tests/`

| File | Locks down |
|------|-----------|
| `test_video_aggregation.py` | The strict `P_video = mean_t(softmax)` aggregation and method_id grouping. |
| `test_infer_modes_length.py` | Length-mode prob_fake is non-saturating, monotonic in the norm, and recovers the raw-norm AUC. |
| `test_losses.py` | Contrastive losses run and are finite (infonce self-pair masking regression). |
| `test_audit_fixes.py` | `delta_auc_per_method` pairs within backbone; `crop_region` survives degenerate boxes; shared spatial augment is un-normalized and varies across calls. |

Run all: `PYTHONPATH=src python -m pytest tests/ -q`.

---

## Key correctness invariants (learned the hard way)

1. **Video aggregation must key on `method_id` + `video_id`.** In FF++/CDF a
   `pair_id` is reused across all 16 forgery methods; grouping by `video_id`
   alone silently averages distinct fake videos together and fabricates a
   direction>raw advantage on CDF that disappears under correct grouping.
2. **Do not divide by 255 after `A.Normalize`.** The transform already
   standardizes; a second `/255` crushes inputs ~250×.
3. **Length-only scores must be a monotonic, non-saturating transform of the
   norm.** Feeding the raw norm (~20–50) into `softmax([-n,+n])` saturates to
   1.0 and collapses AUC to 0.5.
4. **Paired comparisons (direction vs raw) must reuse the same videos/subsample
   and pair within the same backbone.** Independent draws or cross-backbone
   joins break the pairing.

---

## Removed / fixed during the audit

Dead code removed (all were unreferenced anywhere in the repo):

- `evaluation/norm_modes.py` — whole file (duplicated `fuse_scores` /
  `apply_partial_norm` already in `norm_correction.py`, and nothing imported it).
- `data/index_cdf.py` — whole file (superseded by `index_ffpp.load_json_index`
  for the `cdf` domain; no importer).
- `engine/metrics.py::aggregate_video_predictions`, `mean_video_score`
- `data/datasets.py::collate_video_scores` (+ unused `defaultdict` import)
- `data/sampling.py::balance_real_video_ids` (+ unused `random` import)
- `data/index_ffpp.py::build_aligned_pair_key`
- `models/clip_classifier.py::cosine_classifier_logits`
- `models/midt.py` — whole file (`MIDTWrapper`); orphan not wired into
  `factory.build_model` and unused by any pipeline.

Bugs fixed (each covered by a test or a re-run):

1. `diagnosis/infer_modes.py` — length prob_fake softmax saturation.
2. `losses/contrastive.py` — `infonce_bg_face_loss` self-pair mask `IndexError`.
3. `data/datasets.py` — `BgFaceTripletDataset` double-normalization (`/255`
   after `A.Normalize`).
4. `data/datasets.py` + `data/transforms.py` — `AlignedTripletDataset`
   normalization inconsistency + frozen shared-augmentation seed.
5. `data/crops.py` — `crop_region` crash on degenerate boxes.
6. `diagnosis/diagnostics.py` — `delta_auc_per_method` cross-backbone cartesian
   product (720 rows → correct 240).
7. `diagnosis/hypothesis_summary.py` — H3 seed-noise floor inflated by pooling
   backbone variance (2.53× too large).
8. `cdf_balanced_dir_raw.py` — rng reuse produced unpaired raw/direction
   subsamples.
9. `prepare_dfd_df40.py` — face selection picked the face farthest from
   center (a `x and y` expression bug) instead of the largest face.

</content>
</invoke>
