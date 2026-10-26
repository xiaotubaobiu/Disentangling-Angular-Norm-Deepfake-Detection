# Deepfake Detection Feature Geometry Experiments

Code and experiment records for studying whether deepfake detection transfers across datasets through feature **direction**, feature **length**, or neither.

## Main result

Three evaluation modes are compared on the same raw checkpoints:

- `raw`: the classifier uses both feature direction and feature length.
- `feature_norm` (direction): features are L2-normalized before classification, so mainly direction is used.
- `feature_length`: the feature norm alone is the score.

Headline numbers (phase `2026-05-05`):

- FF++: raw and direction-only are both near-perfect.
- CDF: direction-only > raw > length-only.
- DFD: raw > direction-only >> length-only.

The follow-up phases audited the CDF result, and it does not survive as a statement about features:

1. `direction > raw` on CDF only holds when videos are grouped by `video_id` alone, which merges different forgery methods of the same identity. Under the semantically correct `(method_id, video_id)` grouping it disappears (phase `2026-05-05`, `reports/final_results_notes.md`).
2. The residual advantage under `mean_prob` video aggregation is mostly a **sigmoid saturation + probability pooling artifact**: after temperature calibration fitted on FF++ val it goes to ~0, and its sign is method-dependent (FSGAN positive, SadTalker negative) — phase `2026-07-27`.
3. There is no shared cross-method "fake direction": same-frame displacement alignment never beats the mismatch control (6/6) — phase `2026-07-23`, Q4.
4. Ranking transferring is not the same as *deciding* transferring: with the threshold frozen on FF++ val, CDF balanced accuracy drops from 0.963 to 0.616 (oracle 0.814) while AUC stays ~0.89, because real-video scores shift by +4.15 and fake-video scores by only +0.36 — phase `2026-07-30`.

So: video-level evaluation must be calibrated (or use a non-saturating aggregator such as `mean_logit`) before any claim about direction vs length is made.

## Repository layout

```text
src/                  # reusable library code (flat: no extra package layer)
configs/              # reusable training config templates
experiments/          # one self-contained directory per dated experiment phase
tests/                # library-level tests (phase-specific tests live in the phase)
docs/ARCHITECTURE.md  # architecture notes
```

Every script lives in the phase directory it belongs to (`experiments/<date>_<topic>/scripts/`); `src/` holds only code that is shared across phases. See [`experiments/README.md`](experiments/README.md) for the phase index, the cross-phase data flow, and the mapping from pre-reorganization paths to current ones.

### `src/`

| Module | Contents |
|---|---|
| `config_io.py` | YAML config loading with `_base_` inheritance |
| `data/` | dataset indexes (FF++ / CDF / external), loaders, transforms, sampling, crops, frequency and background patches |
| `models/` | model factory plus CLIP / DINOv2 / EfficientNet classifiers, the legacy Xception classifier, and CLIP prompt / bg-contrast variants |
| `losses/` | contrastive losses |
| `engine/` | train and eval loops, metrics, DDP helpers |
| `evaluation/` | sample-level evaluation, video aggregation, norm-correction modes |
| `diagnosis/` | checkpoint discovery, feature extraction, inference modes, frame→video aggregation, bootstrap, diagnostics, plotting |
| `study/` | four-questions study library (IO, canonical keys, pair AUC, flip dissection, length, radial, Q4 alignment) |
| `study_pooling/` | pooling / saturation library (aggregators, saturation, temperature, temporal, gamma, video eval) |
| `study_threshold/` | cross-domain threshold transfer library (source/target IO, calibration, threshold rules, transfer regret, score shift, paired cluster bootstrap, reporting) |

### `experiments/`

| Phase | Topic |
|---|---|
| `2026-05-05_paper_feature_geometry` | 30 checkpoints (3 backbones × 5 seeds × raw/normtrain) and the paper-level raw / direction / length evaluation |
| `2026-07-21_frame_midterm_diagnosis` | full frame-level re-inference of all 30 checkpoints → `per_frame_scores.parquet`, the evidence base for later phases |
| `2026-07-23_four_questions` | Q1 normalization-gain split, Q2 length stability across domains, Q3 length-only reliability, Q4 shared cross-method direction (refuted) |
| `2026-07-27_pooling_saturation` | is `direction > raw` feature geometry or a saturation/pooling artifact? (mostly artifact, method-dependent) |
| `2026-07-30_threshold_transfer` | does a threshold calibrated on FF++ val survive on CDF? (no: 0.198 BA transfer regret, 1% FPR budget realises 64%, calibration does not repair it) |

## Backbones

| Status | Backbones | Notes |
|---|---|---|
| **Supported for training** | CLIP (`clip_finetune`), DINOv2 (`dinov2_vitb14`), EfficientNet (`efficientnet_b0`) | templates in `configs/`; these are the 3 backbones behind all 30 retained checkpoints and every study phase |
| **Legacy model implementation** | Xception (`xception`) | `src/models/xception.py` and its factory entry are kept so old checkpoints stay loadable, but the training configs were removed on 2026-07-29 and live only in `.deleted_configs_backup_20260729.tar.gz`. `launch_training.py` refuses Xception with an explicit message instead of emitting a config with a dangling `_base_`. |

## Installation

```bash
conda env create -f environment.yml
conda activate deepfake-detection
pip install -r requirements.txt
pip install -e .          # puts src/ on the import path
```

Without the editable install, prefix commands with `PYTHONPATH=src`. If you use an existing environment, make sure PyTorch, torchvision, OpenCV, scikit-learn, pandas, PyYAML, DuckDB, pyarrow and the CLIP dependencies are installed.

## Dataset layout

The code expects DF40 / DeepfakeBench-style JSON indexes under the configured dataset root. DFD is loaded from:

```text
/Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json
```

with frame paths under the FaceForensics++-style DFD folders. To build it:

```bash
python experiments/2026-05-05_paper_feature_geometry/scripts/prepare_dfd_df40.py \
  --root /Dataset/deepfake_detection \
  --output-json /Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json \
  --frames-per-video 32
```

## Common commands

```bash
P=experiments/2026-05-05_paper_feature_geometry

# train
torchrun --nproc_per_node=8 $P/scripts/train.py --config configs/clip_base.yaml

# standard evaluation
python $P/scripts/evaluate.py --config configs/clip_base.yaml \
  --checkpoint $P/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth \
  --domain all

# raw / direction / length evaluation
python $P/scripts/evaluate_norm_correction.py --config configs/clip_base.yaml \
  --checkpoint <ckpt> --splits ffpp,cdf,dfd --mode raw,feature_norm,feature_length

# tests (pytest config already sets pythonpath=src and collects phase tests)
python -m pytest -q
```

New training runs default to `experiments/_runs/<timestamp>/`; promote a finished run into a dated phase directory when it becomes part of a study.

## Artifact policy

The repository tracks code, configs, reports, docs and READMEs. These stay local and are git-ignored:

```text
experiments/*/results/     # checkpoints, tables, parquet caches, logs
experiments/*/figures/
experiments/_runs/
tests/
features/  logs/
*.pth *.pt *.ckpt *.onnx *.dat *.parquet *.npz
```

Checkpoints (~6.6 GB), the frame-level parquet (~1.4 GB) and the generated tables/figures are never pushed; each phase README says how to regenerate them.

## Metrics

AUC is the main ranking metric and is threshold-free; EER also scans thresholds. ACC, balanced ACC, real recall and fake recall depend on the chosen threshold.

DFD is highly imbalanced (fakes roughly 10× reals), so ordinary accuracy is misleading: an almost-always-fake threshold reaches high ACC with real recall near zero. Use AUC and balanced accuracy for DFD analysis.

Video-level scores must be temperature-calibrated (or aggregated in logit space) before comparing raw against direction — see the main result above.
