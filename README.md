# Deepfake Detection Feature Geometry Experiments

This repository contains the code for studying whether deepfake detection transfers across datasets through feature direction, feature length, or both.

## Main result

The paper compares three evaluation modes on the same raw checkpoints:

- `raw`: classifier uses both feature direction and feature length.
- `feature_norm`: features are L2-normalized before classification, so the classifier mainly uses direction.
- `feature_length`: the feature norm is used as the score, isolating length alone.

Across FF++, CDF, and DFD, the observed pattern is:

- FF++: raw and direction-only are both near-perfect.
- CDF: direction-only > raw > length-only.
- DFD: raw > direction-only >> length-only.

## Repository layout

```text
train.py                              # training entrypoint
evaluate.py                           # standard FF++ / CDF / DFD evaluation entrypoint
scripts/evaluate_norm_correction.py   # paper raw / direction / length evaluation
scripts/prepare_dfd_df40.py           # prepare DFD in DF40-style JSON/frame layout
scripts/summarize_paper_results.py    # print paper AUC tables
src/deepfake_detection/               # package code
configs/                              # reusable training config templates
outputs/                              # local run outputs, ignored by git
```

## Installation

```bash
conda env create -f environment.yml
conda activate deepfake-detection
pip install -r requirements.txt
```

If you use an existing environment, make sure PyTorch, torchvision, OpenCV, scikit-learn, pandas, PyYAML, and CLIP dependencies are installed.

## Dataset layout

The code expects DF40/DeepfakeBench-style JSON indexes under the configured dataset root. DFD is loaded from:

```text
/Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json
```

with frame paths under the FaceForensics++-style DFD folders.

## Prepare DFD

```bash
PYTHONPATH=src python scripts/prepare_dfd_df40.py \
  --root /Dataset/deepfake_detection \
  --output-json /Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json \
  --frames-per-video 32
```

## Train

```bash
PYTHONPATH=src python train.py --config configs/clip_base.yaml
```

Use `configs/clip_base.yaml`, `configs/dinov2_base.yaml`, or `configs/efficientnet_base.yaml` as starting points. Local training outputs are written under `outputs/` and are ignored by git.

## Standard evaluation

```bash
PYTHONPATH=src python evaluate.py \
  --config configs/clip_base.yaml \
  --checkpoint outputs/<run>/best_model.pth \
  --domain all
```

The retained artifact folders usually store the original checkpoint path in `output/checkpoint_path.txt` instead of copying large checkpoint files into the repository.

## Paper feature-geometry evaluation

```bash
PYTHONPATH=src python scripts/evaluate_norm_correction.py \
  --config configs/clip_base.yaml \
  --checkpoint outputs/<run>/best_model.pth \
  --splits ffpp,cdf,dfd \
  --mode raw,feature_norm,feature_length \
  --output-dir outputs/<run>
```

## Summarize paper results

```bash
PYTHONPATH=src python scripts/summarize_paper_results.py
```

This prints the AUC table used in the paper.

## Artifact policy

This GitHub repository is code-only. The following are intentionally ignored and should stay local:

```text
outputs/
experiments/
figures/
features/
tests/
```

Large checkpoints, generated experiment CSVs, figures, frame-level feature caches, and local tests are not pushed.

## Metrics

AUC is the main ranking metric and is independent of any fixed threshold. EER also scans thresholds. ACC, balanced ACC, real recall, and fake recall depend on the chosen threshold.

DFD is highly imbalanced, with fake samples roughly ten times more frequent than real samples. Ordinary accuracy can therefore be misleading: an almost-always-fake threshold can reach high ordinary ACC while giving real recall near zero. Use AUC and balanced accuracy for DFD analysis.

Use AUC and balanced accuracy for DFD analysis.
