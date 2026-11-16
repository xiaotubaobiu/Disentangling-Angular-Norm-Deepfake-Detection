# Pipeline Effects in Cross-Domain Face Deepfake Detection

Public code repository for the manuscript **Pipeline Effects in Cross-Domain Face Deepfake Detection**, prepared for IEEE Access.

This work evaluates how preprocessing history, frame-to-video scoring, and threshold transfer affect cross-domain face deepfake detection. It studies CLIP ViT-B/16, DINOv2 ViT-B/14, and EfficientNet-B0 using paired comparisons; it does not introduce a new detector architecture.

Repository: [xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection).

## Experiments

- **Preprocessing history (H):** compare direct dlib processing (B, 224 × 224) with RetinaFace processing (A, 336 × 336) followed by a new dlib detection/alignment pass (A→B, 224 × 224). Train on FF++ c23 and evaluate on FF++ and DFD with all four train/test pipeline combinations. Report video coverage separately from AUC on eligible videos. Three backbones, three seeds, and two training pipelines give 18 models.
- **Feature scoring and video aggregation (D):** evaluate the full 2 × 2 × 2 combination of Raw/L2 features, mean-logit/mean-probability pooling, and correct/erroneous video grouping. Each natural video contributes eight frames. Correct grouping retains the manipulation method in the video identity; erroneous grouping pools the eight frames from each of the methods sharing an identifier. Three backbones and five seeds give 15 models.
- **Threshold transfer:** compare default thresholds, source-real calibration, and target-real calibration using 10/20/40 real videos. Report AUC, balanced accuracy, FPR, and TPR on a fixed decision-test cohort.

D evaluates the same 21 DF40 manipulation methods (9 face swapping and 12 face reenactment methods) across FF++ and Celeb-DF source content. These target samples are the DF40-CDF collection, rather than the original Celeb-DF fake-video collection.

## Branches and source layout

The `main` branch contains the journal implementation. The separate [`conference-original`](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection/tree/conference-original) branch preserves the conference implementation, including the historical method-blind video-grouping rule analyzed and corrected in the journal work.

```text
src/journal_detector/      Training, data preparation, inference, and evaluation
configs/                   Model settings, dataset protocols, and run snapshots
scripts/                   Development, execution, validation, and analysis tools
tests/                     Implementation tests
third_party/               Pinned upstream implementations and their licenses
legacy/31_methods/         Historical 31-method implementation
legacy/21_methods/         Historical 21-method implementation
docs/                      Usage instructions and execution history
run.py                     Command-line entry point
requirements.lock.txt      Recorded experiment environment
```

## Training protocol

| Setting | CLIP | DINOv2 | EfficientNet-B0 |
|---|---|---|---|
| Initialization | OpenAI ViT-B/16 | Official ViT-B/14 | ImageNet B0 |
| Feature dimension | 512 | 768 | 1280 |
| Learning rate | 1e-5 | 5e-5 | 1e-4 |
| D maximum epochs | 2 | 30 | 2 |
| Final H maximum epochs | 10 | 10 | 10 |

All models fine-tune the full visual encoder and binary head without training-time output L2 normalization. Shared settings are AdamW, cross-entropy loss, weight decay 5e-4, batch size 128 per GPU, AMP, gradient-norm clipping at 1.0, epoch-based warmup/cosine scheduling, and EMA decay 0.999. DF40 image augmentations precede the normalization required by each pretrained backbone.

D uses development seeds 1007 and 1042 to select the epoch budgets with source validation and an isolated CDF development set. Formal seeds are 7, 42, 123, 2026, and 3407. Each formal run selects its best EMA checkpoint by correctly grouped source-validation mean-probability AUC within the frozen budget. Patience is five epochs.

H uses seeds 7, 42, and 123. Within each seed and backbone, B/AB models share initialization and sampling/augmentation random keys. A common checkpoint epoch maximizes their average AUC on the same source-B validation cohort. The final paired runs select epoch 10. DINOv2 and EfficientNet-B0 were rerun with ten-epoch schedules after inspection of longer exploratory runs; this is a matched-budget sensitivity experiment. DINOv2 source-validation performance was still improving at the cap. The original H-CLIP runs are retained. One D epoch has 799 mini-batches; one H epoch has 185.

The [initial run snapshots](configs/experiments/journal_20261102/README.md) document the original execution protocol. The [H revision configurations](configs/experiments/access_20261103/README.md) specify the twelve replacement DINOv2/EfficientNet-B0 runs. Historical configurations remain available for provenance; use the final revision when reconstructing manuscript H results.

## Data and evaluation cohorts

Obtain FF++, DFD, DF40, and pretrained model weights through their official distribution channels and observe their access and licensing requirements. Images, videos, trained checkpoints, prediction caches, and manuscript sources are not distributed in this code repository. Split definitions are stored in `configs/data_protocol/`.

D training samples real inputs with total probability 0.5 and each fake method with probability 0.5/21, with replacement. Evaluation uses eight fixed, distinct RGB frames per natural video.

The isolated CDF development cohort contains 20 real and 420 fake videos; related videos are excluded from formal evaluation and calibration. The formal ranking cohort contains 157 real and 12,037 fake videos. Threshold comparisons reserve 64 real videos for calibration and use the same 93-real/12,037-fake decision-test cohort for every threshold policy.

H uses the same raw frames for B and A→B. Failed preprocessing does not trigger replacement-frame sampling. All-B and common-success cohorts are reported separately, so coverage changes are not confused with paired performance changes.

## Usage and reproducibility

Use an isolated Python 3.11 environment and consult `requirements.lock.txt` for the recorded package versions. CUDA/PyTorch installation must match the execution environment; the recorded runs used PyTorch 2.6.0 with CUDA 12.4.

```bash
git clone https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection.git
cd Disentangling-Angular-Norm-Deepfake-Detection
python run.py --help
python run.py status
python -m pytest tests -q
```

The commands above inspect the interface/status and run implementation tests. Dataset preparation and experimental runs require locally configured data, pretrained weights, and output paths.

- [Detailed usage and data preparation (Chinese)](docs/期刊实验代码说明.md): module roles, split construction, and execution commands; historical sections describe earlier budgets.
- [Runtime validation](docs/execution/runtime-validation.md): GPU/environment qualification and restart checks.
- [Retained analysis source snapshots](scripts/reproducibility/journal_20261102/README.md): provenance and checksums for auxiliary analyses. Some scripts require retained per-frame predictions and experiment artifacts.

The accompanying manuscript supplement contains seed-level tables, numerical summaries, and reproduction scripts. Full prediction/checkpoint artifacts are not bundled here. Historical receipts may record the original machine's absolute paths; configure paths for your own environment rather than treating those paths as dependencies. Optional local `data`, `paper`, and `local/` entries are ignored by Git; the manuscript/Overleaf repository is maintained separately.

## Authors

Shiyu Chen, Nomundari Batbayar, Munkhdelgerekh Batzorig, Tserenochir Batkhuyag, Esam Ali Mohammed ALBHELI, and Kangbin Yim.

Munkhdelgerekh Batzorig and Kangbin Yim are with the Department of Information Security Engineering, Soonchunhyang University. The other authors are with the Department of Software Convergence at the same university.

Corresponding author: Kangbin Yim — yim@sch.ac.kr.

Third-party code retains the license and attribution supplied by its respective project.
