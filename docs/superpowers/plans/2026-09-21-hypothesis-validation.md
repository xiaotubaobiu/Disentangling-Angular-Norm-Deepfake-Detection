# Deepfake Feature-Geometry Hypothesis Validation Plan

> **Superseded:** This plan came from an earlier misinterpretation of the user
> request and is not the executed Visual-Jev experiment. The authoritative plan
> is `docs/superpowers/plans/2026-09-21-visual-jev-h1-implementation.md`, and the
> completed result is
> `experiments/2026-09-21_visual_jev_h1/reports/h1_validation_result.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, from the retained model-A checkpoints and the datasets actually present on this machine, whether the proposed cross-domain feature-direction / radial-angular hypothesis is supported, refuted, or still underdetermined.

**Architecture:** Treat the 15 raw-trained checkpoints (3 backbones × 5 seeds) as model A unless a separate checkpoint is identified. Recompute the decisive metrics from retained frame/video outputs, using source-only fitting on FF++ validation and CDF only for final target evaluation; separate rank transfer, calibration, and fixed-threshold transfer. NanoJev is used as a validation-discipline reference (held-out splits, full probability metrics, baseline separation, and reproducible artifacts), not as a transferable model architecture.

**Tech Stack:** Python 3.11, PyTorch checkpoint metadata, pandas, DuckDB, scikit-learn, SciPy, pytest, Markdown.

**Spec:** `experiments/2026-05-05_paper_feature_geometry/docs/2026-06-03-midt-paper-design.md`, constrained by the later falsification protocols in `experiments/2026-07-27_pooling_saturation/docs/2026-07-27-pooling-saturation-design.md` and `experiments/2026-07-30_threshold_transfer/docs/DESIGN.md`.

## Global Constraints

- Use only datasets and cached inference outputs present under `/Dataset/deepfake_detection` and this repository.
- Do not retrain model A; the user states it is already trained.
- Never fit calibration, thresholds, aggregation choices, or model selection on CDF target labels.
- Aggregate videos by `(method_id, video_id)`, never by `video_id` alone.
- Compare raw and direction on identical videos with paired resampling.
- Report AUC separately from fixed-threshold balanced accuracy/FPR.
- Use only GPUs 0–3 for fresh backbone re-inference; GPUs 4–7 are occupied and out of scope.
- Preserve all existing dirty-worktree changes and write new evidence only to `/tmp` plus the final validation report.

## Review Focus

- A repeated `video_id` across forgery methods must remain separate through aggregation; the repository regression tests must catch accidental merging.
- FF++ test rows must not be used to fit the source calibrator or threshold; only FF++ validation cache is valid for fitting.
- Raw/direction comparisons must share frame/video identities and bootstrap draws; unpaired subsampling invalidates the delta.
- Probability pooling saturation must not be mistaken for feature-geometry gain; verify with mean-logit and calibrated counterfactuals.
- A specific trained adaptive-gamma model cannot be claimed validated unless its checkpoint and predictions are present; absence must be explicit.

---

### Task 1: Evidence and provenance audit

**Files:**
- Read: `experiments/2026-05-05_paper_feature_geometry/results/checkpoints/**/output/best_model.pth`
- Read: `experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet`
- Read: `experiments/2026-07-27_pooling_saturation/results/val_cache/*.parquet`
- Read: `/Dataset/deepfake_detection/**`

**Interfaces:**
- Consumes: checkpoint directories, saved configs, source-validation caches, CDF frame scores.
- Produces: counts for checkpoints, domains, methods, frames, videos, and an explicit mapping of “model A” to retained artifacts.

- [x] **Step 1: Confirm checkpoint coverage**

Run:

```bash
find experiments/2026-05-05_paper_feature_geometry/results/checkpoints -name best_model.pth | wc -l
find experiments/2026-05-05_paper_feature_geometry/results/checkpoints -name meta.json | wc -l
```

Expected: `30` checkpoints and `30` metadata files; the hypothesis comparison uses the 15 raw-trained checkpoints.

- [x] **Step 2: Run the source/target input audit**

Run:

```bash
python experiments/2026-07-30_threshold_transfer/scripts/00_audit_inputs.py
```

Expected: `34/34 inputs present`, 15 FF++ validation caches, all 15 raw checkpoints represented on CDF, DFD explicitly excluded.

- [x] **Step 3: Query authoritative parquet coverage**

Run DuckDB against `per_frame_scores.parquet` to verify 30 checkpoints, three domains, 39,830,640 rows, and the expected raw/direction/length versus normtrain mode layout.

- [x] **Step 4: Check whether an adaptive-gamma checkpoint exists**

Run:

```bash
rg -n -i "adaptive.?gamma|gamma.?head|radial.?angular|径向.?角向" src experiments --glob '*.py' --glob '*.yaml'
find . -type f \( -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \)
```

Expected: document either the exact checkpoint/prediction path or the limitation that only raw/normtrain checkpoints are present.

### Task 2: Correctness gates

**Files:**
- Test: `tests/`
- Test: `experiments/2026-07-23_four_questions/tests/`
- Test: `experiments/2026-07-27_pooling_saturation/tests/`
- Test: `experiments/2026-07-30_threshold_transfer/tests/`

**Interfaces:**
- Consumes: current repository code and retained test fixtures.
- Produces: a pass/fail gate for aggregation keys, paired sampling, calibration, leakage, and bootstrap behavior.

- [ ] **Step 1: Run all collected tests**

Run:

```bash
PYTHONPATH=src python -m pytest -q
```

Expected: all 99 collected tests pass.

- [ ] **Step 2: Run threshold-transfer leakage tests verbosely**

Run:

```bash
PYTHONPATH=src python -m pytest experiments/2026-07-30_threshold_transfer/tests/test_no_leakage.py -vv
```

Expected: source-only calibration and threshold selection tests pass.

### Task 3: Independent result replay

**Files:**
- Read: `experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_video_scores.parquet`
- Read: `experiments/2026-07-27_pooling_saturation/results/tables/*.csv`
- Read: `experiments/2026-07-30_threshold_transfer/results/tables/*.csv`
- Create temporarily: `/tmp/deepfake-hypothesis-validation/`

**Interfaces:**
- Consumes: immutable saved score tables.
- Produces: independently recomputed native CDF deltas, corrected per-method deltas, calibration counterfactuals, transfer regret, and paired confidence intervals.

- [ ] **Step 1: Replay native CDF direction-minus-raw AUC**

Run:

```bash
python experiments/2026-07-21_frame_midterm_diagnosis/scripts/cdf_native_dir_raw.py \
  experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914 \
  --out-dir /tmp/deepfake-hypothesis-validation/native
```

Expected: 15 pooled cells and 240 per-method cells; semantically correct grouping yields pooled mean direction−raw near `-0.0044` with direction winning `4/15`, not the original `+0.0142` and `15/15` produced by the invalid grouping.

- [ ] **Step 2: Recompute headline mechanisms from tables**

Compute directly from the saved CSVs:

```text
pooling artefact: mean Delta_temp where Delta_prob > 0
method dependence: I_pool(FSGAN) - I_pool(SadTalker)
threshold transfer: mean/median/range R_transfer at t_BA
deployment failure: source and target BA/FPR plus 95% paired bootstrap CI
```

Expected: calibrated direction advantage approximately zero; opposite FSGAN/SadTalker signs; `R_transfer≈0.198`; paired raw/direction difference CI contains zero.

- [ ] **Step 3: Verify report-table identity**

Compare fresh native CSVs with retained CSVs using exact keys and `max(abs(numeric_delta)) == 0`. Check that every headline value in the final report is traceable to one table row/group operation.

### Task 4: Representative fresh backbone replay

**Files:**
- Read: `experiments/2026-05-05_paper_feature_geometry/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth`
- Read: `/Dataset/deepfake_detection/DF40_all/**`
- Create temporarily: `/tmp/deepfake-hypothesis-validation/fresh-inference/`

**Interfaces:**
- Consumes: one retained raw-trained checkpoint, its exact saved config, FF++/CDF images, and GPUs 0–3.
- Produces: a fresh image→backbone→logit smoke result that can be compared with the retained frame-score artifact.

- [ ] **Step 1: Verify CUDA visibility outside the restricted sandbox**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: GPUs 0–3 are V100 32GB and available; do not launch on GPUs 4–7.

- [ ] **Step 2: Re-run one representative checkpoint**

Use the phase-2 extraction entrypoint with `clip_raw_s42`, restricting the smoke replay to FF++ and CDF and writing only under `/tmp/deepfake-hypothesis-validation/fresh-inference/`.

Expected: checkpoint/config load successfully, real images are read from the configured local dataset, and raw/direction logits are produced without touching retained outputs.

- [ ] **Step 3: Compare fresh and retained scores**

Join the fresh rows to the retained frame parquet by checkpoint/domain/image path/inference mode. Report row coverage, maximum absolute signed-logit difference, and AUC difference. If the extraction entrypoint cannot safely bound the sample count, run a deterministic loader subset through the same model API and label the result a smoke replay rather than a full-domain rerun.

### Task 5: Verdict and reproducibility report

**Files:**
- Create: `experiments/2026-09-21_hypothesis_validation/reports/hypothesis_validation_report.md`

**Interfaces:**
- Consumes: Tasks 1–3 evidence plus NanoJev’s public validation methodology.
- Produces: one self-contained Chinese report with hypotheses, gates, results, limitations, commands, and verdicts.

- [ ] **Step 1: Apply explicit verdict gates**

Use these gates:

```text
Strong direction hypothesis = supported only if corrected direction−raw delta is
positive across backbones/seeds, survives mean-logit/calibration counterfactuals,
and its paired uncertainty excludes zero.

Zero-calibration threshold hypothesis = supported only if all checkpoint×mode
cells have R_transfer <= 0.05 and source FPR budgets remain within 0.05 absolute
error on CDF.

Adaptive radial-angular mechanism = supported only as a training proposal if its
gain is selected on source validation and confirmed on untouched target data;
target-oracle gamma scans are mechanism diagnostics, not deployable evidence.
```

- [ ] **Step 2: Write separate verdicts**

Report independently:

```text
H1 direction is the primary transferable carrier
H2 test-time normalization is a cross-domain free lunch
H3 threshold near 0.5 transfers without target calibration
H4 one shared fake direction supports global multi-method alignment
H5 radial information can still be useful conditionally
H6 the trained adaptive model A itself beats frozen baselines
```

H6 must be `证据不足` unless a distinct adaptive checkpoint and untouched-test predictions are identified.

- [ ] **Step 3: Record the GPU replay boundary**

State exactly which checkpoint/domains/sample count were freshly replayed on GPUs 0–3. Keep all broad 15-checkpoint conclusions grounded in the retained full frame logits/features unless every checkpoint was rerun.

- [ ] **Step 4: Validate the report**

Run:

```bash
rg -n "成立|不成立|部分成立|证据不足|复现命令|局限" \
  experiments/2026-09-21_hypothesis_validation/reports/hypothesis_validation_report.md
```

Expected: every hypothesis has one verdict and at least one directly traceable numeric artifact.
