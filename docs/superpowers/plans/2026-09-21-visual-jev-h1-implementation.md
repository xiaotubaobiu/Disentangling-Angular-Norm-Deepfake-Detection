# Visual-Jev H1 Implementation Plan

> Execute in the existing checkout because the required checkpoints and dataset
> are large local resources that are not available in a clean worktree.

**Goal:** Train and evaluate the research-plan-defined Visual-Jev model B against
the selected CLIP binary model A and produce a falsifiable H1 verdict.

**Architecture:** Reuse and freeze A's fine-tuned CLIP ViT-B/16 visual tower;
precompute the two fixed candidate embeddings with the frozen CLIP text tower;
train one shared `4d -> 512 -> 128 -> 1` candidate scorer with cross entropy.

**Stack:** Python, PyTorch, OpenAI CLIP, pytest, scikit-learn, four Tesla V100 GPUs.

---

### Task 1: Freeze the protocol and baseline provenance

**Files:**
- Create: `docs/superpowers/specs/2026-09-21-visual-jev-h1-design.md`
- Create: `experiments/2026-09-21_visual_jev_h1/README.md`
- Create: `experiments/2026-09-21_visual_jev_h1/configs/visual_jev_s42.yaml`
- Create replication configs for seeds 7 and 2048

Record the A checkpoint, its validation-only selection rationale, the fixed
prompts, optimization settings, domains, metrics, and one-point decision gate.

### Task 2: Test-drive the Visual-Jev model

**Files:**
- Create: `tests/models/test_clip_visual_jev.py`
- Create: `src/models/clip_visual_jev.py`
- Modify: `src/models/factory.py`

Write failing tests for shared scoring, frozen encoders/buffers, output shape,
candidate permutation equivariance, and target/loss permutation invariance.
Then implement the smallest model that passes them. Add a factory construction
test and verify that A's visual weights load while its linear head is ignored.

### Task 3: Make evaluation model-agnostic and video-correct

**Files:**
- Modify: `src/data/datasets.py`
- Modify: `src/evaluation/sample_eval.py`
- Modify: `src/engine/trainers.py`
- Create or extend tests under `tests/`

Add `method` and semantic `video_key` to frame batches and sample rows. Aggregate
by `video_key`, not bare `video_id`. Evaluate logits through a common
`classify_features` interface so A and B use the same row and metric pipeline.
Add frame/video AUROC, AP, Brier, and reference accuracy without changing the
validation-AUROC checkpoint criterion.

### Task 4: Build a dedicated reproducible runner

**Files:**
- Create: `experiments/2026-09-21_visual_jev_h1/scripts/evaluate_models.py`
- Create: `experiments/2026-09-21_visual_jev_h1/scripts/compare_models.py`
- Modify: `experiments/2026-05-05_paper_feature_geometry/scripts/train.py` only
  when necessary for frozen-parameter optimization or generic metrics

The evaluator must save per-frame rows, per-video rows, metrics, parameter
counts, prompt metadata, and checkpoint provenance. The comparer must join A and
B by semantic video key and compute deterministic paired bootstrap intervals.

### Task 5: Smoke test on GPU 0

Run a bounded one-batch forward/backward test and a short subset training run.
Verify finite loss, scorer gradients, no visual gradients, exact candidate-order
behavior, checkpoint save/load, and metric output. Fix failures using systematic
debugging before launching full runs.

### Task 6: Train B on GPUs 0-2

Launch seed 42, 7, and 2048 as independent single-GPU processes pinned to
physical GPUs 0, 1, and 2. Keep GPU 3 available for A evaluation. Monitor logs,
GPU utilization, validation convergence, and early stopping. Select each run's
checkpoint only by validation video AUROC.

### Task 7: Evaluate A and B without target tuning

Evaluate selected A and each B checkpoint on FF++ test and Celeb-DF-v2 using the
same corrected pipeline. Do not inspect target results until checkpoint
selection is complete. Run the order-swap check against the saved B checkpoint.

### Task 8: Compare and issue the verdict

Produce a machine-readable comparison JSON/CSV and a concise report containing:
baseline selection evidence, convergence, metrics, parameter counts, paired
bootstrap intervals, the pre-registered one-point gate, limitations, and one of
`H1 supported`, `H1 not supported`, or `evidence insufficient`.

### Task 9: Verification before completion

Run focused model/evaluation tests, the relevant full test suite, configuration
validation, artifact existence checks, and a clean reproduction of the final
comparison command. Only then mark the goal complete.
