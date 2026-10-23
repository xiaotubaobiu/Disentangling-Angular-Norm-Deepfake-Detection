# DF40 v1.1 Implementation Plan

> **For agentic workers:** use superpowers:executing-plans for implementation; preserve the ledger and do not restart completed tasks.

**Goal:** reproducible 31-method, three-backbone/five-seed scoring and source-threshold study, isolated to physical GPU4–7.

**Architecture:** an independent Python package with a single launcher, immutable source data, hash-bound stage artifacts, and a resumable queue. Data, training, scoring and analysis each have separate stage markers.

**Tech Stack:** Python3.11, PyTorch2.6, torchvision0.21, numpy, pandas/pyarrow, sklearn, Pillow, Albumentations2.0.8, pytest.

**Spec:** ../specs/2026-10-07-df40-scoring-transfer-design.md and the unchanged v1.1 snapshot.

## Global Constraints

- 31 methods:9FS+12FR+10EFS; e4s included, e4e/FE excluded. No new-method updating/replay.
- Three backbones × seeds7/42/123/2026/3407;15 epochs and final epoch;30 frozen matched heads.
- 224², AdamW WD5e-4, effective batch128; CLIP/DINO/EfficientNet learning rates1e-5/5e-5/1e-4.
- Shared fixed eight-frame video pools, independent EFS images; no cross-method merging or real duplication.
- No CDF model/flow/threshold/scale/data-policy selection. Calibration unavailable unless independent source pool established.
- Preserve released/physical/source/experiment roles; source changes require explicit scientific authorization.
- Only physical4–7; stop/restore only original running onecat-vllm, leave gateway and Jev untouched.

## Review Focus

- Same video ID across methods must remain separate throughout sampling, caches and bootstrap.
- Source calibration cannot silently share original sources or bytes with training/testing.
- EFS bucket IDs cannot be treated as natural videos or certified generation ancestry.
- Checkpoint interruption cannot silently change samples, RNG, effective batch or final-epoch choice.
- Failed or missing inputs/results cannot be silently dropped or replaced by historic results.

## Tasks

### 1. Storage, source audit and policy gate
Files: configs/v1.1.json, src/deepfake_detector/manifest.py, tests/test_manifest.py, reports/storage.json.
Interfaces: audit(config)->audit.parquet+summary.json; freeze(config,policy)->manifests+complete.json.
- [ ] Test role-aware EFS paths, canonical real keys, numeric8 ordering, same-ID separation, missing paths and content conflicts; observe RED.
- [ ] Implement bounded-worker full byte/RGB/decode audit into resumable SQLite, preserving all source rows and reasons.
- [ ] Quantify conflicts and independent-source calibration candidates; request a scientific ruling only where the design does not decide policy.
- [ ] Freeze one identical manifest for all15runs only after source/role/31-method gates pass; record calibration availability explicitly.
- [ ] Verify GREEN and commit scoped files.

### 2. Scoring, operating points and analysis correctness
Files: src/deepfake_detector/scoring.py, analysis.py, tests/test_scoring.py.
Interfaces: score/features->four video/two image flows; threshold(source_real_scores,weights)->FPR5; metrics+pair transitions+paired bootstrap->tables.
- [ ] Test nonlinear pooling, monotone/affine negative controls, inclusive ties, macro/shared-real weighting, pair-ledger AUC identity, EFS units; observe RED.
- [ ] Implement FP64 statistics, source-only scale/length controls and independently fitted thresholds.
- [ ] Implement regularized zero-initialized raw/direction matched heads and separate calibration/test resamples.
- [ ] Verify GREEN with exact synthetic cases, record failures/invalid resamples and commit.

### 3. Training, feature cache and resume
Files: models.py, training.py, tests/test_training.py, requirements.lock.txt.
Interfaces: train(config,run_id,device)->final15.pt+state; extract(final,manifest)->aligned chunkedfeatures.
- [ ] Test epoch-boundary resume equality, config/hash mismatch rejection, balanced sampling and frozen feature extraction; observe RED.
- [ ] Pin reference revisions/weights/environment; implement full raw visual finetuning with the prescribed augmentation, effectivebatch and checkpoint semantics.
- [ ] Verify all15 resolved configs and CPU tiny-model train/resume, signed-logit/cache equivalence; commit.

### 4. CLI, queue and GPU lifecycle
Files: cli.py, orchestration.py, run.py, README.md, tests/test_orchestration.py.
Interfaces: one launcher commands audit/freeze/preflight/probe/run/features/analyze/report/status/resume.
- [ ] Test lock contention, UUID binding, no0–3, input-hash stage mismatch, originally-stopped LLM handling and queue failures; observe RED.
- [ ] Record nonsecret original container state; prepare models and CPU gates before stopping service.
- [ ] Run GPU probes on4–7 only, then one end-to-end pilot before the immutable15run queue.
- [ ] Preserve all statuses, checkpoints and recovery instructions; restore original LLM after required GPU work or terminal stop.
- [ ] Verify GREEN and commit.

### 5. Final evidence and manual reproduction
Files: reports/*, README.md, Obsidian appended experiment record.
- [ ] Whole-package independent review; fix material findings with RED→GREEN evidence.
- [ ] Recompute all summary tables from saved sample scores, show all five seeds separately by backbone.
- [ ] Report original-data immutability, scope/units/conditional uncertainty, failures, negative results, service restoration and exact reproducible commands.

Execution already authorized by the user's explicit “开始实验” instruction; implementation-plan handoff is not a second permission gate. GPU/data-policy gates remain evidence requirements rather than generic authorization prompts.
