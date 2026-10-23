# H data acceptance, source qualification and training queue

This implementation prepares the H branch while the existing D GPU queue and
CPU H pixel preprocessing continue. Passing CPU tests does not accept production
H data, qualify a GPU model, or provide a scientific result.

## Required order

1. Finish all 8,431 pixel candidates, including 560 `SOURCE_UNUSED` videos.
2. In a fresh process run the repaired `geometry_h audit-existing` command. The
   already-running pixel process loaded the older finalizer and cannot supply
   final acceptance by itself.
3. Run `data_h.inspect_h_data(root)`, review the complete graph, quarantines and
   reconstructed 13 tables independently, then call
   `data_h.freeze_h_data(root, independent_review_path)`. The review schema and
   exact hash bindings are documented in `data_h.py`. A synthetic review is
   never a production review. Every freeze creation performs the deep scan.
4. Run the H queue only after that freeze. H qualification and training dispatch
   wait until the D supervisor reports
   `P3_D_CACHE_COMPLETE_ANALYSIS_PENDING` with no active jobs. Fresh physical
   GPU0–3 UUID, compute-process, memory and shared lifetime-lock checks apply.
5. Complete real source qualification for all three backbones in AMP and FP32.
   The queue prepares matched B/AB source probes and schedules the six jobs;
   each job tests both histories with the actual trainer/loader, continuous vs
   resumed states, gradients, features and independently recomputed source
   scores. Only a documented CUDA OOM permits 32→16→8 microbatch fallback.
6. Accept the source qualification gate, freeze the dedicated H execution
   header, run development, freeze source-only shared recipes, then run formal
   training and all-five-seed paired update audits.

## Commands

Run from the v2 code directory. `R` below means the artifact root; use the
existing `R/environment/bin/python` and bounded CPU thread settings.

```text
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 R/environment/bin/python -m reliability_v2.queue_h status --root R

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 R/environment/bin/python -m reliability_v2.geometry_h audit-existing --manifest-paths R/data/H/H_processed_manifest_paths.json --artifacts R/data/H

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 R/environment/bin/python -m reliability_v2.queue_h run --root R --code /absolute/path/to/v2
```

The `status` action is read-only and launches nothing. The `run` action is an
explicit future launch; it is not a background automation and was not executed
during implementation acceptance. Do not start duplicate supervisors. The CLI
holds `H_pipeline.lock`; direct workers also validate provenance, D priority and
physical devices before allocations, and hold a wrapper lock plus the unchanged
trainer's run/GPU locks. Existing owned workers block a replacement supervisor.
If a supervisor fails with active workers, preserve them until they finish and
inspect their evidence before starting a replacement.

## Frozen comparisons

- Standard development: 20 models (CLIP 4, DINOv2 12, EfficientNet 4), using two
  development seeds × B/AB. DINO's three learning rates are source-only. The
  shared B/AB recipe uses all four last-five-epoch source macro-AUC histories.
- DINO extension: at most four fresh 45-epoch and four fresh 60-epoch models,
  using the selected learning rate when the registered adjacent-window rule
  triggers. No warm continuation replaces a fresh experiment.
- Formal: 30 AMP models, three backbones × B/AB × five seeds, using frozen last
  checkpoints. Source-best checkpoints are retained for sensitivity analysis.
- Each B/AB pair uses identical logical mothers, sampling masses, effective
  batch 128, draw indices, keyed augmentations and update-attempt budgets. Draw
  count is rounded from common logical rows and does not asymmetrically merge
  B/AB RGB files.
- Any unequal successful-update positions within a backbone's five AMP pairs
  require all ten formal models of that backbone to run in uniform FP32.
  Existing AMP artifacts remain; selective seed replacement is prohibited.
  FP32 skips or nonfinite gradients stop for source protocol review.
- A backbone's target-export gate requires all five effective pairs to pass.
  No target performance selects recipes or fallback precision.

## Recovery, provenance and capacity

H uses `configs/H_execution.json`, `H_queue_status.json` and
`H_run_registry.json`; it does not edit the live D configuration or gate.
The H bundle hashes `data_h`, `qualification_h`, `queue_h`, `worker_h` and the
unchanged shared `queue_d` helpers. The frozen trainer/model hash is separate.

Workers verify train/DEV input bytes before the first or resumed invocation.
Matching `H_training_start.json` must predate all training artifacts; a missing
start over an existing partial/completed run fails closed. A terminal seal
binds the fixed inventory of checkpoints, full history, identity/config,
source scores, draw/method/augmentation arrays and all update logs. Recovery
after a terminal-save/worker crash requires matching start evidence and
authoritative checkpoint history. Existing seals are verified, never replaced
to endorse changed files. Source recipe selection reads sealed histories.

Initial dispatch requires more than 200 GiB estimated workset plus 50 GiB
reserve. Subsequent dispatch preserves 50 GiB; conditional FP32 reruns reserve
an additional 1.2× the retained ten AMP-run bytes. These are operational budget
estimates, not guaranteed bounds. No data is deleted or silently migrated.

Production freeze and qualification acceptance perform full validation.
Routine qualification validation freshly hashes proof files up to 8 MiB and
checks all accepted file sizes/mtime-ns; large checkpoint freshness is cached
metadata validation, not a new full cryptographic proof. Use `--deep` for full
checkpoint/proof verification. H freeze runtime validation similarly verifies
accepted metadata/hash bindings; `deep=True` rechecks all raw/image artifacts.

The queue ends at `H_FORMAL_COMPLETE_TARGET_CACHE_ANALYSIS_PENDING`. H target
feature-cache export, common 2×2 and all-B deployment analyses, five-seed paper
tables/figures and final bilingual manuscript integration remain later stages.
