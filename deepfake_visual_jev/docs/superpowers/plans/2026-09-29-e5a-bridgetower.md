# E5-A BridgeTower Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether BridgeTower native ITM shared candidate scoring passes the frozen FF++ semantic qualification gates, then stop with PASS, FAIL, or INCONCLUSIVE.

**Architecture:** Use the pinned Hugging Face `BridgeTower/bridgetower-base-itm-mlm` checkpoint. Freeze its `vision_model` and `text_model`; train only cross-modal/bridge modules and the native `itm_score`. Score each complete question-and-candidate sentence through the same model; use its ITM match logit and candidate cross entropy. Reuse E4's frozen FF++ physical record manifests and E2/E3 language bank; do not use CDF.

**Tech Stack:** Python, PyTorch, Transformers, existing Visual-Jev data and evaluation modules.

## Task 1: Lock provenance and data

- [x] Pin checkpoint revision `6cfb318bb08588808368a9ab1fe26609bc148f1f` and download original files.
- [x] Audit E4 frozen physical train/val/FF++ manifests; verify physical train/val/test disjointness and FF++ semantic subset.
- [x] Save checksum and model loading audit, including the legacy HF config limitation and module parameter counts. The tokenizer audit found and corrected a Transformers 5.5 Fast tokenizer incompatibility before any test evaluation.
- [x] Freeze training and evaluation configuration before the reportable optimization run.

## Task 2: Implement shared candidate training

- [x] Add tests for complete question+candidate composition, task-balanced draws, and correct pretrained BPE tokenization; run real candidate permutation and frozen-module gradient checks.
- [x] Add E5 model/data/training code using the existing FF++ manifests and language templates.
- [x] Run a real batch forward/backward smoke with all frozen-module gradients absent and no NaN.
- [ ] Train seed 42 only; select checkpoint by FF++ validation candidate CE only. Audit skipped steps and loss history. Current reportable run has completed epoch 3/5; earlier incomplete runs were isolated before test.

## Task 3: Stage B killer test

- [ ] Evaluate FF++ T2 seen and held-out with video aggregation and balanced accuracy.
- [ ] Evaluate polarity counterfactual on matched physical videos and candidate permutation consistency.
- [ ] If any Stage B gate clearly fails, stop and write FAIL artifacts; do not run Stage C or tune.

## Task 4: Conditional Stage C and final report

- [ ] Only if Stage B passes, evaluate FF++ T1–T5 seen/held-out, T4 family ranking, and permutation.
- [ ] Produce `reports/e5_a_final_report_zh.md` and machine-readable metrics JSON/CSV and gate status JSON.
- [ ] Verify every reported metric against saved prediction files and exit with PASS, FAIL, or INCONCLUSIVE. Do not start E5-B/C/D.
