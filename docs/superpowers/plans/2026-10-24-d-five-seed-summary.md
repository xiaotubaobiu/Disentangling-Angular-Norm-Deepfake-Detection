> 历史执行计划：保留当时路径与决策。当前期刊主线为 `src/journal_detector`；旧实验产物已按用户要求清理，现行操作见根目录 README。

# D Five-Seed Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Add one CPU-only, provenance-checked D five-seed summary entry without changing running training, caches or statistical definitions.
**Architecture:** A single reliability_v2/summary_d.py consumes five completed per-model analysis_d outputs for one registered backbone. Reuse analysis_d completed-analysis validation and stats_h descriptive seed summaries; retain conditional per-seed intervals by bound references. Follow the existing H summary lock, inventory, resume and end-of-run revalidation pattern.
**Tech Stack:** Existing locked Python, pandas/numpy, pathlib/json, trainer locks; no new dependencies.
**Spec:** User-authorized frozen plan sections7–9; Obsidian 论文方法与设置准备_v2/20261024_覆盖整合版/D五种子汇总_冻结需求与接口核对_20261024T0730.md. Parent explicitly requested implementing this saved interface contract, using existing structure, tests and independent review. No new scientific design is being proposed.

## Global Constraints

- Formal seeds7/42/123/2026/3407, separate backbones and source_test/target_full/target_test cohorts.
- Same-seed deltas before seed mean/sampleSD(ddof1); calibration subsets200 and TEST draws2000 never become training seeds.
- No five-seed mean/SD when any registered seed estimate is unavailable. Preserve NA and descriptive reversal counts/denominators; tolerance1e-10.
- RQ4 target_real budgets10/20/40/80, primary20, every metric on its fixed TARGET_TEST; no endpoint interpolation or TEST threshold tuning.
- CPU only. Do not modify trainer/models/cache/scoring/statistics implementations, frozen config/data, training processes, GPU0–3 allocation, protected4–7 or H admission. Preserve all existing artifacts.
- One module, one new test file, update existing usage doc. No second scoring implementation or automatic analysis/training launcher.

## Review Focus

1. Missing/duplicate/unexpected formal seeds and wrong-backbone/development substitution must fail before output commitment.
2. Partial/corrupt/extra/unsafe analysis files or stale cache/final/config/code identities must reject both initial assembly and reuse.
3. One numerical NA must not become a four-seed main mean; per-seed operating counts remain available where ranking is NA.
4. CAL repeats must first reduce within model; unavailable budgets, common TEST hashes and nested subset identities remain explicit.
5. Concurrent launch, partial resume and an input/code mutation during assembly must never produce a stale complete marker.

### Task 1: CPU table assembly
**Files:** Create reliability_v2/summary_d.py; create tests/test_summary_d.py.
**Interfaces:** summarize_tables(per-seed collected tables, backbone) -> named pandas tables; uses scoring_d.summarize_seed_effects and stats_h.summarize_seed_comparisons. No raw-image/model inference.
- [x] Write failing tests for literal five-seed mean/sampleSD, within-seed interaction/reversal versus mean reversal, one NA, and separation of model/CAL layers.
- [x] Run focused tests; expect assertion that the missing entry/function is unavailable.
- [x] Implement minimal table collection/reduction for T4/T5, CAL point/spread and comparisons, oracle metrics, numeric statuses; preserve bound supporting-interval/diagnostic references.
- [x] Run focused tests and record RED→GREEN.

### Task 2: Verified filesystem entry and CLI
**Files:** Same module/tests; update docs/analysis_driver_usage.md.
**Interfaces:** summarize_backbone(config_path, backbone) -> completion receipt; summary_status(config_path, backbone) -> read-only marker availability. CLI status/run --config --backbone.
- [x] Add failing synthetic integration tests with real filesystem/hash/locking and actual analysis outputs, mocking only the slow model/cache boundary where needed; cover all Review Focus cases.
- [x] Verify RED, then implement exact5 registration, all7cache/current-analysis verification, exact artifact inventory, atomic marker-last writes, output lock,50GiB reserve, reuse/partial identity guards and input revalidation before commit.
- [x] Verify GREEN, read actual status against production R: waiting, no target processing/output creation.
- [x] Run full CPU test suite and compare protected source/config/data hashes.
- [x] Request one independent whole-change review, fix material findings with regression tests, re-run applicable/full checks, commit only reviewed additive module/tests/doc/plan.

## Decisions and self-review

The saved interface contract and parent request already authorize this scope. We implement inline then use an independent reviewer, retaining one canonical entry. The dedicated publish checkout is already isolated from the original /home/z/project project; work in place on its non-main feature branch, as requested, with frozen production modules untouched. Keep the reproducibility ledger and all failure/test receipts in R/reports rather than deleting them. No skill-local Windows bookkeeping script is executed against the remote Linux repo. Every requirement maps to Task1 or Task2; interfaces match; no additional statistical inference or cross-seed bootstrap CI is introduced.
