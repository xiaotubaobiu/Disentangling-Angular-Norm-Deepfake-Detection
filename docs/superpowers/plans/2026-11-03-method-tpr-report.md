# Saved-score method TPR reporting implementation plan

> For agentic workers: use superpowers:executing-plans inline. This is an already authorized part of the experiment, not a new training study.

**Goal:** Export the method-level TPR required by the approved RQ4 plan from saved video scores and saved calibration thresholds.

**Architecture:** A CPU-only script outside the frozen scientific package derives method counts using the original FP64 `score >= threshold` rule. Every available operating point must reconstruct the saved FP, TP, pooled TPR/BA and available macro TPR/BA. Supplemental files stay outside the immutable 65-file analyses. Seed aggregation waits for the original strict five-seed summary.

**Tech stack:** Python, pandas, NumPy, Parquet; no model loading.

**Spec:** Existing Obsidian experiment plan, section 9: report method-level TPR alongside FPR/TPR/BA, using identical TEST and the registered CAL subsets.

## Global constraints

- Existing training/scoring/analysis modules, hashes, data, thresholds and recipes remain unchanged.
- Operating points use saved FP64 scores; high-precision ranking scores serve a different purpose.
- Decisions include ties (`>=`). Natural-video weights are one.
- Retain every fixed method, every available CAL subset and explicit unavailable n80 rows.
- CAL-subset variation is reduced within a model before describing variation across exactly five training seeds.
- Numeric missingness stays NA; incomplete seeds cannot become a full summary.

## Review focus

- Unequal fake-method counts must reconstruct different pooled and macro TPR: literal 3/1 fake counts test.
- Scores exactly equal to threshold must count positive: same literal test.
- Unavailable CAL budgets must yield NA rather than zero: unavailable-row test.
- Missing seeds or methods must fail rather than silently reduce the denominator: missing-method/seed tests.
- Changed saved inputs must fail hash binding: actual-file mutation test.

## Task 1: Derive method operating points and subset summaries

**Files:** create `scripts/report_method_tpr.py`; create `tests/test_method_tpr_report.py`.

**Interfaces:** `method_operating_rows(videos, calibration, methods)`; `subset_summary(rows)`; `five_seed_summary(rows)`; `report_run(root, run)` and `report_backbone(root, backbone)`.

- [x] Write literal tests for threshold ties, macro/pooled reconstruction, unavailable budgets, wrong saved counts, missing methods, 200-subset spread, exact five-seed means/SD and missing estimates.
- [x] Run the new test file and observe expected missing-feature failures.
- [x] Implement the numerical functions and hash-bound saved-input readers. Return NA for unavailable estimates and preserve counts.
- [x] Run focused tests; fix actual failures without weakening independent expectations.
- [x] Implement CLI run/backbone dispatch and supplementary output receipts under `reports/D_method_operating/`.
- [x] Run the full project CPU test suite with CUDA hidden.
- [x] Process all six currently completed D models; verify every valid saved FP/TP and macro value, raw/summary counts and unchanged upstream hashes.
- [x] Document the command and review the implementation before publication.
- [ ] Run the same script for later finished models and aggregate after original strict five-seed summaries.

## Verified implementation checkpoint

Nine literal regression tests were observed failing before the new script existed, then passed after implementation. The full CPU suite exited0:196passed/21upstream warnings in504.60s. Six real models each reconstruct2,408valid operating points and retain67,368per-subset method rows plus504model summaries. All65original artifact hashes per analysis still match. The independent read-only review found no correctness issues; the original strict five-seed summary has not yet completed, so successful whole-backbone integration remains pending.

Evidence under the actual artifact root: `operations/method_tpr_report_red.log`, `method_tpr_report_green.log`, `method_tpr_full_cpu_suite_acceptance.json` and `method_tpr_six_model_acceptance.json`. This checkpoint covers the saved-score supplement; it does not claim completion of all33models or replace strict raw-input/cache acceptance.
