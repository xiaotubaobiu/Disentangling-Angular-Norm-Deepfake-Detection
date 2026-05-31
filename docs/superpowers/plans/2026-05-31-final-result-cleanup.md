# Final Result Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep concise final experiment results and figures while deleting large or temporary intermediate artifacts.

**Architecture:** Copy canonical result CSVs into `experiments/final_results/`, document the experiment semantics there, then delete caches and duplicate intermediate outputs. Keep timestamped experiment directories and PNG figures for reproducibility and presentation.

**Tech Stack:** Bash filesystem operations, Python/pandas CSV verification, Markdown documentation.

---

### Task 1: Create Final Results Directory

**Files:**
- Create: `experiments/final_results/auc_video_level_60rows.csv`
- Create: `experiments/final_results/dfd_balanced_thresholds_45rows.csv`
- Create: `experiments/final_results/dfd_balanced_thresholds_summary.csv`
- Create: `experiments/final_results/README.md`

- [ ] **Step 1: Create the final result directory**

Run:

```bash
mkdir -p experiments/final_results
```

Expected: command exits with status 0.

- [ ] **Step 2: Copy canonical CSVs**

Run:

```bash
cp experiments/video_level/unified_decomposition_results.csv experiments/final_results/auc_video_level_60rows.csv
cp experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_all_backbones.csv experiments/final_results/dfd_balanced_thresholds_45rows.csv
cp experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_summary.csv experiments/final_results/dfd_balanced_thresholds_summary.csv
```

Expected: all three final CSVs exist.

- [ ] **Step 3: Write final result README**

Create `experiments/final_results/README.md` explaining the 30 models, 60 AUC rows, 45 balanced-threshold rows, retained figures, and deleted feature cache.

### Task 2: Delete Intermediate Outputs

**Files:**
- Delete: `features/frame_level/`
- Delete: `experiments/video_level/unified_eval_results/`
- Delete: `experiments/video_level/unified_eval_logs/`
- Delete: `experiments/video_level/dfd_threshold_analysis/normtrain_logs/`
- Delete: `experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_normtrain.csv`
- Delete: `experiments/video_level/dfd_threshold_analysis/dfd_best_balanced_thresholds.csv`
- Delete: `experiments/video_level/dfd_threshold_analysis/dfd_best_thresholds.csv`

- [ ] **Step 1: Remove large frame-level feature cache**

Run:

```bash
rm -rf features/frame_level
```

Expected: `features/frame_level` no longer exists.

- [ ] **Step 2: Remove duplicate video-level intermediate files**

Run:

```bash
rm -rf experiments/video_level/unified_eval_results
rm -rf experiments/video_level/unified_eval_logs
rm -rf experiments/video_level/dfd_threshold_analysis/normtrain_logs
rm -f experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_normtrain.csv
rm -f experiments/video_level/dfd_threshold_analysis/dfd_best_balanced_thresholds.csv
rm -f experiments/video_level/dfd_threshold_analysis/dfd_best_thresholds.csv
```

Expected: final result CSVs remain under `experiments/final_results/`.

### Task 3: Verify Final State

**Files:**
- Read: `experiments/final_results/*.csv`
- Read: `figures/frame_level/**/*.png`
- Read: `experiments/**/best_model.pth`

- [ ] **Step 1: Verify CSV row counts and grouping**

Run:

```bash
python - <<'PY'
import pandas as pd
auc = pd.read_csv('experiments/final_results/auc_video_level_60rows.csv')
thr = pd.read_csv('experiments/final_results/dfd_balanced_thresholds_45rows.csv')
summary = pd.read_csv('experiments/final_results/dfd_balanced_thresholds_summary.csv')
assert len(auc) == 60, len(auc)
assert len(thr) == 45, len(thr)
assert len(summary) == 9, len(summary)
counts = thr.groupby(['backbone', 'variant']).size()
assert (counts == 5).all(), counts
print('final CSVs ok')
PY
```

Expected: prints `final CSVs ok`.

- [ ] **Step 2: Verify retained and deleted artifact counts**

Run:

```bash
test ! -e features/frame_level
test "$(find figures/frame_level -type f -name '*.png' | wc -l)" -eq 87
test "$(find experiments -maxdepth 3 -name 'best_model.pth' | wc -l)" -eq 30
```

Expected: all commands exit with status 0.

- [ ] **Step 3: Run repository tests**

Run:

```bash
python -m pytest
```

Expected: all tests pass.
