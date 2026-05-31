# Final Result Cleanup Design

## Goal

Make the experiment artifacts concise and easy to read by keeping final results, figures, and reproducibility roots while deleting large or temporary intermediate outputs.

## Keep

- Video-level AUC table with 60 rows:
  - source: `experiments/video_level/unified_decomposition_results.csv`
  - final name: `experiments/final_results/auc_video_level_60rows.csv`
- DFD balanced-threshold table with 45 rows:
  - source: `experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_all_backbones.csv`
  - final name: `experiments/final_results/dfd_balanced_thresholds_45rows.csv`
- DFD balanced-threshold summary:
  - source: `experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_summary.csv`
  - final name: `experiments/final_results/dfd_balanced_thresholds_summary.csv`
- Final frame-level PNG figures under `figures/frame_level/`.
- Timestamped experiment directories under `experiments/2026.../`, including checkpoints, configs, logs, and summaries. These are the reproducibility roots for the 30 models.
- Utility scripts needed to recompute unified AUC and DFD normtrain thresholds.

## Delete

- `features/frame_level/`: 32GB of frame-level cached arrays with `features`, `labels`, `probs`, and `norms`. These arrays are only needed to regenerate frame-level plots. Existing final plots are retained.
- `experiments/video_level/dfd_threshold_analysis/normtrain_logs/`: temporary logs from the normtrain threshold补算.
- Intermediate duplicate video-level outputs once final result copies are written:
  - `experiments/video_level/unified_eval_results/`
  - `experiments/video_level/unified_eval_logs/`
  - `experiments/video_level/dfd_threshold_analysis/dfd_balanced_thresholds_normtrain.csv`
  - `experiments/video_level/dfd_threshold_analysis/dfd_best_balanced_thresholds.csv`
  - `experiments/video_level/dfd_threshold_analysis/dfd_best_thresholds.csv`
- The old PDF figure outputs are not restored. PNG is the retained figure format.

## Final Documentation

Create `experiments/final_results/README.md` explaining:

- 30 models = 3 backbones x 5 seeds x 2 train modes.
- 60 AUC rows = raw train with `raw`, `direction`, `length` inference plus normtrain with `norm` inference.
- 45 DFD balanced-threshold rows = `raw`, `direction`, and `normtrain_norm`, each with 3 backbones x 5 seeds.
- Frame-level figures are visualization outputs. Main experiment claims should use the video-level CSVs.
- Deleted `features/frame_level/` was a cache and can be regenerated from checkpoints.

## Verification

After cleanup:

- `experiments/final_results/auc_video_level_60rows.csv` has 60 rows.
- `experiments/final_results/dfd_balanced_thresholds_45rows.csv` has 45 rows and every `(backbone, variant)` group has 5 seeds.
- `experiments/final_results/dfd_balanced_thresholds_summary.csv` has 9 rows.
- `figures/frame_level/` still contains 87 PNG files.
- `find experiments -maxdepth 3 -name best_model.pth` still returns 30 checkpoints.
- `features/frame_level/` no longer exists.
