# Artifact Organization Design

## Goal

Organize retained experiment artifacts so video-level evaluation data and frame-level visualization data are clearly separated.

## Directory Layout

- `experiments/video_level/unified_decomposition_results.csv`: combined video-level AUC table.
- `experiments/video_level/unified_eval_results/`: per-experiment video-level evaluation CSVs.
- `experiments/video_level/unified_eval_logs/`: logs from video-level unified evaluation runs.
- `experiments/video_level/external_index_cache/`: cached external DFD video/frame index used by video-level evaluation.
- `experiments/video_level/dfd_threshold_analysis/`: DFD threshold analysis tables.
- `features/frame_level/`: frame-level feature, probability, label, and norm arrays used for visualization.
- `figures/frame_level/`: frame-level visualization outputs grouped by plot type.

Timestamped experiment directories under `experiments/2026.../` remain in place because they are raw training/evaluation artifacts with configs, logs, checkpoints, and summaries.

## Metric Semantics

Video-level experiment checks must use the unified evaluation outputs. These aggregate frame rows to video predictions before computing AUC.

Frame-level arrays under `features/frame_level/` are for plots such as score distributions, norm histograms, threshold curves, calibration plots, cosine heatmaps, and t-SNE/UMAP views. They should not be used as the source of paper-level video AUC claims unless a script explicitly performs video aggregation.

For DFD, ordinary accuracy is secondary because the dataset is highly imbalanced. AUC and balanced accuracy are the preferred checks.

## Compatibility

Scripts that default to artifact paths should be updated to the new layout. The repository documentation should state the new paths and the frame-level versus video-level distinction.

PDF figure deletions are accepted as a format migration to PNG outputs. The organization work should not restore the old PDFs.
