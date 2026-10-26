# Threshold-transfer input audit

- config: `experiments/2026-07-30_threshold_transfer/configs/threshold_study.yaml`
- source domain: FF++ **validation** (`experiments/2026-07-27_pooling_saturation/results/val_cache`)
- target domain: `cdf` from the phase-2 frame parquet
- excluded: dfd

**34/34 inputs present.**

| item | path | exists | detail |
|---|---|---|---|
| frame_scores parquet | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 1.38 GB |
| checkpoint root | experiments/2026-05-05_paper_feature_geometry/results/checkpoints | True | 30 checkpoint dirs |
| FF++ val cache dir | experiments/2026-07-27_pooling_saturation/results/val_cache | True | 15 parquet files |
| val cache clip_raw_s7 | experiments/2026-07-27_pooling_saturation/results/val_cache/clip_raw_s7__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache clip_raw_s42 | experiments/2026-07-27_pooling_saturation/results/val_cache/clip_raw_s42__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache clip_raw_s123 | experiments/2026-07-27_pooling_saturation/results/val_cache/clip_raw_s123__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache clip_raw_s999 | experiments/2026-07-27_pooling_saturation/results/val_cache/clip_raw_s999__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache clip_raw_s2048 | experiments/2026-07-27_pooling_saturation/results/val_cache/clip_raw_s2048__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache dinov2_raw_s7 | experiments/2026-07-27_pooling_saturation/results/val_cache/dinov2_raw_s7__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache dinov2_raw_s42 | experiments/2026-07-27_pooling_saturation/results/val_cache/dinov2_raw_s42__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache dinov2_raw_s123 | experiments/2026-07-27_pooling_saturation/results/val_cache/dinov2_raw_s123__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache dinov2_raw_s999 | experiments/2026-07-27_pooling_saturation/results/val_cache/dinov2_raw_s999__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache dinov2_raw_s2048 | experiments/2026-07-27_pooling_saturation/results/val_cache/dinov2_raw_s2048__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache efficientnet_raw_s7 | experiments/2026-07-27_pooling_saturation/results/val_cache/efficientnet_raw_s7__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache efficientnet_raw_s42 | experiments/2026-07-27_pooling_saturation/results/val_cache/efficientnet_raw_s42__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache efficientnet_raw_s123 | experiments/2026-07-27_pooling_saturation/results/val_cache/efficientnet_raw_s123__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache efficientnet_raw_s999 | experiments/2026-07-27_pooling_saturation/results/val_cache/efficientnet_raw_s999__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| val cache efficientnet_raw_s2048 | experiments/2026-07-27_pooling_saturation/results/val_cache/efficientnet_raw_s2048__val.parquet | True | 25584 frames / 3200 videos / 17 methods / 12792 fake frames |
| target cdf clip s7 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf clip s42 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf clip s123 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf clip s999 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf clip s2048 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf dinov2 s7 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf dinov2 s42 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf dinov2 s123 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf dinov2 s999 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf dinov2 s2048 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf efficientnet s7 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf efficientnet s42 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf efficientnet s123 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf efficientnet s999 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| target cdf efficientnet s2048 | experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet | True | 792568 frames (raw+direction) / 2154 videos / 17 methods |
| excluded domains present in parquet | dfd | True | 6200340 rows exist but are never read by this phase |
