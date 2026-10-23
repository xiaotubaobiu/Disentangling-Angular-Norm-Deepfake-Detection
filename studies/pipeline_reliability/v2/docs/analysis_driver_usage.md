# D analysis entry point: frozen formal caches only

This note documents the new CPU analysis code. Its regression fixtures and
performance profiles are synthetic; they are not results of the detector study.
The running D queue is unchanged and stops after all formal FP32 caches complete.
No analysis watcher or additional training process is started by this module.

Run from `/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2`
with the isolated environment at
`/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python`.
Set `CUDA_VISIBLE_DEVICES=` and CPU thread variables to 2 for these commands.

```bash
python -m reliability_v2.analysis_d status \
  --config /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/configs/execution.json

# Only after the registered formal run and all seven role caches are complete:
python -m reliability_v2.analysis_d run \
  --config /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/configs/execution.json \
  --run-id D_formal_clip_vit_b16_seed7
```

`status` inventories formal cache markers only and explicitly labels that limit.
`run` verifies the live frozen formal training context, all seven frozen manifest
hashes, and each role cache against the caller's expected run/role identity.
The DEV cache is verified even though it is not scored as a final test pool.
Development checkpoints and historical caches cannot be substituted. The
original FP32 head must agree across roles. Completed analysis reuse checks
every saved analysis artifact hash and all seven live caches; changed or corrupt
receipts require investigation rather than silent replacement.

Outputs are under `artifact_root/analysis/D/<run_id>/`:

- `source_test/` and `target_full/`: raw FP64 frame margins, all eight video-score
  cells, pooled/macro AUC, within-seed effects, numerical controls and diagnostics,
  pair-transition ledgers, common source/collision-closure bootstrap, and four
  correct-grouped source-group bootstrap tables.
- `rq4/`: separate SOURCE_CAL, TARGET_CAL and fixed TARGET_TEST frame/score
  exports; four correct-grouped flows (Raw/L2 × mean-logit/mean-probability), each
  with defaults, source-real threshold, 200 common nested subsets at budgets
  10/20/40/80, three actual oracle vertices, independent saved-score recomputation,
  and the primary n20 target-versus-source operating comparison.
- `training_reference.json` and `training_radii.npy`: weighted TRAIN radius
  reference computed directly on canonical RGB rows and their summed sampling
  masses. TRAIN is never misinterpreted as eight-row natural-video groups.
- `identity.json`, `limitations.json`, `independent_rank_recomputation.json` and
  `complete.json`: provenance, scope, numerical NA conditions and artifact hashes.

The serialized rank field contains 350 significant decimal digits and a type
tag. Reloading preserves mpmath comparisons; the FP64 `score` remains the frozen
empirical threshold space. Unresolved ordering through 320 digits yields NA for
affected ranking outcomes and controls. It does not erase valid operating counts.
Oracle uses actual high-precision score boundaries and never interpolates FPR.

Independent AUC recomputation uses integer twice-concordance counts, separate
from the scorer. Independent operating recomputation counts `score >= threshold`
directly. Both reread saved score tables. RQ4 recomputes all metrics on its own
fixed TEST rather than importing CDF-full AUC. Missing registered methods
invalidate macro estimates. An unavailable n20 budget produces an explicit NA
record, not a copied or duplicated calibration sample.

Bootstrap kernels precompute precision-preserving tie bins, then apply exact
integer source/closure draw multiplicities. Regression tests compare against
explicitly materialized repeated draws, including repeated components and
missing-method replicates. The 2,000 TEST replicates, 200 CAL subsets and five
training seeds remain distinct. RQ4 TEST intervals condition on the frozen model,
full CAL pool and recorded subsets; CAL subset spread is a separate output.
The min-20-groups/max-0.5-largest-group gates are conservative operational rules,
not an independence or asymptotic adequacy theorem. Inadequate closure produces
descriptive per-seed effects without a fabricated paired CI.

Sparse bootstrap multiplicities are expanded when writing JSON. This preserves
exact replay but output serialization is O(repeats × groups); the synthetic
kernel profile excludes large-scale production JSON expansion. Measure actual
output sizes and respect the 200-GiB working-set estimate plus 50-GiB reserve.
The current CDF source scope is `PROXY_SCOPE_EMPIRICAL_ONLY`: no certified complete
ancestry or strict population Neyman–Pearson guarantee is inferred.

This entry point handles one formal model at a time. Final five-seed table
assembly, H model orchestration/caches/analysis, figures, full-paper integration
and final scientific review remain subsequent work. They must not be claimed as
completed by passing these synthetic regression tests.
