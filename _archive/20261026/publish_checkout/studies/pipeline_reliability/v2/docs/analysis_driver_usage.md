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

This entry point handles one formal model at a time. The CPU five-seed assembly
entry below consumes completed outputs; it does not launch the per-model work.
Production statistics, figures, full-paper integration and final scientific
review still require the complete real experiment and cache matrix.


## Verified D five-seed assembly

After the five registered analyses for one backbone are complete:

```bash
python -m reliability_v2.summary_d status --config /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/configs/execution.json --backbone clip_vit_b16
python -m reliability_v2.summary_d run --config /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/configs/execution.json --backbone clip_vit_b16
```

Use the same CPU-only environment/thread limits as above. `status` is a marker
inventory, creates no directories, and cannot certify provenance. `run` requires
exactly seeds7/42/123/2026/3407 of the selected backbone, new frozen formal last
checkpoints, all seven current caches and the exact independently recomputed
analysis artifact inventory. It reuses read-only cache/context validators and
checks the producer identity without calling the dual-purpose `analyze_run`;
absent or partial per-model analysis is rejected, never silently launched. Development,
historical, stale, corrupt or substituted data cannot enter a completed summary.

One canonical output is `analysis/D/<backbone>/five_seed_summary/`. The twelve
CSV tables retain per-seed ranking metrics/effects, within-model CAL estimates
and subset spread, same-seed target-minus-source comparisons, oracle metrics and
numerical-status counts, with separate five-seed means and sampleSD(ddof1).
The default/source point and each budget10/20/40/80 remain separate; n20 is the
frozen primary budget. Missing numerical or CAL estimates produce explicit NA
five-seed means/SD rather than silently averaging four models. Reversal counts
and their valid-seed denominator remain distinct from mean reversal. Neither
200CALsubsets nor2000TESTdraws become training replicates or significance tests.

`supporting_artifacts.json` binds every original file by path/hash, preserving
per-seed normal/closure/operating conditional intervals,3x3ledgers and detailed
video/method diagnostics without duplicating large bootstrap files. CI endpoints
are not averaged into a new interval. `limitations.json` keeps CDF-full ranking
separate from the fixed RQ4 TEST, Raw/L2 and both aggregation flows separate, and
states the fixed-head/proxy-ancestry interpretation. High-precision oracle
thresholds remain serialized per seed; oracle never interpolates a ROC point.

Summary lifetime locks prevent duplicate writers. Input identities/hashes are
checked before use and again before commitment; completion is written last.
Reuse validates original inputs plus the exact summary inventory and hashes.
Partial identities or changed committed data/code are rejected and preserved.
A50GiB free-space reserve is required. This command performs no image decoding,
training, feature extraction, model/threshold selection, GPU scheduling or H
admission. Passing synthetic tests proves implementation behavior only; real
five-seed results remain unavailable until actual production caches/analyses pass.

All input/output markers, artifact files and lock paths reject symlinks (including
dangling links) and nonregular files. The private lock uses a nofollow file
descriptor; tables/JSON are atomically replaced from exclusive in-root scratch
files. Receipt content and completion hash are one verified snapshot.
