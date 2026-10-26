# H formal caches and CPU analysis

Implementation tests are separate from production data acceptance, pretrained
GPU qualification and scientific results. No production H cache or analysis was
generated during implementation. Existing D training and H preprocessing keep
their original code/configuration and processes.

## Prerequisites and seven roles

Finish all H pixels, run the repaired full audit in a fresh process, obtain the
hash-bound independent production data review and freeze, then complete real H
source qualification, development and formal training. Each backbone must pass
all five effective paired numerical ledgers before target export. If uniform
FP32 fallback was required, only those effective FP32 runs are eligible; retained
AMP models cannot be substituted. `H_run_registry.json` must finish with
`H_FORMAL_COMPLETE_TARGET_CACHE_PENDING` and `target_export_allowed: true`.

Each effective model exports these seven role caches under
`features/H/<run_id>/<role>/`:

```text
B/SOURCE_CAL
B/SOURCE_TEST
B/TARGET_TEST
common/B/SOURCE_TEST
common/B/TARGET_TEST
common/AB/SOURCE_TEST
common/AB/TARGET_TEST
```

The original FP32 two-output head and raw FP32 features are retained. Head rows
and biases are promoted to FP64 before subtraction. Export is eval/FP32 with
autocast disabled. Chunk/file hashes, exact role alignment, input PNG bytes/RGB,
checkpoint/head identity, H and training bundles, environment, gates and paired
ledgers are checked. A committed corrupted chunk stops; an uncommitted partial
chunk may be regenerated. No source-best or development model can infer targets.

`cache_h.extract` owns one approved idle GPU0–3 for the complete export. It
rechecks physical UUIDs and D priority, uses the shared lifetime GPU lock, and
checks capacity before allocation. Schedule at most four independent exporters;
do not launch them while existing jobs own those devices. GPU4–7 remain protected.
The H training queue itself ends before this separate cache/analysis stage.

## Entry points

Run from the existing v2 code directory using `R/environment/bin/python`, where
`R` denotes the artifact root. Read `effective_runs` in `H_run_registry.json` to
obtain eligible run IDs. Never assume an AMP run remained effective.

```text
# Read-only; no GPU allocation or result directory is created.
CUDA_VISIBLE_DEVICES= R/environment/bin/python -m reliability_v2.analysis_h status --root R

# Future cache launch: CUDA_VISIBLE_DEVICES must be one approved idle UUID.
R/environment/bin/python -m reliability_v2.cache_h --config R/configs/H_execution.json --run-id <effective-run-id>

# CPU pair analysis, after both histories have all seven verified role caches.
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 R/environment/bin/python -m reliability_v2.analysis_h pair --config R/configs/H_execution.json --backbone <backbone> --seed <formal-seed>

# After all five pair analyses of this backbone have completed.
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 R/environment/bin/python -m reliability_v2.analysis_h summary --config R/configs/H_execution.json --backbone <backbone>
```

Backbones: `clip_vit_b16`, `dinov2_vit_b14`, `efficientnet_b0`. Formal seeds:
`7, 42, 123, 2026, 3407`. Pair analysis outputs to
`analysis/H/<backbone>/seed<seed>/`; summaries to `five_seed_summary/`.

## Comparisons and acceptance

Raw mean-logit is primary; Raw mean-probability is sensitivity only. H adds no
L2 model, collision grouping or target-real calibration pool. All-B deployment
compares H10−H00 on every independent TEST video where B succeeds. The separate
common B/AB cohort produces H00/H01/H10/H11 and its own H10−H00, test-history
effects, training effect on AB and interaction. No all-B/common subtraction is
reported. FF source validation and DFD target tests remain distinct.

Each model selects its empirical FPR≤5% minimal `nextafter` threshold from its
own all-B `B/SOURCE_CAL` real scores. Its AB test cells reuse that B deployment
threshold. Real-only CAL never enters AUC. Default thresholds remain 0/logit and
0.5/probability. Operating decisions use FP64 `score >= threshold`; AUC retains
precision-safe ranking objects. Unresolved numerical ordering yields AUC NA,
while reportable operating metrics remain available.

Each cohort uses 2,000 common source/proxy-group bootstrap draws with exact
integer multiplicities across cells. Repeated groups are preserved. Missing
fixed methods invalidate macro draws; absent classes invalidate pooled draws.
Too few/concentrated groups yield no interval. Intervals are pointwise,
conditional on the fixed model and known/proxy relationships. DFD proxies do not
certify full ancestry. Five training-seed summaries require exactly five rows
per stratum and sample SD (`ddof=1`); missing seed estimates are not dropped.

The pair completion receipt binds a fixed 72-file inventory, all input/cache
markers and analysis dependencies. Saved score exports are re-read and AUC and
integer operating counts independently recomputed. Complete reuse re-verifies
the inventory/hashes; source/code changes require a separately registered
analysis revision. Five-seed summaries bind all five accepted pair completions
and a fixed six-file inventory. Both paths retain the 50 GiB storage reserve.

`coverage_h.report_coverage` requires the accepted H freeze and reports every
candidate, processing eligibility, global/video/frame failures, administrative
quarantine/SOURCE_UNUSED exclusions, class/method distributions, all-B/common
coverage, original/A-redetection multiface and padding denominators. Missing
geometry metadata is explicit. Failures are preserved without score-based
exclusion or frame replacement. This is coverage accounting, not missing-at-
random inference.

Final journal tables/figures and bilingual master integration still require
actual completed formal results and a separate final claim/result audit.
