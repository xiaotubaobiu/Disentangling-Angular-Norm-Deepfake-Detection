# Task 7 implementation report

Status: complete. Commit: `5665c385dc246acbc5556cd4a96bc04137dd3a24`
(`feat: preregister E2 gate adjudication`). Base: `b665188`.

Committed exactly:

- `src/evaluation/e2_gates.py`
- `tests/test_e2_gates.py`
- `experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py`

The unrelated migration is untouched. The pre-staged rename from
`src/deepfake_detection/__init__.py` to `src/__package_init_old.py` remains
staged as R100 after the explicit-pathspec commit.

## Bootstrap and controller rulings

The primary estimate is the mean paired difference over exactly seeds 42, 7,
and 2048. Physical `method_id::video_id` clusters are stratified by label and
method. Every question/variant row stays inside its video cluster. Both paired
conditions use the same sampled multiplicities.

The controller explicitly corrected the brief's independent-per-seed draw
pseudocode: the **same physical-video multiplicities are shared across all
three fixed seeds**, preserving cross-seed covariance on the common videos.
Seeds are not resampled. Physical populations/labels/strata must match across
seeds. Language variants may differ across seeds because the actual renderer
depends on the experiment seed; full meaningful/control variant identities
must match exactly within each seed.

Final CLI adjudication always uses exactly 10,000 resamples. G1 uses the 5th
percentile lower bound; G6 uses the 2.5th percentile and strictly requires it
to exceed zero. A displayed upper bound is the 97.5th percentile. Numeric
score sorting/ties and class-recall sufficient statistics are prepared before
bounded batches of multinomial cluster draws; no DataFrames are rebuilt in
the resample loop. Weighted AUROC handles score ties exactly.

All six gate thresholds are hard-coded and supplied threshold overrides are
rejected. Any point failure yields `not_supported`; otherwise any required
CI failure yields `insufficient_evidence`; only all passes yield `supported`.
There is no calibration or secondary-metric override.

The controller also ruled that FF++/CDF are required and DFD semantic
performance stays exploratory. If any DFD permutation artifact exists, all
three seed audits are required, and all supplied DFD audits contribute to G3.
A DFD audit failure fails G3; a partial audit is invalid evidence.

## Artifact contract and fail-closed behavior

The CLI supports the planned arguments:

```text
python experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py \
  --evaluation-root <evaluation root containing seed_42, seed_7, seed_2048> \
  --baseline-root <H1 root containing the same seed directories> \
  --output-root <results/summary> \
  --report <reports/e2_validation_result.md>
```

Each E2 seed requires all seven condition directories and primary-domain
metrics files. Fixed-H1 uses `{domain}_videos.csv`; meaningless conditions use
`{domain}_paired_videos.csv`; counterfactuals use `{domain}_controls.csv`.
H1 input is exactly `{domain}_video_predictions.csv`; seed/domain metadata
comes from its path. String identifiers retain leading zeroes.

G1 recomputes paired AUROC and checks the stored Fixed-H1 metric. G2/G4/G5
consume normalized global (`method_id == __all__`) primary metrics. G3 uses
global maxima/minima, not seed averages. Counterfactual consistency and
ordering are physical-video binary records, with exact required populations
and fake-only T4 pairs. G6 recomputes each actual primary drop from paired
probabilities; it never uses target-confidence diagnostics. Stored drop,
meaningful/control scores, and row counts must agree with the probabilities.

Missing seeds/domains/tasks, malformed/nonfinite primary evidence, duplicate
pairs, mismatched variants/labels/frame counts, null identities, truncated
paired files, and inconsistent stored counts/scores are rejected before final
resampling. Undefined secondary per-method AUROC remains a printable NA,
because legitimate single-class slices have no defined AUROC.

Successful adjudication writes `seed_metrics.csv`, `bootstrap_intervals.csv`,
`gate_status.json`, `gate_status.csv`, and the requested Markdown report.
Evidence includes exact thresholds, point and confidence decisions,
per-seed scores/deltas and sample SD, physical/variant counts, source paths,
per-method diagnostics, and calibration diagnostics. Library diagnostic
runs may reduce resamples only with explicit `final=False`; reports label
these NONFINAL. There is no reduced-count CLI option.

CLI exit 0 means adjudication completed, even if its verdict is not supported.
Invalid/missing evidence returns exit 2, writes `insufficient_evidence` with
`adjudicated=false` and the error, and clears stale summary measurements so a
previous supported result cannot survive as the current report.

## Verification

TDD RED began with the required missing-module import failure. Additional RED
tests exposed and then closed count/metric contradictions, cross-seed variant
overrestriction, ignored optional DFD audits, stale intervals after failed
reruns, and corrupted physical labels/null identities.

- Focused gate/comparison iteration: **33 passed**, 16.67 seconds; the final
  two integrity regressions subsequently passed and are included below.
- Final full suite, run once after implementation:
  `CUDA_VISIBLE_DEVICES='' pytest -q`: **243 passed, 32 warnings**, 50.62 seconds,
  exit 0. This includes all 32 gate tests and the existing comparison tests.
- `git diff --cached --check -- <three scoped files>`: clean.
- Real Task 6 CPU integration: generated all seven conditions for three seeds
  and two domains through the existing deterministic test model and actual
  evaluator/writer, then consumed the artifacts through the new loader and
  gates. Correct result: `not_supported` (G1/G2/G3 pass, G4/G5/G6 fail).
- Full final CLI synthetic fixture: all 16 required bootstrap intervals used
  **10,000** resamples, with final=true. A subsequent missing-artifact run
  replaced that success with invalid-evidence output and empty stale tables.
- Weighted AUROC/balanced-accuracy calculations match sklearn reference
  calculations with nonuniform physical weights and repeated variants.
  An independent shared-draw reference checks the exact 5th/2.5th percentiles.
- CPU performance benchmark: **28.024 seconds** for one exact paired AUROC
  bootstrap, **10,000 physical videos × three seeds × 10,000 resamples**.
  Larger language suites retain all rows; runtime scales with that row count.

Warnings are existing CLIP/pkg_resources namespace deprecations, sklearn
warnings from intentionally single-class test slices, and existing GradScaler
and Jupyter deprecations. No test failures were omitted. No GPU was used and
no trained E2 checkpoint or final scientific evaluation was run.

## Remaining operational concerns

The adjudicator is ready for Task 9 artifact roots. Full final runtime depends
on the actual number of retained language variants; batching bounds temporary
numeric arrays, but all required 10,000 exact draws remain mandatory. The
paired G6 artifacts are the authoritative exact meaningful counterparts;
they need not be identical to a separately rendered heldout suite. Input root
layout is explicit, with no ambiguous recursive discovery of H1 predictions.

## Review fix round 1

The independent Task 7 review found that the frozen final command needed
`--config`, and G2/G4/G5 could accept tampered aggregate metrics without
checking the underlying seen/heldout videos. The fix adds the required config
path and validates locked domains, thresholds, bootstrap count, and seed. It
loads both canonical video suites for every seed/domain and recomputes each
primary metric from their probabilities. The stored metric and row count must
match; physical video keys, labels, method families, target semantics, task
classes, and aggregate frame coverage must also match the H1 baseline.
Incomplete T4 question pairs are rejected.

The predecessor implementation agent exhausted workspace credits after
leaving the fix and tests uncommitted. Inline execution resumed under the
existing plan. Fresh focused tests: 52 passed in 33.09 seconds. Fresh full
CPU suite: 260 passed, 32 existing warnings in 66.04 seconds, exit 0.
`git diff --check` passed for the three Task 7 files. A separate fresh review
of this fix was unavailable because a second implementation subagent also
exited immediately for workspace credits; final whole-branch review remains
required before reporting a scientific verdict.
