# Task 6 implementation report

Status: complete. Commit: `b665188` (`feat: add E2 semantic control suites`).
Base: `6fb7221088d3504cad9401fb504bb071fa3b4f13`.

## Scope and implementation

Resumed and preserved the interrupted implementation. Committed exactly:

- `src/evaluation/semantic_controls.py`
- `tests/test_semantic_controls.py`
- `experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py`

The unrelated migration and pre-staged `src/deepfake_detection/__init__.py` →
`src/__package_init_old.py` rename remain untouched and staged as before. No
language-bank strings, model selection rules, gate thresholds, datasets,
training parameters, or Task 5 metric implementation changed.

Public controls enumerate all 2!/3! candidate orders and remap probabilities
to identity order; build exact opposite-polarity T2 and family-paired T4
requests; and create manifest-only meaningless candidates/questions with
hidden evaluation metadata. The evaluator now dispatches all seven accepted
conditions to their correct inference path. Heldout/control manifest checks
run before text encoding. Text lookups encode heldout text plus only the
control kind needed by the selected meaningless condition.

Permutation auditing selects the first 4,096 lexically sorted unique physical
sample IDs per domain before task expansion and rank sharding. Every eligible
task for those samples receives all valid orders. Rank shards do not pad or
duplicate renders. Image features are extracted once per audited batch.
Maximum error and counts reduce after the local loops, so uneven or empty
rank loops do not deadlock. The standalone public audit still reduces by
default; the evaluator disables its per-batch collective and performs one
final MAX and SUM aggregation.

Counterfactual requests carry a common physical `question_pair_id`,
`render_id`, and `control_pair_id`. Physical pairs are checked for both
polarities before video pooling. T2 logical consistency and fake-only T4
true-family ordering use video+polarity averages, with one output row per
video pair. They never join by row order. Physical IDs remain in frame output
and are normalized for Task 5 video aggregation so they do not split each
video into separate frame groups.

Meaningless controls score their exact meaningful counterparts on the same
cached batch image features. Original question/variant identity is preserved
as evaluation metadata even when the model receives meaningless question
text. This preserves Task 5's strict T4 swap/reenact pair validation. Actual
question control IDs are retained separately as `control_question_id`.
Mixed binary/ternary hidden metadata is padded for normal batch collation.

## Task 7 data contract

For each condition/domain, ordinary and control evaluation writes
`{domain}_frames.npz`, `{domain}_videos.csv`, and `{domain}_metrics.json`.
Permutation writes normalized metrics only. Meaningless controls additionally
write `{domain}_paired_videos.csv`; counterfactual writes
`{domain}_controls.csv`.

Normalized metrics/control records contain `seed`, `domain`, `condition`,
`task_id`, `metric`, `value`, `count`, and `pairing_key`.

The paired-video file stores control `p_real`, `p_fake`, `p_no`, `p_yes`,
`p_camera`, `p_swap`, `p_reenact`, `prob_fake`, and `num_frames`, alongside
the exact counterpart's `meaningful_` prefixed versions. Pairing is an outer
one-to-one join, validated to match on all Task 5 GROUP_KEYS:

`seed, domain, condition, task_id, variant_id, question_id,
question_yes_means, question_pair_id, method_id, video_id, video_key, label,
method_family, target_semantic`.

Frame counts must also match. `pairing_key` serializes those full keys.
Task 5 deliberately retains language variants in its video groups; a physical
video can therefore have multiple paired rows. Task 7 must resample physical
`video_key` clusters and carry all their question/variant rows together for
both conditions, rather than treating each row as an independent video.

G6 rows are actual paired primary performance drops:

- T1/T3: `auroc_drop`
- T2/T4: `balanced_accuracy_drop`
- T5: `macro_balanced_accuracy_drop`

They include `primary_metric`, `meaningful_score`, `control_score`, `drop`,
and `value = meaningful_score - control_score`. Metrics use the existing
Task 5 metric implementation over the same paired rows. A frame-level target
confidence difference remains explicitly named
`target_probability_drop_diagnostic`; it is not a G6 performance metric and
must not be used as a gate. No bootstrap interval or support decision is
claimed here: Task 7 owns the preregistered 10,000 paired video-cluster
resamples, three-seed estimates, intervals, and all numeric gate decisions.

## Balance correction authorized by controller

Removed the interrupted implementation's fixture-tuned
`meaningless_candidates:hidden:1` namespace. The fixed, semantically named
namespace is `meaningless_candidates:hidden`; independent surface order uses
`meaningless_candidates:surface`. Both depend only on control name and
sample ID, not labels, target semantics, candidate positions, or model scores.
Canonical semantic ordering removes dependence on the meaningful candidate
order. No namespace search or final-dataset tuning occurred.

The brief's original 400-ID test has observed `(target_position,label)` counts:

- `(0,0)`: 101; `(0,1)`: 89; `(1,0)`: 99; `(1,1)`: 111.
- Spread: 22, which misses the illustrative <=20 fixture threshold.

The controller explicitly authorized replacing that one-draw threshold with
label-flip/candidate-order invariance, a fixed independent 4,000-ID population,
and a bound fixed before computing its outcomes. The 4,000-ID counts are:

- `(0,0)`: 991; `(0,1)`: 953; `(1,0)`: 1009; `(1,1)`: 1047.
- Every cell is within the declared expected 1,000 ±100 bound (over four
binomial standard deviations per label/position cell).

A separate fixed 6,000-ID population checks the joint output-position,
pseudoword, and hidden-semantic cells for binary and ternary tasks against
a predeclared six-standard-deviation bound. Both pass. These tests establish
label/position independence and statistical counterbalancing, not impossible
exact finite-set equality from a stateless hash.

## Verification

The prior missing-module RED was preserved as controller-provided evidence.
New metric/integration RED observed nine failures: the old confidence-based
metric helper did not consume paired video data, and the evaluator lacked
the dedicated dispatcher. A missing-physical-partner regression also failed
before pair validation was added. Mixed-arity collation failed during real
loader integration and was corrected by padding hidden metadata.

- `pytest -q tests/test_semantic_controls.py tests/test_semantic_eval.py tests/models/test_clip_semantic_jev.py`:
  **67 passed**, 30 warnings, 13.06 seconds, exit 0.
- `pytest -q` (one final full-suite run): **211 passed**, 32 warnings,
  33.87 seconds, exit 0.
- `git diff --check`: clean.

Coverage includes actual control loader → dispatcher → serialized output
paths, all five primary G6 metric types, strict T4 pairing, manifest gating,
minimal text scopes, 4,096 physical samples before expansion, missing pair
rejection, position bias detection, and a real two-process CPU Gloo test with
one rank processing zero batches. Model invariance tests also pass.

Warnings: existing CLIP/pkg_resources namespace deprecations, existing
GradScaler/Jupyter deprecations, and sklearn warnings from intentionally
small per-method test slices whose predictions contain a class absent from
that slice. No failures were omitted. No trained checkpoint or final test
evaluation was run by this task.

Self-review checked the dispatcher/main wiring, no-padding distributed audit,
hidden-model-input separation, T4 frame and video pairing, exact G6 primary
metric identity, Task 5 grouping preservation, and commit path isolation.

## Operational seam / remaining work

The current batch cache prevents repeated visual extraction within a
permutation audit or meaningful/control pair. It does not reuse frozen
per-frame features across tasks, batches, conditions, or seeds. Task 8 can
introduce a physical-sample feature store keyed by visual-tower provenance,
preprocessing provenance, domain, and `sample_id`, then have semantic
datasets render metadata over those cached features. The clean inference
seam is `extract_features` → `classify_features`; feature-store construction
should happen before semantic expansion to avoid repeated image decoding.
Do not reduce the full eligible domain scope or cache trainable projections
across checkpoints. This optimization is not necessary for Task 6 correctness
and is not implemented here.
