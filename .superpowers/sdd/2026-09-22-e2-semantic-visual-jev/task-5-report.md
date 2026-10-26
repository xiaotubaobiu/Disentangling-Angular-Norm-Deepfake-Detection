# Task 5 report: canonical semantic evaluation and video metrics

## Result

Implemented and committed the three authorized Task 5 files in commit
`0f4837f26332676c52a25d14eaf3979d948fdbbb` (`feat: add canonical E2 semantic
evaluation`):

- `src/evaluation/semantic_eval.py`
- `tests/test_semantic_eval.py`
- `experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py`

The pre-existing staged rename
`src/deepfake_detection/__init__.py -> src/__package_init_old.py` was excluded
from the path-limited commit and remains staged as user state.

## TDD and verification evidence

RED was captured before production code existed:

```text
pytest -q tests/test_semantic_eval.py
ERROR tests/test_semantic_eval.py
ModuleNotFoundError: No module named 'evaluation.semantic_eval'
```

Additional focused RED cases were captured for T4 pair combination, canonical
`ap` naming, strict DFD task acceptance, per-sample batch context, and
conflicting video labels before each corresponding implementation change.

Final fresh verification:

```text
pytest -q tests/test_semantic_eval.py tests/test_sample_eval.py tests/test_video_aggregation.py
26 passed, 10 warnings in 4.10s

pytest -q
174 passed, 12 warnings in 25.40s

PYTHONDONTWRITEBYTECODE=1 python experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py --help
exit 0; config/checkpoint/seed/domains/conditions/batch-size/output-dir shown

git diff --cached --check -- <the three Task 5 paths>
exit 0
```

Warnings were existing third-party deprecations (`clip`/`pkg_resources`, AMP,
and Jupyter paths); there were no failures.

## Implemented contracts

- Candidate probabilities are scattered into the exact seven-column order
  defined by `SEMANTIC_TO_ID`; invalid valid-position IDs and duplicate valid
  semantic IDs are rejected.
- Canonical fake probability is T1/T3 `p_fake`, polarity-aware T2, paired T4
  `max(p_yes_swap, p_yes_reenact)`, and T5 `1 - p_camera`.
- T4 rows retain question ID, question semantic, target semantic, ordered
  candidate semantics, render ID, and an exact seed/domain/condition/sample/
  video pairing key. Pair attachment requires aligned seed, domain, condition,
  sample, method, video ID, and video key.
- DFD accepts task IDs T1-T3 only. Fixed-H1 is represented as
  `condition=fixed_h1, task_id=T1`, so it remains valid without introducing a
  fourth DFD task ID.
- Video aggregation validates `video_key == method_id::video_id`, rejects
  duplicate sample IDs inside one semantic condition, rejects conflicting
  labels for a semantic video, keeps methods separate, and averages canonical
  frame probabilities.
- Metrics include task counts and overall/per-method slices: T1/T3/Fixed-H1
  AUROC, AP, Brier, accuracy, and 15-bin H1-compatible ECE; T2/T4 balanced
  accuracy and macro F1; T5 macro balanced accuracy and macro F1.
- The CLI supports Task 5 conditions only (`fixed_h1`, `seen`, `heldout`) via a
  condition registry intended for Task 6 extension. It loads an explicitly
  supplied checkpoint and never performs model selection or writes training
  state.

## Output schema

Paths are preserved as:

```text
<output-dir>/<condition>/<domain>_frames.npz
<output-dir>/<condition>/<domain>_videos.csv
<output-dir>/<condition>/<domain>_metrics.json
```

Frame NPZ arrays include:

```text
seed, domain, condition, task_id, variant_id, question_id,
question_yes_means, candidate_order, sample_id, image_path, method_id,
video_id, video_key, label, method_family, target_semantic, render_id,
pairing_key, p_real, p_fake, p_no, p_yes, p_camera, p_swap, p_reenact,
prob_fake
```

Video CSV rows retain the semantic condition, method/video/pairing keys, label,
family, target semantic, all canonical probabilities, canonical fake
probability, and `num_frames`.

Metrics JSON is a normalized list of records with:

```text
seed, domain, condition, task_id, metric, value, count,
method_id, variant_id, question_yes_means
```

Non-finite per-method metrics (for example AUROC on a one-class method slice)
are serialized as JSON `null`.

## DDP deduplication and leakage audit

- Gathered rows are keyed by the full semantic condition plus `sample_id`.
  Byte-equivalent semantic results caused by `DistributedSampler` padding keep
  the first row; a conflicting duplicate raises instead of being hidden.
- Aggregation performs an independent strict duplicate check, preventing
  padding or accidental repeats from weighting video metrics.
- Semantic conditions include task, template/question/polarity, and candidate
  order, allowing task/template/permutation variants to coexist without
  overwriting one another.
- Text lookups are built lazily and split-scoped: Fixed-H1 encodes `fixed`,
  seen encodes `train`, and held-out encodes `heldout` only.
- Held-out and future control condition names are gated by exact equality
  between the current language-manifest SHA256 and checkpoint
  `language_manifest_sha256` before text encoding or loader construction.
- The evaluator does not read checkpoint validation loss for selection and
  writes evaluation artifacts only.

## Remaining operational concern

The unit/integration suite and CLI import/help path were exercised, but a live
four-rank GPU run against `/Dataset/deepfake_detection/DF40_all` and an actual
E2 checkpoint was not run as part of this implementation task. The exact
multi-rank artifact write should therefore be included in the experiment smoke
run before final evaluation.

---

## Review fix round 1

Commit: `6fb7221088d3504cad9401fb504bb071fa3b4f13`
(`fix: correct E2 semantic video evaluation`). The path-limited commit contains
only the same three authorized Task 5 files. The pre-existing staged RD rename
remains staged and was not included or unstaged.

All four Important review findings were fixed under RED-to-GREEN tests:

1. `candidate_order` remains in frame/control metadata and distributed
   deduplication keys, but was removed from ordinary video `GROUP_KEYS` and
   video output. Frame-level candidate permutations now contribute to one
   semantic video variant and its mean/count.
2. The Task 5 evaluator now wraps the existing Task 2 suite with
   `PairedT4EvaluationDataset`. For every T4 physical frame it forces the one
   split-appropriate swap question and the one split-appropriate reenact
   question, retaining distinct question IDs and a shared candidate-set-scoped
   `question_pair_id`. Aggregation accepts exactly one of each aligned question
   and rejects incomplete, duplicate, or ambiguous pairs rather than emitting
   a NaN canonical probability. No Task 2 file changed.
3. T2/T4 F1 now uses binary positive-class F1. Macro F1 remains exclusive to
   T5.
4. Label consistency is checked by seed/domain/condition plus
   method/video/video-key, independent of task, question, variant, and
   candidate order, so cross-variant physical-video conflicts cannot escape.

The initial focused RED run reported six expected failures and 17 passes. The
final fresh verification was:

```text
pytest -q tests/test_semantic_eval.py tests/test_sample_eval.py tests/test_video_aggregation.py
32 passed, 10 warnings in 4.09s

pytest -q
180 passed, 12 warnings in 25.72s

PYTHONDONTWRITEBYTECODE=1 python experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py --help
exit 0

git diff --check -- <the three Task 5 paths>
exit 0
```

Warnings remain the previously reported third-party deprecations. The same
operational concern remains: a live four-rank GPU/dataset/checkpoint smoke run
was not performed during this code-only review fix.
