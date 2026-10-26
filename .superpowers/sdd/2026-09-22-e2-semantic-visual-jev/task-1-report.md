# Task 1 report: frozen language manifest

## Implementation

- Added the exact frozen five-task language manifest at
  `experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml`.
- Added `data.semantic_language` with frozen text/task records, symbolic-name
  lookup, deterministic IDs sorted by symbolic ID, and strict manifest
  validation.
- Validation rejects duplicate IDs or complete strings, unknown task answer-set
  references, invalid semantic values, non-binary answer sets, non-ternary T5
  candidate sets, malformed control candidate cardinalities, and train/heldout
  string overlap.

## TDD evidence

- RED: `pytest -q tests/test_semantic_language.py` -> collection failed with
  `ModuleNotFoundError: No module named 'data.semantic_language'`.
- GREEN: `pytest -q tests/test_semantic_language.py` -> `3 passed in 0.06s`.
- Manifest parity/API check: exact manifest comparison succeeded; a direct API
  check verified 59 entries, sorted IDs, task lookup, and T2/T5 candidate sets.

## Full test suite

- `pytest -q` -> `113 passed, 11 warnings in 24.28s`.
- Existing third-party deprecation warnings: `pkg_resources` (including
  namespace declarations for mpl_toolkits, ruamel, sphinxcontrib, and zope)
  and Jupyter `platformdirs` migration.

## Files committed

- `experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml`
- `src/data/semantic_language.py`
- `tests/test_semantic_language.py`

## Self-review

- Confirmed only the three Task 1 files are in commit `1b998c5`.
- Confirmed the manifest exactly matches the frozen brief content.
- `TextEntry`, `CandidateSet`, and `TaskLanguage` are frozen dataclasses; text
  IDs are assigned from lexical symbolic-ID ordering.

## Concerns

- PyYAML parses unquoted YAML `yes`/`no` as booleans. The loader deliberately
  normalizes those values back to the required semantic strings; the frozen
  manifest itself remains exactly as specified.

## Round 1 review fix

- Added a regression test that appends a unique third candidate to T1's train
  candidate set and requires loading to fail. RED command:
  `pytest -q tests/test_semantic_language.py` -> `1 failed, 3 passed`; the
  failure was `Failed: DID NOT RAISE <class 'ValueError'>`.
- Added loading-time validation requiring every direct T1/T3 candidate set to
  contain exactly two options with canonical `real`/`fake` semantics. T2/T4
  continue to use the pre-existing shared answer-set validation requiring
  exactly `yes`/`no`.
- GREEN command: `pytest -q tests/test_semantic_language.py` -> `4 passed in
  0.11s`.
- Full suite: `pytest -q` -> `114 passed, 11 warnings in 24.42s`; warning set
  unchanged (third-party `pkg_resources` and Jupyter deprecations).
- Self-review: the regression's third option has a unique symbolic ID and full
  string, so failure is specifically due to candidate cardinality/semantics.
