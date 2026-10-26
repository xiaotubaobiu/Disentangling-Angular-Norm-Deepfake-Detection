# Task 2 report: deterministic E2 semantic sampler

## Files changed

- `src/data/semantic_dataset.py` (new): stable task/template hashing, method-family truth table, fixed-width candidate packing, semantic rendering, and train/suite dataset wrapper.
- `src/data/builders.py`: extracted the existing H1 dataset construction into `build_train_dataset`, `build_val_dataset`, and `build_eval_dataset`; legacy loaders retain their prior sampler, shuffling, worker seeding, pinning, and drop behavior; added semantic train/validation/evaluation loaders.
- `tests/test_semantic_dataset.py` (new): truth-table, polarity, padding, permutation, hash determinism, worker determinism, batch interface, unknown-family, DFD exclusion, and validation tests.

## RED evidence

Command:

```text
pytest -q tests/test_semantic_dataset.py
```

Exact result: exit code 2; collection stopped with `ModuleNotFoundError: No module named 'data.semantic_dataset'`; `1 error in 1.27s`.

## GREEN evidence

Focused command:

```text
pytest -q tests/test_semantic_dataset.py
```

Exact result: exit code 0; `11 passed in 2.15s`.

Focused plus relevant legacy command:

```text
pytest -q tests/test_semantic_dataset.py tests/test_sample_eval.py tests/test_video_aggregation.py
```

Exact result: exit code 0; `20 passed in 2.72s`.

Compilation check:

```text
python -m compileall -q src/data/semantic_dataset.py src/data/builders.py
```

Exact result: exit code 0 with no output.

## Full suite

Command:

```text
pytest -q
```

Exact result: exit code 0; `125 passed, 11 warnings in 24.80s`. Warnings were dependency deprecations from `clip`, `pkg_resources`, and Jupyter path migration.

Fresh post-commit verification: exit code 0; `125 passed, 11 warnings in 25.01s`.

## Self-review

- `task_id` remains the canonical `T1` through `T5` string and `task_index` is a scalar `torch.long`, so default collation yields `[B]` for both interfaces in their respective representations.
- Question, candidate, target, and yes-means numeric fields are scalar/fixed-width tensors; binary candidates use `-1` padding with a false mask, and target indices are computed only after moving text and semantic IDs together.
- Task choice and task-internal choice use separate BLAKE2b-derived seeds and never depend on worker-local RNG state.
- Unknown fake families remain valid for T1 through T3, raise for direct T4/T5 rendering, and the DFD evaluation loader advertises only T1 through T3.
- Existing H1 record selection and `FrameClassificationDataset` construction were moved without changing their logic; legacy loader configuration was left unchanged.
- Staging was restricted to the three task-owned files. The index already contained a user-owned `src/deepfake_detection/__init__.py` to `src/__package_init_old.py` rename before Task 2 began; it is excluded from the Task 2 commit with an explicit commit pathspec and its existing staged state is preserved. This report is intentionally not staged under the controller's commit restriction.

## Concerns

No implementation blocker or known correctness concern. The full suite emits 11 pre-existing third-party deprecation warnings.

## Commit

`a385cbe16befe50ddb487b36b262803d902b1cfa` (`feat: add deterministic E2 semantic sampler`)
