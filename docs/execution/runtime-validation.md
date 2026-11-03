# Runtime input validation

Frozen D inputs can live on NTFS/FUSE. Serial file-attribute requests can hold an
otherwise idle GPU for a substantial part of cache extraction. The optional
`scripts/run_parallel_d_validation.py` entrypoint schedules the existing input
validator across 32 threads. Exact equal image paths stay in the same shard, so
conflicting duplicate metadata is still rejected. Every shard invokes the
unchanged original validator; no input check or hash rule is removed.

The entrypoint supports `journal_detector.cache_d`,
`journal_detector.analysis_d` and `journal_detector.summary_d`.
It calls their existing public entrypoints, preserving frozen config, model,
precision, frame order and scoring behavior. Existing committed cache chunks are
verified and resumed by the original cache implementation.

Execution specs pass the SHA256 of the runtime script as its first argument,
followed by the module name and its normal arguments. The script verifies its own
hash and logs the runtime settings, original validator file hash and config hash.
For example, use a registered config and run ID in place of the placeholders:

```bash
PYTHONPATH=src python scripts/run_parallel_d_validation.py \
  d0b58294028860a5897cde8475451c9ba5304e0c2430d4977a97ed30e2d9dcf0 \
  journal_detector.cache_d --config CONFIG --run-id RUN_ID
```

`scripts/run_bounded_cache.py` separately provides the cache runtime thread
bounds for either D or H: main Torch 4, inter-op 1, OpenCV 1, and at most six
cache loader processes. It uses the same self-hash/module/arguments convention.
Its recorded SHA256 is
`7ca69e6a141ca170d070dabcbf16726a8186ec074631806bd0c7d030c55cae4e`.

Qualification compared original and parallel validation, checked every unique
path exactly once, and exercised duplicate conflicts, missing inputs, changed
bytes and changed-stat/same-byte rehashing. A source-only 512-path probe measured
3.09 s serial and 0.42 s parallel in the execution environment. These measurements
describe metadata validation, not whole-pipeline training or inference speedup.

## Strict-summary acceptance rounds

`scripts/run_parallel_summary_validation.py` is an optional CPU entrypoint for
the original D/H strict summaries. It schedules each original per-seed `_accept`
call in up to three spawned processes. Initial acceptance and final revalidation
are separate fresh rounds. Exceptions propagate without omitting or retrying a
seed. The original collection, cross-seed consistency checks, statistics,
locking and completion writes stay in the original summary modules.

Pass the wrapper hash and the original summary module hash from the checked
code snapshot. The entrypoint and its children hide CUDA; scientific children
use Torch4/inter-op1 and OpenCV1. D children retain the existing32-thread input
validator scheduling. For example:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES= python scripts/run_parallel_summary_validation.py \
  SCRIPT_SHA journal_detector.summary_d --summary-sha ORIGINAL_SUMMARY_SHA \
  --config CONFIG --backbone efficientnet_b0
```

Before deployment, use `--qualify --qualification-report REPORT.json` against an
already completed strict summary. Qualification runs its full original readback,
including both acceptance rounds, and requires the returned record, completion
hash and every saved summary artifact hash to remain identical. It performs no
training or inference. Qualification does not substitute for the original
validators or authorize changing an active worker. Runtime scheduling changes
to pending specs require a registry/supervisor handoff with all existing
scientific process handles preserved.
