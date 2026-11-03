# Parallel strict-summary validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shorten pending strict summaries by concurrently invoking their unchanged per-seed acceptors.

**Architecture:** A runtime-only process pool supplies each original acceptance round. Original collection, cross-seed consistency, statistics, writes and final revalidation stay unchanged.

**Tech Stack:** Python concurrent.futures, multiprocessing spawn; existing locked scientific environment.

**Spec:** `docs/superpowers/specs/2026-11-03-parallel-summary-validation-design.md`.

## Global Constraints

- Physical GPUs0–3 only; this entrypoint and its children hide CUDA.
- Frozen D/H scientific modules, data, config, model checkpoints and statistics remain unchanged.
- At most three validation children; no active worker is restarted or migrated.
- Never reuse initial validation results for the final acceptance round.
- Keep original table collection and all cross-seed tests in the original modules.

## Review Focus

- Changed source inputs between acceptance rounds must be read again.
- A validator exception must prevent completion, without dropping/retrying a seed.
- Spawned children must call the unpatched original acceptor and remain CPU-only.
- Unexpected argument/seed/order changes must fail instead of serving stale results.
- Coordinator registry handoff must preserve all existing scientific worker handles.

### Task 1: Round-based process-pool adapter and CLI

Files: create `scripts/run_parallel_summary_validation.py` and `tests/test_parallel_summary_validation.py`.

Interfaces: `AcceptanceRounds(module_name, arguments_by_seed, workers=3)` is a
context manager; `.accept(*args)` returns the original acceptor result. CLI uses
self-hash/module/config/backbone and original-module-hash arguments.

- [x] Write tests for fresh rounds, real child execution, exception propagation,
  repeated/unknown arguments, worker bounds, CLI self-hash and original-module hash.
- [x] Run the focused file and observe failures from the absent implementation.
- [x] Implement only the round adapter and original summary invocation.
- [x] Focused tests pass, then full `pytest tests -q` passes.
- [ ] Verify frozen scientific hashes before and after; ledger evidence.

### Task 2: Qualification, review and pending-only deployment

Files: document runtime entrypoint in `docs/execution/runtime-validation.md`;
qualification receipts and deployment records stay in the artifact root.

Interfaces: `--qualify` requires an existing strict completion record, performs
the normal original readback with parallel acceptance and verifies every saved
summary artifact and completion hash remain unchanged.

- [ ] Qualify on completed H EfficientNet and completed D DINOv2 summaries.
- [ ] Compare returned complete records and before/after artifact hashes exactly.
- [ ] Obtain one fresh review of the complete change; fix important findings in one pass with regression tests.
- [ ] Only after all gates pass, commit/archive the wrapper and update the two
  not-started specs through a recorded coordinator-only handoff.
- [ ] Verify live scientific process PID/start handles survive the handoff and
  original scientific/config/checkpoint hashes remain unchanged.
