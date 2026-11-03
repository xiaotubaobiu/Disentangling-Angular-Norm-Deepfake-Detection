# Parallel validation in pending strict summaries

The authorized experiment requires concurrent CPU/GPU execution without changing
the frozen scientific protocol. Both strict summary modules accept each seed in
sequence before reading their tables and repeat acceptance before committing.
An optional runtime entrypoint will schedule those unchanged acceptance calls in
up to three spawned CPU processes. The original collection, cross-seed checks,
statistics, locks, completion records and final revalidation remain in place.

This is an implementation choice within the user's existing throughput
authorization. No new experiment, data exclusion or model-selection rule is
introduced. Already running workers retain their original entrypoints.

Alternatives: retain serial validation (kept for active workers); threads (not
chosen because numerical libraries can have shared precision/thread state);
spawned processes (chosen to isolate state while preserving every validator).
Only acceptance is parallelized: splitting table collection would require
duplicating its cross-seed consistency checks and is outside this change.

The wrapper pins its own file hash and the original summary module hash, hides
CUDA, and bounds each child to the existing CPU settings. A new acceptance round
starts only when the original summary requests the next round. Results from the
initial round cannot be reused for final validation. Validator exceptions fail
the summary; no failed seed is omitted or retried. Pools close when the wrapper
exits. A qualification run may only read a previously completed strict summary
and must return its exact existing completion record without changing artifacts.

Deployment requires focused tests, the full CPU suite, real completed-summary
qualification and independent review. Only the not-yet-started EfficientNet D
and CLIP H summary specs may change. Their data/config dependencies, original
scientific code, output paths and seed sets must remain byte-identical. Updating
the immutable coordinator registry requires a recorded supervisor handoff that
leaves all live scientific workers running. If either target has already
started, leave it unchanged.
