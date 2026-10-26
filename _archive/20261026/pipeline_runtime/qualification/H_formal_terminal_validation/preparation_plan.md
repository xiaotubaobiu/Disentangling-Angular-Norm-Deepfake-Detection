# H formal terminal CPU auditor implementation plan

**Goal:** Prepare a reusable, read-only entry point for sealed E30 H formal AMP B/AB subsets. Root alone executes real audits.
**Architecture:** Pin and import the frozen health auditor (c59bd151...), call the original worker_h completion verifier for every selected run before checkpoint inspection, then add terminal weight identity, source DEV bisect AUC, and complete paired-stream checks.
**Scope:** New qualification script and isolated validation files only. No V/config/job changes, production locks, CUDA initialization, FP32, cache, target inference, or real audit during preparation.

- [x] Write failing synthetic protocol tests for arguments, formal/AMP/E30 paired selection, incomplete seal/artifact inventory, report safety, terminal weights, source DEV/ties/mismatches, and pair skip positions.
- [x] Run tests RED because the new entry point is absent; retain log.
- [x] Implement audit_H_formal_terminal.py with repeated --run-id and required --report, fixed R root, physical recipe linkage, original verifier before all audits, and exclusive report creation.
- [x] Run isolated tests plus all 16 frozen health protocol tests; compile new source without writing bytecode.
- [x] Review original verifier/trainer contracts and read-only boundaries, record SHA and limitations in preparation_acceptance.json, freeze delivery.

**Review focus:** stale or incomplete seals; source recipe bytes changed; wrong/partial B/AB subsets; final/source-best identity or tensor differences; source DEV cohort/score numeric differences. Any mismatch stops without repair. An AMP skip-stream difference is recorded for the later all-five-pair conditional protocol and never dispatches work.
