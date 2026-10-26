# M source-best five-seed summary Implementation Plan

> For agentic workers: inline execution under explicit delegation; parent owns final independent review. No commit or new agent.

Goal: add tools/m_source_best_five_seed_summary.py and tests/test_m_source_best_five_seed_summary.py only.
Architecture: accepted report/read-only consumer -> fixed core endpoints -> exact5-seed descriptive tables -> bounded append-only publication.
Tech: existing Python/pandas plus stdlib statistics/Fraction,csv,json; no install.
Spec: M_source_best_five_seed_summary_design_20261026.md and frozen b018b019 §9.

Global constraints: keep3backbones separate;all5seeds;source-best prefix disclosure;NA/negative preserved;TEST2000/CAL200 distinct;PREPARE_ONLY default;no core/source/live config edits;no actual production/slot/experiment/weights.
Review focus: acceptedFalse or forged/missing complete;crossjob/backbone/LAST/DEV;initial digest drift/path escape;missing/NA seed selection;ddof1 andlayer confusion;closure-gated CI;publication budget/namespace.

- [x] Task1: write new small synthetic tests for exact endpoints/layers,prefix/acceptance/refs and five-seed math; observe RED for missing module.
- [x] Task2: implement minimal pure extraction/statistics and accepted report loading with pins; run same tests GREEN.
- [x] Task3: add PREPARE_ONLY/runtime admission/append-only bounded publication tests; RED then minimal GREEN.
- [x] Task4: only new targeted tests and CLI prepare verification; bind source/test SHA and report no production. Parent final independent review pending.

Ruling: explicit user/root scope forbids old-suite repetition and commits; run only new synthetic tests,not the old suite. No actual summary data or slot. Streamed immutable SHA rebinding is not source/tensor revalidation. Tests may mock external real slot/resource admission while keeping math/files/digests real.

Fresh verification:72 new synthetic tests passed2.71s, default CLI PREPARE_ONLY exit0. Parent independent review remains pending; no production summary executed.
