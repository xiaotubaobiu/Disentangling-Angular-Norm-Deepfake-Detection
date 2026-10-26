# SDD ledger — plan: docs/superpowers/plans/2026-09-21-hypothesis-validation.md

Superseded by the user-approved Visual-Jev H1 protocol. See
`docs/superpowers/plans/2026-09-21-visual-jev-h1-implementation.md` and
`experiments/2026-09-21_visual_jev_h1/reports/h1_validation_result.md`.

Setup: executing inline in the existing checkout on branch `study/pooling-saturation`.
Ruling: work in place instead of creating a linked worktree — the validation requires 8+ GB of ignored checkpoints/parquets and a large uncommitted repository reorganization that a new worktree would not contain — cost if wrong: new report files share a dirty checkout, mitigated by writing fresh experimental outputs only to a new phase directory and `/tmp`.
Ruling: replace the original no-CUDA limitation with a GPU 0–3 replay — the user supplied authoritative host `nvidia-smi` output showing four idle V100 32GB devices while GPUs 4–7 are occupied — cost if wrong: a replay launch may fail, but no retained artifact will be overwritten.
Ruling: the installed `executing-plans` package has no `task-start` or `task-done` scripts despite its text requiring them — use `task-brief` plus explicit commands and record results manually — cost if wrong: ledger formatting is manual, but command outputs remain captured in this session.

Pre-flight Task 1 → Task 5: checkpoint/data mapping and coverage are report inputs; compatible.
Pre-flight Task 2 → Tasks 3/4: aggregation, pairing, leakage, and bootstrap gates govern replay interpretation; compatible.
Pre-flight Task 3 → Task 5: independently recomputed statistics are verdict inputs; compatible.
Pre-flight Task 4 → Task 5: representative fresh inference bounds what may be claimed from retained full-score artifacts; compatible.

Task 1: Ruling: interpret “model A” as the 15 raw-trained checkpoints for the feature-geometry hypothesis, but do not equate them with the proposed adaptive-gamma model — all 30 available configs are plain raw/normtrain CLIP, DINOv2, or EfficientNet and no adaptive-gamma implementation/checkpoint is present — cost if wrong: a separately stored trained adaptive model must be added before H6 can be decided.
Task 1: Ruling: DuckDB 1.5.5 reserves `rows` in the attempted select alias; minimal reproduction `select 1 rows` failed while `select 1 as n_rows` passed, so coverage queries use `n_rows` — cost if wrong: none to project code; this only corrects the audit command.
Task 1: complete (documentation-only audit; checkpoint/meta 30/30; input audit 34/34 with identical pre/post SHA-256; parquet 39,830,640 rows, 30 checkpoints, 3 domains; adaptive checkpoint absent).
