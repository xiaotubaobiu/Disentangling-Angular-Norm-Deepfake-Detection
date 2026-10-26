# D accepted result figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. The independent figure implementation runs alongside the controller's original D/H supervision; it cannot dispatch scientific jobs.

**Goal:** Deliver the frozen score-ECDF/calibration-variation figures and concatenate accepted per-backbone tables, preserving all seeds, NA and source/test boundaries.
**Architecture:** One CPU-only operational exporter extends the established R/qualification figure structure. It consumes accepted summary packets and saved natural-video/FP64 operating scores; never recalculates models, thresholds or scientific statistics. Product Git and running science code remain unchanged; exporter code hashes travel with outputs.
**Tech Stack:** Existing Python/NumPy/Matplotlib; English plotted labels, Chinese+English captions; PNG+SVG+exact plotted data JSON.
**Spec:** /home/z/obsidian/paper/deepfake-detector/新实验设计/20261023/新实验计划_本机适配版.md sections8–9 (SHA b018b01915a62c645fc3718134970cd49dc0651ecfd02792220f3a0086c6594d); existing 最终论文表格与图件_数据回填契约_中英.md under论文方法与设置准备_v2/20261024_冻结数据版. Read both relevant sections.

## Global Constraints
- Only R=/home/z/workspace/deepfake_pipeline_reliability_v2_20261023; science V=/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2 stays read-only.
- No GPU, training, cache generation, threshold/model/seed/test changes, cold relocation, automated dispatcher or main manuscript edits.
- Fixed backbones clip_vit_b16,dinov2_vit_b14,efficientnet_b0; five seeds7,42,123,2026,3407 per backbone;200CAL/2000TEST remain distinct.
- All3 real five-seed summary completions AND independent audits must pass and bind the exact current source hashes before production plotting. Tests/fixture renders do not satisfy this gate.
- No overwriting old figures or accepted result packets. Existing3 preparation figures remain unchanged.
- Source-best HOLD; no candidate execution or budget credit.

## Review Focus
1. Missing/NA seed cannot silently become a4-seed mean or dropped box; fail incomplete input, preserve legitimate NA reasons.
2. Natural-video correct grouping only for ECDF; saved FP64 score domain explicitly labelled; no mixing CAL with TEST or source_test with target_full.
3. All4 CAL budgets,200 paired/nested repeats, fixed TEST hash and distinct subset counts; n80 may contain only1 unique calibration set, never200 independent models.
4. Training-seed descriptive summaries, conditional CAL spread and TEST bootstrap CIs stay separate; no pooled15-model estimator or new inference.
5. Mutable/stale hash, symlink/path escape or existing output must reject before production writes; input hashes before/after.

### Task1: Input gates and exact plotted data (TDD)
Files: create qualification/export_D_result_figures.py and qualification/D_result_figures_validation/test_result_figures.py; ledger in same validation directory/progress.md.
Interfaces: ecdf_coordinates(values)->(unique sorted x,right-continuous cumulative probabilities); calibration data retains backbone,seed,representation,budget,repeat,3 operating metrics and unique calibration set identity; build_packet(root) validates accepted data and produces JSON/CSV-ready data without plotting.
- [ ] Write/run failing behavioral tests: ties[2,1,2] ->x[1,2],cdf[1/3,1]; no role/class/seed pooling; duplicate natural uid rejected; finite-score requirement; incomplete backbone/seed gate; tampered completion/source hash rejection.
- [ ] Implement minimum using real saved CSV structure. Exact tables: analysis/D/RUN/{source_test,target_full}/video_scores.csv has role,label,video_uid,grouping,representation,aggregation,score,rank_score_decimal; use score for displayed FP64 mean-logit margin and label it, not a new ranking metric. Select grouping=correct and raw/l2, mean_logit, separately byrole/label/seed.
- [ ] CAL tests: missing repeat rejected, valid budget NA retained, common IDs between raw/l2 and nested sets checked; unique-set count recomputed rather than trusting label; a repeated full80 pool reports1 distinct set; fixed test_id_hash required.
- [ ] Tests: independent hand-checkable fixtures, no model mocks; project operational plotting tests run fully, science suite needn't repeat if V untouched (user scope and check discipline).

### Task2: Safe export and rendering (TDD)
- [ ] Add tests for no-overwrite, safe output paths, unchanged inputs and exact preservation of per-backbone/seed table rows/NA.
- [ ] Two ECDF figures separately for source_test and target_full; backbone rows×raw/l2 columns; real/fake hues, five seed line styles. Full exact coordinates in JSON, no score-based example selection or CAL samples.
- [ ] Calibration variation covers budgets10/20/40/80 and FPR/TPR/pooledBA, separately for each backbone/model/representation. Suggested3 figures (one/backbone), rowsbudgets×colsmetrics, paired raw/l2 boxes over200CAL with five distinct training-seed xpositions. Boxes show CAL quartiles/range (explicitly NOT TEST CI or across-model SD); annotate actual distinct set counts/NA including full80 count1. All saved table rows retained; no source-only performance hidden (source/default remain in complete tables/captions scope).
- [ ] Concatenate12 accepted summary tables across3backbones into a cross-backbone packet without averaging acrossbackbones; preserve backbone/seed identity and original condition/CI references.
- [ ] Production output R/manuscript_figures/D_accepted_results; never overwrite. SHA-bound complete receipt, captions_cn_en.md, plotting data JSON and SVG+PNG. No production launch until controller confirms all3 real independent summary acceptances.
- [ ] Render synthetic fixtures only in validation-owned temp paths, read PNGs for layout if useful; mark synthetic/test clearly.
- [ ] One independent whole exporter review at end; address important findings with regression tests. Root performs production admission/export/render+independent data QA when inputs ready.

## Execution rulings and ledger
Ruling: This is implementation of frozen user-authorized figures, not a new experiment or a request for new design approval. User explicitly authorized direct safe plan execution. Layout details above are executor choices and can be adjusted for actual readability without changing data or inference.
Ruling: Keep new operational plotting code under existing R/qualification, isolated from frozen science repo so current qualification/code hashes do not change; code SHA and tests provide reproducibility. No Git commit to V or deletion of workspaces/evidence.
Pre-flight: figures consume accepted3×5 summary packets; controller owns individual analysis/summary launch/HANDOFF. Implementer owns only figure code/tests/own implementation report. No shared mutable source artifacts.
