# IEEE Access H-DINO10 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace only six H-DINO60 models with fresh fixed10 models, revise the Chinese manuscript, add an English Quarto manuscript and compile an IEEE Access LaTeX delivery with an Overleaf Git tutorial.

**Architecture:** Keep the original execution worktree and all old artifacts immutable. This isolated worktree adds an opt-in fixed-epoch selection policy; a new artifact root copies frozen metadata but reads original image bytes. Final tables combine newly accepted DINO summaries with unchanged CLIP/EfficientNet summaries through explicitly recorded roots. Quarto consumes accepted tables and figures, emits portable IEEE Access TeX and a locally verified PDF.

**Tech Stack:** Existing PyTorch environment, Python/pandas, Quarto, official IEEE Access class, pdfLaTeX/BibTeX.

**Spec:** /home/z/workspace/deepfake_access_20261103/operations/user_spec.json

## Global Constraints

- Only physicalGPU0–3; preserve GPU4–7 and all original/Dataset bytes.
- H-DINO: seeds7/42/123, both B/AB, full10 epochs, LR5e-5, batch128, original augmentation/optimizer/EMA. Best paired sourceDEV-B mean-prob EMA remains the selection rule.
- D15 and H CLIP/EfficientNet unchanged. Retain old60 results in their original archive; replace all primary H-DINO tables/figures/text only after all three pairs accepted.
- Authors are Shiyu Chen and Kangbin Yim, in that order; Yim is corresponding author. Do not invent missing conference/funding/biography details.
- Prepare Overleaf folder and tutorial; no remote writes or credentials requested.

## Review Focus

1. Fixed10 training must finish10 even when source AUC plateaus, retain earlier bestEMA, and recover after interruption; legacy early stopping must remain unchanged.
2. Mixed-root H publication must bind each backbone to its own complete three-seed summary, same frozen cohorts, and reject duplicates/missing seeds.
3. All old H-DINO60 numeric prose, tables and figures must be replaced together; D outputs must remain byte-identical.
4. English citations and claims must match actual accepted sources, especially the revised budget, processing coverage, and bounded rank example.
5. LaTeX folder must compile independently of Quarto/source absolute paths, include class/bibliography/figures, and never publish training artifacts or credentials.

### Task1: Fixed-budget training and registration

- Add failing regression to existing paired-resume test: fixed10 plateau runs10 and chooses epoch1; legacy stops6.
- Add focused selection policy tests for mismatched/nonboolean policy values.
- Implement a small opt-in selection factory in trainer_h.py; defaultpatience5 unchanged, fixed policy patience epochs+1.
- Run focused training/protocol tests. Copy immutable H metadata into the revision root and verify hashes. Freeze user-requested10 budget and exact six-run registry, including post-initial-results revision provenance.
- Launch three paired runs on idle approved GPUs with persistent completion/log handles. Queue six exports and three pair analyses plus strict summary.

### Task2: IEEE Access manuscript tooling during training

- Verify current official IEEE Access template, author requirements, Quarto LaTeX export, Overleaf token/branch documentation.
- Install isolated user-level Quarto/TeX dependencies if absent; record versions and checksums.
- Add a portable Quarto/template/build scaffold and minimal compile fixture; verify source-to-TeX and local LaTeX compilation.

### Task3: Accept revised H results and update documents

- Replay all three training histories/bestEMA identities and accept six feature exports, three analyses, original strict summary plus independent mean/SD reductions.
- Merge the new DINO tables with unchanged other H summaries; recompute all H comparisons, operating points and cross-backbone rank witnesses.
- Regenerate H figures with Source Data and rendered QA. Add the D developer-curve figure.
- Back up existing four Obsidian documents, revise Chinese prose/results/plan/record, and write the English manuscript.qmd from accepted evidence. Audit all numeric claims and old60 references.

### Task4: Portable LaTeX and delivery

- Render Quarto to latex/, verify bibliography/cross-references, compile PDF and inspect all pages.
- Prepare a clean Overleaf import ZIP and commands using a separate document checkout, token authentication, actual remote branch detection and no force-push.
- Run relevant complete tests and one final fresh-context whole-change review; address material findings with regressions.
- Publish code/documents per existing authorized Git workflow, preserve conference branch, and deliver Chinese/English/TeX/PDF/tutorial with remaining author-supplied fields clearly identified.
