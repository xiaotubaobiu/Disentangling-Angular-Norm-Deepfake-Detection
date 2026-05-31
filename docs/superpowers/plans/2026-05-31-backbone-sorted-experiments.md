# Backbone-Sorted Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 30 timestamped experiment directories under `experiments/final_results/checkpoints/` grouped by backbone, training mode, and seed.

**Architecture:** Infer backbone, train mode, and seed from each experiment config, then move the whole timestamp directory into a deterministic grouped path. Update result documentation and verify counts.

**Tech Stack:** Python filesystem operations, Markdown documentation, pytest.

---

### Task 1: Move Experiment Directories

**Files:**
- Move: `experiments/2026*/`
- Move: `experiments/configs/`

- [ ] **Step 1: Move timestamped experiments by metadata**

Run a Python script that reads `config/config.yaml`, computes `<backbone>/<train_mode>/seed_<seed>__<timestamp>`, and moves each timestamped directory to `experiments/final_results/checkpoints/`.

- [ ] **Step 2: Move reusable configs**

Move `experiments/configs/` to `experiments/final_results/configs/` if it exists.

### Task 2: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `ARTIFACTS.md`
- Modify: `experiments/final_results/README.md`

- [ ] **Step 1: Replace timestamp list with grouped final-results location**

Document that checkpoints now live under `experiments/final_results/checkpoints/<backbone>/<train_mode>/seed_<seed>__<timestamp>/`.

### Task 3: Verify

**Files:**
- Read: `experiments/final_results/checkpoints/**/best_model.pth`
- Read: `experiments/final_results/*.csv`

- [ ] **Step 1: Verify checkpoint and result counts**

Run assertions for 30 checkpoints, 0 top-level timestamp directories, 10 checkpoints per backbone, 5 per train mode, and final CSV row counts of 60, 45, and 9.

- [ ] **Step 2: Run tests**

Run `python -m pytest` and confirm all tests pass.
