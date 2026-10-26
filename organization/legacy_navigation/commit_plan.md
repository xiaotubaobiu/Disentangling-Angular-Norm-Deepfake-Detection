# Commit plan — 2026-07-30

未执行提交。原因不是「怕」，而是**这次任务的文件无法在不夹带无关历史改动的前提下形成可用的提交**：

- 工作区里 95 条 deleted + 12 条 untracked 来自 **2026-07-28 的目录重组**（`src/deepfake_detection/*` → `src/*`、`scripts/*` → 各阶段、`outputs/*` 撤销），那不是本任务的工作，见 `reports/repository_preflight.md`。
- 本任务的新代码（`src/study_threshold/*`）依赖的 `src/study/`、`src/study_pooling/`、`src/config_io.py` 全部仍是 untracked。只提交本任务文件会得到一棵 **import 就会失败的树**。
- 所以顺序必须是：先把重组落成一个 commit（提交者应是做重组的人/得到确认），再叠本任务的三个逻辑提交。

分支：`study/pooling-saturation`，HEAD `ed3af13`。以下命令按顺序执行即可，**不含** `stash` / `reset` / `clean`。

## Commit 0（前置，不属于本任务）— 落定 2026-07-28 的重组

```bash
git add -A src/ experiments/ docs/ configs/ tests/ \
          README.md pyproject.toml .gitignore \
          train.py evaluate.py scripts/ outputs/          # 记录旧路径的删除
git commit -m "refactor: flatten src/ and reorganize experiments into dated phases

- src/deepfake_detection/* -> src/* (no extra package layer); load_config extracted to src/config_io.py
- scripts/*, train.py, evaluate.py, tests/study*, configs/*_config.yaml -> owning experiments/<date>_<phase>/
- outputs/ retired; artifacts live in experiments/<phase>/results|figures
- .gitignore tracks phase code/configs/reports/docs, ignores results/ and figures/
- pyproject: flat package discovery + pytest (pythonpath=src, --import-mode=importlib)"
```

注：`git add -A` 会一并记录 2026-07-29 删除 xception 配置这件事（备份 `.deleted_configs_backup_20260729.tar.gz` 需要一起 add，否则它是游离的未跟踪文件）。

## Commit 1 — chore: restore canonical stage2 artifact layout

```bash
git add experiments/2026-07-21_frame_midterm_diagnosis/README.md \
        experiments/2026-07-21_frame_midterm_diagnosis/scripts/run_diag_parallel.sh \
        experiments/2026-07-21_frame_midterm_diagnosis/scripts/wait_and_run_diag.sh \
        experiments/2026-07-21_frame_midterm_diagnosis/scripts/wait_and_verify.sh \
        experiments/2026-07-21_frame_midterm_diagnosis/scripts/cdf_native_dir_raw.py \
        experiments/README.md tests/test_stage2_layout.py \
        reports/repository_preflight.md reports/stage2_layout_migration.md
git commit -m "chore: restore canonical stage2 artifact layout

Move the 2026-07-21 diagnosis artifacts back to
results/parallel_20260721_213914/ (atomic mv, 1.4GB parquet not copied), with
figures/, logs/ and reports/ inside the run directory. Launcher and verifier now
write logs into <run>/logs/. md5 of every migrated table/JSON unchanged; the
frame parquet still reads 39,830,640 rows / 30 checkpoints.

tests/test_stage2_layout.py pins the layout (single run dir, subdirs, downstream
configs resolve, cdf_* helpers write into <run>/figures)."
```

产物本身（`results/**`）被 `.gitignore` 忽略，提交里只有代码与文档——这正是要的效果。

## Commit 2 — chore: remove stale xception training references

```bash
git add experiments/2026-05-05_paper_feature_geometry/scripts/launch_training.py \
        experiments/2026-05-05_paper_feature_geometry/tests/test_launch_training_backbones.py \
        experiments/2026-05-05_paper_feature_geometry/README.md \
        README.md docs/ARCHITECTURE.md .gitignore
git commit -m "chore: remove stale xception training references

configs/xception_base.yaml was archived on 2026-07-29 but launch_training.py
still hard-referenced it and required --xception-lr. Now: supported training
backbones are dinov2/efficientnet (+ CLIP via configs/clip_base.yaml), learning
rates are validated per requested backbone, and asking for xception exits with
'Xception training configs are archived in
.deleted_configs_backup_20260729.tar.gz and are not maintained in the current
experiment.' src/models/xception.py and its factory entry stay so old
checkpoints remain loadable. README/ARCHITECTURE now separate supported training
backbones from legacy model implementations."
```

## Commit 3 — feat: add cross-domain threshold transfer study

```bash
git add src/study_threshold/ \
        experiments/2026-07-30_threshold_transfer/README.md \
        experiments/2026-07-30_threshold_transfer/configs/ \
        experiments/2026-07-30_threshold_transfer/docs/ \
        experiments/2026-07-30_threshold_transfer/scripts/ \
        experiments/2026-07-30_threshold_transfer/tests/ \
        experiments/2026-07-30_threshold_transfer/reports/ \
        reports/threshold_transfer_final_report.md \
        README.md experiments/README.md docs/ARCHITECTURE.md
git commit -m "feat: add cross-domain threshold transfer study (2026-07-30)

Freeze calibration and threshold selection on FF++ validation, then deploy to
CDF unchanged. Balanced accuracy falls 0.963 -> 0.616 while AUC stays 0.890
(CDF oracle 0.814, transfer regret 0.198, 0/30 cells within 0.05). A 1% source
FPR budget realises 64% FPR on CDF. Temperature scaling cannot help by
construction (p=0.5 <=> s=0 for any T); Platt moves the boundary but not the
regret. Cause: real-video scores shift +4.15 while fake shifts +0.36
(differential shift -3.79) in 25/30 cells.

- src/study_threshold/{io,calibration,thresholds,transfer,score_shift,bootstrap,reporting}.py
  reusing config_io, study.io, study.cluster_bootstrap, study_pooling.{aggregators,temperature}
- 8 stage scripts + run_all.py (dry run 11s, full run 8.5min, CPU only)
- 23 tests incl. leakage guards (AST + monkeypatched target loader), real-video
  dedup, raw/direction parity, bootstrap equivalence with the reused reference"
```

## 提交后自检

```bash
python -m pytest -q                 # 期望 99 passed
git status --porcelain | wc -l      # 期望 0（除本地产物已被忽略）
git log --oneline -4
```

## 不进 git 的东西

`experiments/*/results/`、`experiments/*/figures/`（含 1.4 GB parquet、6.6 GB checkpoint、92 MB CSV、图）、`__pycache__`、`.pytest_cache`。
`.deleted_configs_backup_20260729.tar.gz` 保留在工作区，本任务既不解压也不删除；若要跟踪它，请在 Commit 0 里一并 add。
