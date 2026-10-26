# Repository preflight — 2026-07-30

本次维护任务开始前的仓库状态快照。**没有执行任何 `stash` / `reset` / `clean` / 强制覆盖**；用户已有的工作区改动全部保留。

## 当前分支与 HEAD

```text
branch : study/pooling-saturation
HEAD   : ed3af13b4acfbe766f254116d799a1b5ee18e125  2026-07-28 00:46:01 +0900
subject: feat(pooling): temporal consistency + final A/B/C/D report (exp7)
```

`git status --porcelain` 共 111 条：`95` deleted、`12` untracked、`3` modified、`1` renamed-deleted。

## Modified（3）

| 文件 | 说明 |
|---|---|
| `.gitignore` | 2026-07-28 重组时改写：阶段 `scripts/configs/reports/docs` 可跟踪，`experiments/*/results`、`experiments/*/figures` 忽略 |
| `pyproject.toml` | 扁平化后的包发现 + pytest 配置（`pythonpath=src`、`--import-mode=importlib`、`testpaths`） |
| `README.md` | 重写为新布局（src 模块表、4 个阶段表、artifact policy） |

## Deleted（95，全部是 2026-07-28 目录重组产生的「旧路径」）

这些删除**不是丢文件**，而是文件已被 `mv` 到新位置、git 还没记录重命名：

| 旧路径组 | 数量 | 现位置 |
|---|---|---|
| `src/deepfake_detection/**` | 51 | `src/**`（去掉包层，另加 `src/config_io.py`） |
| `scripts/**` | 19 | `experiments/<对应阶段>/scripts/**` |
| `tests/study/**`、`tests/study_pooling/**` | 18 | `experiments/2026-07-23_four_questions/tests/`、`experiments/2026-07-27_pooling_saturation/tests/` |
| `train.py`、`evaluate.py` | 2 | `experiments/2026-05-05_paper_feature_geometry/scripts/` |
| `configs/study_config.yaml`、`configs/pooling_saturation_config.yaml` | 2 | 各自阶段的 `configs/` |
| `configs/xception_base.yaml` | 1 | **2026-07-29 删除**，备份在 `.deleted_configs_backup_20260729.tar.gz` |
| `outputs/*/.gitkeep` | 4 | `outputs/` 目录整体撤销，产物进入各阶段 `results/` |

另有 1 条 `RD src/deepfake_detection/__init__.py -> src/__package_init_old.py`：扁平化中转文件，已在磁盘上删除。

## Untracked（12）

```text
.deleted_configs_backup_20260729.tar.gz   # 2026-07-29 删 xception 配置时的备份（16 个 yaml），本任务不解压、不删除
docs/                                     # ARCHITECTURE.md（历史上被 .gitignore 忽略过）
experiments/                              # 4 个阶段目录全部内容
src/config_io.py  src/data/  src/diagnosis/  src/engine/  src/evaluation/
src/losses/  src/models/  src/study/  src/study_pooling/
```

> 注意：`src/diagnosis/*` 与阶段 2 的脚本**从未进入任何 commit**（`git log --all -- 'src/deepfake_detection/diagnosis/*'` 为空），阶段 2 的结果目前没有 commit 可锚定。

## 已知的漂移（本任务要修的）

1. **阶段 2 布局漂移**：真实数据在 `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/`，但阶段 3/4 的 YAML、阶段 2 的 README/provenance、`cdf_*_dir_raw.py` 用法、流水线默认输出都指向 `results/parallel_20260721_213914/`。图和 provenance 目前散在 `results/figures/`、`results/reports/`，形成两个「看起来都像真实来源」的目录。
2. **Xception 失效引用**：`configs/xception_base.yaml` 与阶段 1 的 xception 配置已于 2026-07-29 删除（有备份），但 `launch_training.py` 仍硬引用该路径且 `--xception-lr` 为必填参数；README / ARCHITECTURE 未区分「当前支持训练的 backbone」与「保留的 legacy 模型实现」。

## 本任务计划修改/新增的文件

### 修（阶段 2 布局）
```text
experiments/2026-07-21_frame_midterm_diagnosis/README.md
experiments/2026-07-21_frame_midterm_diagnosis/scripts/frame_midt_diagnose.py
experiments/2026-07-21_frame_midterm_diagnosis/scripts/run_diag_parallel.sh
experiments/2026-07-21_frame_midterm_diagnosis/scripts/wait_and_run_diag.sh
experiments/2026-07-21_frame_midterm_diagnosis/scripts/wait_and_verify.sh
experiments/2026-07-21_frame_midterm_diagnosis/scripts/cdf_native_dir_raw.py
experiments/2026-07-21_frame_midterm_diagnosis/scripts/cdf_balanced_dir_raw.py
experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/reports/provenance.md   （移动 + 更新）
experiments/README.md
```
移动（同文件系统 `mv`，不复制）：`parallel_20260721_213914/` → `results/`；`results/figures/` → run 目录内；`results/reports/` → run 目录内；run 目录内 `*.log` → `logs/`。

### 修（Xception）
```text
experiments/2026-05-05_paper_feature_geometry/scripts/launch_training.py
experiments/2026-05-05_paper_feature_geometry/README.md
experiments/2026-05-05_paper_feature_geometry/tests/test_launch_training_backbones.py   （新增）
README.md
docs/ARCHITECTURE.md
.gitignore                                  （允许跟踪 tests/）
```

### 新增（阈值迁移实验）
```text
src/study_threshold/{__init__,io,calibration,thresholds,transfer,score_shift,bootstrap,reporting}.py
experiments/2026-07-30_threshold_transfer/
  README.md  configs/threshold_study.yaml  docs/DESIGN.md
  scripts/{00_audit_inputs,01_prepare_ffpp_validation,02_fit_calibrators,03_select_source_thresholds,
           04_evaluate_transfer,05_analyze_score_shift,06_generate_report,run_all}.py
  tests/（10 项）  reports/  results/{tables,val_cache,logs}/  figures/
reports/{repository_preflight,stage2_layout_migration,threshold_transfer_final_report,commit_plan}.md
```

不动的东西：任何历史结果文件的内容与语义、`.deleted_configs_backup_20260729.tar.gz`、其它阶段的 `results/`。
