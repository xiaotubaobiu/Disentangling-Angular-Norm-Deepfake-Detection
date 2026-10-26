# Stage-2 layout migration — 2026-07-30

把 `2026-07-21_frame_midterm_diagnosis` 恢复成唯一规范布局。全部用同文件系统 `mv`（`/dev/mapper/ubuntu--vg-ubuntu--lv`），**1.4 GB parquet 没有被复制**，也没有产生第二份数据源。

## 旧 → 新

| 旧路径 | 新路径 | 方式 |
|---|---|---|
| `<phase>/parallel_20260721_213914/` | `<phase>/results/parallel_20260721_213914/` | `mv`（目录整体移动，19 个文件） |
| `<phase>/results/figures/` (9 PNG) | `<run>/figures/` | `mv` |
| `<phase>/results/reports/provenance.md` | `<run>/reports/provenance.md` | `mv` |
| `<run>/parallel.log`、`parallel2.log`、`worker_gpu0-5.log` | `<run>/logs/` | `mv`（8 个） |

`<run>` = `experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/`

迁移后：run 根 11 个文件（2 parquet + 8 CSV + 1 JSON）、`logs/` 9 个、`figures/` 9 个、`reports/` 1 个，合计 30 个文件。阶段根只剩 `README.md`、`scripts/`、`results/`——不存在第二个「看起来像真实来源」的目录（`tests/test_stage2_layout.py::test_single_run_directory_under_results` 会持续守住这一点）。

## 校验值（迁移前 → 迁移后，逐一相同）

| 文件 | md5 | 大小 |
|---|---|---|
| `per_frame_scores.parquet`（前 64 MiB） | `1519e302a5205a7facbf17c6ad5db621` | 1,381,098,121 B |
| `per_video_scores.parquet` | — | 33,969,324 B |
| `aggregate_metrics.csv` | `d969de174638ef0a3355a918c6af7b57` | |
| `bootstrap_direction_vs_raw.csv` | `fdabc17b498b9c5e750b8fa754b56ecc` | |
| `cdf_balanced_delta_auc.csv` | `8c0f0202577fe9fd122aa98e629aa846` | |
| `cdf_delta_auc_per_method.csv` | `ddf639537714f4333e14f5860443d770` | |
| `cdf_delta_auc_pooled.csv` | `3c1b0a931d213743d0a38ad19087e561` | |
| `delta_auc_per_method_cdf.csv` | `e8baf6550c13d8740a6a672a56472340` | |
| `norm_quartile_buckets.csv` | `fd932bcf8e0a41710ff3c10f36dcf7a3` | |
| `per_method_metrics.csv` | `c0ae8acc4b8bd3199d7719903cd37840` | |
| `hypothesis_summary.json` | `dae52356328e5ebef6ead87c7aed0b98` | |
| `reports/provenance.md`（迁移瞬间） | `fd08127f5cc00e08bd9bde9492738bdb` | 之后按新路径更新了正文 |

parquet 迁移后复读：`per_frame_scores.parquet` = 39,830,640 行 / 30 个 checkpoint，与 `logs/parallel2.log` 记录一致。

## 一并修掉的引用

| 文件 | 改动 |
|---|---|
| `scripts/run_diag_parallel.sh` | launcher/worker 日志与 pid 文件改写到 `$RUN_DIR/logs/`；启动时预建 `logs/`、`figures/` |
| `scripts/wait_and_verify.sh` | 完成校验报告改写到 `$RUN_DIR/logs/_completion_report.txt` |
| `scripts/wait_and_run_diag.sh` | 注释说明监视日志留在 `results/_wait_monitor.log`（此时还没有 run 目录） |
| `README.md` | 目录树补 `logs/`、`reports/`；明确「唯一 run 目录 = 唯一真实来源」；provenance 链接改到 `<run>/reports/` |
| `<run>/reports/provenance.md` | 所有日志证据路径加 `logs/` 前缀；新增 §4.6 记录 07-29 的删除与漂移；顶部注明布局历史 |
| `experiments/README.md` | 阶段 2 说明与 provenance 位置同步 |

阶段 3/4 的 YAML **没有改动**——它们本来就写的是 `results/parallel_20260721_213914/...`，迁移后重新解析成功（见下）。历史设计文档（`experiments/*/docs/*.md`）里 `outputs/frame_midt_diag/...` 的旧路径按既有约定不改写，映射表在 `experiments/README.md`。

## 一并恢复的丢失文件

`<run>/_completion_report.txt`（1519 B，mtime 07-22 01:26）在 2026-07-29 被删除。正文从 07-28 的会话记录恢复为 `<run>/logs/_completion_report.recovered_20260730.txt`，**用不同文件名**表明无法证明逐字节一致；内容与 `logs/parallel.log`（aggregate rc=1 @01:25）互相印证。原件之外的 28 个文件都是原件。

## 验证

`tests/test_stage2_layout.py`（新增，7 passed / 1 skipped）：

| 测试 | 断言 |
|---|---|
| `test_single_run_directory_under_results` | `results/` 下只有一个 run 目录；阶段根没有 `parallel_*`、`figures/`、`reports/` 残留 |
| `test_run_directory_subdirs` | `figures/`、`logs/`（含 `*.log`）、`reports/provenance.md` 都在 run 目录内 |
| `test_frame_parquet_readable` | duckdb 读出 39,830,640 行 / 30 checkpoint |
| `test_downstream_configs_resolve` | 阶段 3 (`parquet.frame`/`parquet.video`/`ckpt_root`)、阶段 4 (`parquet_frame`/`ckpt_root`)、阈值阶段 (`inputs.frame_scores`/`inputs.checkpoint_root`) 逐个 `exists()` |
| `test_balanced_helper_writes_into_run_dir_figures` | 合成 run 目录跑 `cdf_balanced_dir_raw.main()`，4 张热图落在 `<run>/figures/`、CSV 落在 run 根 |
| `test_native_helper_writes_into_run_dir` | 同上，`cdf_native_dir_raw.main()` 两张 CSV 落在 run 根 |

run 目录是 git-ignored 的本地产物，所以这些测试在没有产物的机器上自动 skip，不会误报。
