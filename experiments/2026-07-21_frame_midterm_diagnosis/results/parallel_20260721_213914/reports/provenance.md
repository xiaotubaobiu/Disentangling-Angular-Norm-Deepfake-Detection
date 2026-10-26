# 阶段 2 溯源与复现审查（frame-level midterm diagnosis）

审查日期：2026-07-28。目的：让这一阶段的每个历史产物都能被定位到「哪个脚本、哪条命令、哪次运行」，并把不可复现的部分明确写出来。

运行目录：`results/parallel_20260721_213914/`（下文简称 `<run>`）；本文件位于 `<run>/reports/`，日志在 `<run>/logs/`，图在 `<run>/figures/`。日志里只有 `HH:MM:SS`，日期由文件 mtime 推得。

布局历史：2026-07-29 这批产物曾被移出 `results/`（run 目录被提到阶段根、figures/reports 散在 `results/` 下），2026-07-30 已用原子 `mv` 恢复为唯一规范布局，校验值逐一比对一致 —— 见仓库根 `reports/stage2_layout_migration.md`。

## 1. 时间线（证据：`<run>/logs/parallel.log`、`logs/parallel2.log`、`logs/worker_gpu*.log`、`logs/_completion_report.recovered_20260730.txt`、文件 mtime）

| 时间 | 事件 | 证据 |
|---|---|---|
| 07-21 21:39:14 | 6 卡并行 extract 启动，30 个 checkpoint 排队，GPU 0–5（launcher 硬性拒绝 >6 卡） | `logs/parallel.log` 前 10 行，含 30 个 checkpoint_id 列表与 6 个 worker pid |
| 07-21 21:56 – 07-22 00:28 | `wait_and_verify.sh` 每 60s 轮询分片进度 0→30 | `logs/_completion_report.recovered_20260730.txt` |
| 07-22 00:10 – 00:28 | 6 个 worker 依次退出（rc=0），30/30 分片完成 | `logs/parallel.log`、`logs/worker_gpu*.log` 末尾 |
| 07-22 00:28 – 01:25 | 第一次 aggregate **失败**：`KeyError: 'domain'`（`diagnosis/hypothesis_summary.py:45 summarize_h1`），rc=1 | `logs/parallel.log` 尾部 traceback |
| 07-22 01:26 | 校验脚本据此判定 MISSING（`hypothesis_summary.json` 与 6 张图都缺） | `logs/_completion_report.recovered_20260730.txt` 末尾 VERDICT |
| 07-22 ~10:20 – 11:40 | 修好后**手工重跑 aggregate**（`--stage aggregate --resume-dir <run>`，输出 tee 到 `parallel2.log`），全部产物写出 | `logs/parallel2.log` 完整成功日志；parquet mtime 10:44、csv 11:07/11:13、bootstrap+summary+图 11:40 |
| 07-22 16:48 | native（未平衡）CDF direction−raw 表 | `cdf_delta_auc_pooled.csv`、`cdf_delta_auc_per_method.csv` mtime |
| 07-23 14:04 | balanced（178v178，配对子采样）CDF 表 + 4 张热图 | `cdf_balanced_delta_auc.csv` + `figures/balanced_heatmap_cdf_*.png` mtime |

extract 单卡吞吐参考：6 卡跑完 30 个 checkpoint × 3 个域约 2h49m（21:39→00:28），即约每卡 5 个 checkpoint。

## 2. 覆盖范围（今日用 duckdb 直接查 parquet 复核）

- `per_frame_scores.parquet`：**39,830,640** 帧行，30 个 checkpoint、3 个域、4 个 infer_mode，16 列（`checkpoint_id, backbone, seed, train_mode, infer_mode, domain, method_id, label, video_id, image_path, feature_norm, real_logit, fake_logit, signed_logit, prob_fake, prediction_correct`）。
- `per_video_scores.parquet`：**982,740** 视频行，15 列（含 `video_prob_fake, mean/std_feature_norm, mean/std_signed_logit, num_frames`）。
- infer_mode × train_mode：`raw` 训练的 15 个 checkpoint 各有 `raw / direction / length` 三行视图（每种 245,685 视频行），`normtrain` 的 15 个只有 `normtrain_norm`（245,685 行）。
- 分域：ffpp 9,853,260 帧 / 182,940 视频行 / 17 个 method_id（含 real）；cdf 23,777,040 帧 / 594,060 视频行 / 17 个 method_id / 2,154 个 video_id；dfd 6,200,340 帧 / 205,740 视频行 / 2 个 method_id。
- 视频级聚合键是 `(method_id, video_id)`（不是单独的 `video_id`），CDF 每个 checkpoint 178 个 real 视频。

## 3. 逐产物溯源

| 产物 | 规模 | 生成者 | 生成命令 | 状态 |
|---|---|---|---|---|
| `per_ckpt/<checkpoint_id>.parquet` × 30 | — | `frame_midt_diagnose.py --stage extract` | `run_diag_parallel.sh 6`（每卡 `--stage extract --only-ckpt <id> --resume-dir <run> --batch-size 256 --num-workers 8 --device cuda`） | **已删除**（聚合后清理），需重跑 extract 才能再生成 |
| `per_frame_scores.parquet` | 1.4 GB / 39,830,640 行 | `frame_midt_diagnose.py --stage aggregate` | `--stage aggregate --resume-dir <run> --n-bootstrap 2000` | 在盘；阶段 3/4 的输入 |
| `per_video_scores.parquet` | 34 MB / 982,740 行 | 同上 | 同上 | 在盘 |
| `per_method_metrics.csv` | 2,160 行 | 同上 | 同上 | 在盘 |
| `aggregate_metrics.csv` | 12 行 | 同上 | 同上 | 在盘 |
| `norm_quartile_buckets.csv` | 180 行 | 同上（`diagnosis/diagnostics.py`） | 同上 | 在盘 |
| `bootstrap_direction_vs_raw.csv` | 30 行（n_bootstrap=2000） | 同上（`diagnosis/bootstrap.py`） | 同上 | 在盘 |
| `delta_auc_per_method_cdf.csv` | **720 行**（无 backbone 列） | 同上（`diagnostics.delta_auc_per_method` 修复前版本） | 同上 | 在盘，但**口径已废弃**，见 §4.2 |
| `hypothesis_summary.json` | H1–H5 | 同上（`diagnosis/hypothesis_summary.py`） | 同上 | 在盘；H1 证据受 §4.2 影响 |
| `figures/02..06_*.png` | 5 张 | 同上（`diagnosis/plotting.py`） | 同上 | 在盘 |
| `figures/01_cdf_delta_auc_heatmap.png` | — | 同上 | 同上 | **已撤回**，见 §4.2（重跑 aggregate 会重新生成，但用的是修好后的口径） |
| `cdf_delta_auc_pooled.csv` | 15 行 | 原为未提交的 `cdf_dir_raw_heatmap.py`（已丢失）→ 今日重写为 `scripts/cdf_native_dir_raw.py` | `python scripts/cdf_native_dir_raw.py <run>` | 在盘，**已验证可复现**（§5） |
| `cdf_delta_auc_per_method.csv` | 240 行 | 同上 | 同上 | 在盘，**已验证可复现**（§5） |
| `cdf_balanced_delta_auc.csv` | 240 行 | `scripts/cdf_balanced_dir_raw.py`（RNG_SEED=20260722 固定） | `python scripts/cdf_balanced_dir_raw.py <run>` | 在盘，**已验证可复现**（§5） |
| `figures/balanced_heatmap_cdf_{clip,dinov2,efficientnet,pooled}.png` | 4 张 | 同上 | 同上 | 在盘，**已验证可复现**（§5） |
| `logs/parallel.log` / `logs/parallel2.log` / `logs/worker_gpu*.log` | — | launcher 与校验脚本（现已直接写入 `<run>/logs/`） | — | 在盘，是本文件的证据来源 |
| `logs/_completion_report.recovered_20260730.txt` | — | `wait_and_verify.sh` | — | **原件 2026-07-29 被删**，07-30 从会话记录恢复为独立命名的副本，见 §4.7 |

跨阶段消费者：

- 阶段 3 `configs/study_config.yaml` 的 `parquet.frame` / `parquet.video`、阶段 4 `configs/pooling_saturation_config.yaml` 的 `parquet_frame` 都指向本 `<run>` 下的 parquet。
- 阶段 1 `reports/final_results_notes.md` 里「`(method_id, video_id)` 分组下 CDF Δ=−0.0044，direction 只胜 4/15」正是 `cdf_delta_auc_pooled.csv` 的汇总——今日重算得 mean Δ=−0.00440、4/15，数值对上。

## 4. 已知偏差与不可复现点

### 4.1 per_ckpt 分片已删除
`aggregate` 阶段需要 `<run>/per_ckpt/*.parquet`。这 30 个分片在聚合完成后被清理，因此**不能只重跑 aggregate**；要重建必须先重跑 extract（6 卡约 3 小时，GPU 只允许 0–5）。`per_frame_scores.parquet` 本身保留，所以下游阶段 3/4 不受影响。

### 4.2 cartesian-join 缺陷：`delta_auc_per_method_cdf.csv`、`figure 01`、`hypothesis_summary.json` 的 H1
当时的 `diagnostics.delta_auc_per_method` 用 `(method_id, seed)` 作为 join 键，漏掉 `backbone`，于是 raw 与 direction 跨 backbone 交叉配对，行数被放大 3 倍（720 而非 240），把一个 backbone 的 raw AUC 和另一个 backbone 的 direction AUC 相减。

- 在盘的 `delta_auc_per_method_cdf.csv` 就是这个 720 行版本（没有 backbone 列），`direction>raw = 347/720 = 0.4819`。
- `hypothesis_summary.json` 的 H1 证据 `n_methods_direction_higher_cdf_auc=347 / n_methods_total=720 / pct=0.4819` 同源。
- `figures/01_cdf_delta_auc_heatmap.png` 画的是同一张表（且 `aggfunc="mean"` 把 backbone 抹平），已被删除撤回，改用按 backbone 分开的 native / balanced 热图。

修复已在库代码里：`src/diagnosis/diagnostics.py::delta_auc_per_method` 现在用 `["backbone","method_id","seed"]` 作键，并有回归测试 `tests/test_audit_fixes.py::test_delta_auc_pairs_within_backbone` 锁住。

用修好后的口径从在盘的 `per_method_metrics.csv` 重算（今日执行）：

| 口径 | 行数 | direction>raw | 比例 | mean Δ |
|---|---|---|---|---|
| 历史（无 backbone 键） | 720 | 347 | 0.4819 | — |
| 修正（含 backbone 键） | 240 | 109 | **0.4542** | −0.00407 |

H1 的判定方向不变（仍 <50%，"partially_supported"），但**引用数字时应使用 109/240 = 45.4%**，不要再引 347/720。修正值与阶段 3 报告里的「CDF video 跨 16 方法 direction 胜的比例 = 0.45」一致。

### 4.3 balanced 脚本的配对缺陷（已修）
`cdf_balanced_dir_raw.py` 早期版本对 raw 和 direction 各自独立抽样，导致两个模式评的不是同一批视频、delta 不配对。现在的版本抽一次索引集、同时应用到两个模式（见函数 `_paired_balanced_auc` 的 docstring）。在盘的 CSV/热图是修复后版本产出的——今日重跑得到完全相同的数值（§5）。

### 4.4 第一次 aggregate 失败留下的误导性文件
`_completion_report.txt` 的结论是「MISSING OR INCOMPLETE ARTIFACTS」，那是针对 01:25 崩掉的第一次聚合。真实完成的是 07-22 上午的第二次聚合（`parallel2.log`）。读日志时不要把这份 report 当作最终状态。

### 4.5 本阶段代码没有进 git
`src/diagnosis/*` 与本阶段的 `scripts/*` 从未被提交（`git log --all -- 'src/deepfake_detection/diagnosis/*'` 为空），所以这批结果没有 commit hash 可锚定，只能靠本文件 + 日志。**建议把阶段 2 的脚本与 `src/diagnosis/` 提交**，否则后续任何改动都无法与 07-21 那次运行做差异比对。

### 4.6 `_completion_report.txt` 原件丢失（2026-07-29）与布局漂移
原 `<run>/_completion_report.txt`（1519 bytes，mtime 07-22 01:26）在 2026-07-29 被删除，同一天这批产物还被移出 `results/`。07-30 已恢复布局（原子 `mv`，校验值一致），并把该报告的正文从 07-28 的会话记录恢复为 `logs/_completion_report.recovered_20260730.txt`——**因为无法证明逐字节一致，它用了不同的文件名**，不冒充原件。其余 28 个文件均为原件，md5 与迁移前一致（见 `reports/stage2_layout_migration.md`）。

### 4.7 数据集依赖
`extract` 依赖 `/Dataset/deepfake_detection/DF40_all`（checkpoint config 里的 `dataset.root`，今日确认存在）。DFD 索引来自 `/Dataset/deepfake_detection/DF40_all/dataset_json/DeepFakeDetection.json`（阶段 1 的 `prepare_dfd_df40.py` 生成）。数据集不在仓库内，换机器必须先备好。

## 5. 今日复现验证记录

| 检查 | 方法 | 结果 |
|---|---|---|
| checkpoint 发现 | `discover_checkpoints(DEFAULT_CKPT_ROOT)` | 30 个，`clip_raw_s42` 的 `best_model.pth` 与 `config/config.yaml` 均存在 |
| 配置加载 | `config_io.load_config(<ckpt>/config/config.yaml)` | OK，`model.name=clip_finetune`，`dataset.root=/Dataset/deepfake_detection/DF40_all` 存在 |
| 脚本可加载 | `frame_midt_diagnose.py --help`、`cdf_native_dir_raw.py --help`、`cdf_balanced_dir_raw.py`（无参数用法提示） | 全部 rc=0；`DEFAULT_CKPT_ROOT`/`DEFAULT_OUTPUT_DIR` 指向重组后的正确目录 |
| native 表可复现 | `cdf_native_dir_raw.py <run> --out-dir /tmp/native_check`，与在盘 CSV 逐格比对 | pooled 15 行、per-method 240 行，列名一致，`max|diff| = 0.0`（含 `n_fake`/`n_real`） |
| balanced 表可复现 | 重跑 `cdf_balanced_dir_raw.py <run>`，与备份逐格比对后还原原文件 | 240 行，`max|diff| = 0.0`；热图重新生成成功 |
| parquet 完整性 | duckdb 统计（§2） | 行数、checkpoint 数、infer_mode 组合全部符合 `parallel2.log` 记录（39,830,640 帧 / 982,740 视频 / 2,160 method 行） |
| H1 修正口径 | 从 `per_method_metrics.csv` 按 `(backbone, method_id, seed)` 重算 | 109/240 = 0.4542，mean Δ = −0.00407 |
| extract 阶段 | **未重跑**（按要求不占 GPU）；仅静态校验路径、参数与依赖 | 若要真验证：`CUDA_VISIBLE_DEVICES=0 python scripts/frame_midt_diagnose.py --stage extract --only-ckpt clip_raw_s42 --domains ffpp --output-dir /tmp/diag_repro`（GPU 只用 0–5） |
