# 2026-07-21 帧级中期诊断（frame-level midterm diagnosis）

时间跨度：2026-07-21 21:39 ~ 2026-07-23 14:04（extract → aggregate → CDF direction−raw 复核）

逐产物溯源、时间线、已知偏差与复核记录见 [`results/parallel_20260721_213914/reports/provenance.md`](results/parallel_20260721_213914/reports/provenance.md)。**引用 H1 数字前请先看那份文档的 §4.2。**

本阶段只有一个 run 目录 `results/parallel_20260721_213914/`，它是唯一的真实来源；表格与 parquet 在其根下，图在 `figures/`，日志在 `logs/`，溯源文档在 `reports/`。

## 做了什么

把阶段 1 的 30 个 checkpoint 在 FF++ / CDF / DFD 上做**帧级全量重推理**，落成一张口径统一的 parquet 表，再在这张表上做诊断。目的不是刷指标，而是造一个后续所有分析都能共用的证据底座。

- 30 个 checkpoint × 3 个域，逐帧记录 `feature_norm / real_logit / fake_logit / signed_logit / prob_fake / prediction_correct` 加 `video_id / method_id / infer_mode` 等键。
- `raw` 训练的 15 个 checkpoint 产出 `raw / direction / length` 三种推理视图，`normtrain` 的 15 个产出 `normtrain_norm`。
- 两阶段流水线：`--stage extract`（6 卡并行，每个 checkpoint 独立写 `per_ckpt/<id>.parquet`，mkdir 原子抢占 + 原子重命名，可断点续跑）→ `--stage aggregate`（合并 + 指标 + bootstrap + 图 + 假设判定）。
- 视频级聚合按 `(method_id, video_id)` 分组，不用单独的 `video_id`——后者会把同一身份的不同伪造方法平均成一个"视频"，正是阶段 1 那个 `direction > raw` 结论的成因。

实际规模（今日用 duckdb 复核）：帧级 **39,830,640** 行、视频级 **982,740** 行、per-method 指标 **2,160** 行；ffpp / cdf / dfd 帧数分别 9,853,260 / 23,777,040 / 6,200,340；CDF 每个 checkpoint 178 个 real 视频、16 个伪造方法。

## 主要结论（`<run>/hypothesis_summary.json`，`<run>` = `results/parallel_20260721_213914/`）

| 假设 | 判定 | 要点 |
|---|---|---|
| H1 direction 是否降低 FF++→CDF 域偏移 | 部分支持 | D_domain 从 0.0276（raw）降到 0.0106（direction）；但按方法看 direction 的 CDF AUC 更高的只有 **109/240 = 45.4%**（json 里记的 347/720 是修复前的跨 backbone 交叉配对，见 provenance §4.2） |
| H2 direction 的低阈值是否伴随更小 threshold regret | 否 | direction regret 0.2546 ≈ raw 0.2489，只是概率往 0.5 收缩；direction 最优阈值 0.5825±0.0361 |
| H3 方法间 direction 分数差异是否大于 seed 波动 | 否 | D_method_cdf 0.0136 vs seed 噪声底 0.0348，比值 0.39 |
| H4 normtrain 是否降低错误但加大方法间离散 | 否 | D_method_cdf：direction 0.0136 / normtrain 0.0105；T_method_cdf：0.0127 / 0.0082 |
| H5 feature norm 能否预测帧是否判对 | 部分支持 | corr(norm, prob_fake)=−0.031、corr(norm, correct)=+0.040，标记 `mixed`，更像 "confidence polarizer" |

CDF 上 direction vs raw 的三种口径（都在 `results/parallel_20260721_213914/`）：

| 口径 | 文件 | 结果 |
|---|---|---|
| native pooled（全部 fake vs 178 real） | `cdf_delta_auc_pooled.csv` | mean Δ = −0.0044，direction 只在 4/15 个 checkpoint 上胜 |
| native per-method | `cdf_delta_auc_per_method.csv` | 240 格，mean Δ = −0.0041，109/240 direction 胜 |
| balanced 178v178（配对子采样） | `cdf_balanced_delta_auc.csv` + `figures/balanced_heatmap_cdf_*.png` | clip mean Δ +0.0000、dinov2 −0.0005、efficientnet −0.0141 |

即：在语义正确的分组下，「direction 更好」在 CDF 上并不成立。

## 目录内容

```text
scripts/
  frame_midt_diagnose.py    # 主流水线（--stage extract / aggregate / all）
  run_diag_parallel.sh      # 6 卡并行 launcher（只用 GPU 0-5，>6 卡直接报错）
  wait_and_run_diag.sh      # 等 GPU 0-5 空闲后自动开跑
  wait_and_verify.sh        # 轮询完成度并校验必需产物
  cdf_native_dir_raw.py     # CDF native（未平衡）pooled/per-method direction−raw（补回的生成脚本）
  cdf_balanced_dir_raw.py   # CDF balanced 178v178 配对子采样版 + 4 张热图
results/parallel_20260721_213914/     # 唯一的历史 run 目录
  per_frame_scores.parquet      1.4 GB / 39,830,640 行 —— 阶段 3、4 的输入
  per_video_scores.parquet      34 MB / 982,740 行
  per_method_metrics.csv        2,160 行
  aggregate_metrics.csv         12 行
  norm_quartile_buckets.csv     180 行
  bootstrap_direction_vs_raw.csv 30 行（n_bootstrap=2000）
  delta_auc_per_method_cdf.csv  720 行（修复前口径，已废弃，见 provenance §4.2）
  cdf_delta_auc_pooled.csv / cdf_delta_auc_per_method.csv     native 口径
  cdf_balanced_delta_auc.csv                                 balanced 口径
  hypothesis_summary.json       H1-H5 判定
  figures/                      02-06 主图 + 4 张 balanced 热图（01 已撤回）
  logs/                         parallel.log、parallel2.log、worker_gpu0-5.log、
                                _completion_report.recovered_20260730.txt
  reports/provenance.md         逐产物溯源 / 时间线 / 已知偏差 / 复核记录
```

`per_ckpt/` 的 30 个分片在聚合后被清理，所以 aggregate 不能单独重跑（详见 provenance §4.1）。

## 怎么重跑

GPU 只允许使用 0–5 号卡（launcher 已硬性限制）。

```bash
cd <仓库根>
P=experiments/2026-07-21_frame_midterm_diagnosis
R=$P/results/parallel_20260721_213914

# 全量重建（extract 6 卡约 3 小时，然后自动 aggregate；新 run 目录为 $P/results/parallel_<时间戳>/）
bash $P/scripts/run_diag_parallel.sh 6

# 单卡跑全流程
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python $P/scripts/frame_midt_diagnose.py \
  --n-bootstrap 2000 --batch-size 256 --num-workers 8 --device cuda

# 单 checkpoint / 单域冒烟（验证环境与路径，不动历史产物）
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python $P/scripts/frame_midt_diagnose.py \
  --stage extract --only-ckpt clip_raw_s42 --domains ffpp --output-dir /tmp/diag_repro

# 只重跑聚合（需要 per_ckpt/ 分片存在）
PYTHONPATH=src python $P/scripts/frame_midt_diagnose.py --stage aggregate --resume-dir <run_dir>

# CDF direction−raw 复核（纯 CPU，几十秒，读 per_video_scores.parquet）
python $P/scripts/cdf_native_dir_raw.py   $R      # 加 --out-dir 可写到别处做比对
python $P/scripts/cdf_balanced_dir_raw.py $R
```

`--ckpt-root` 默认指向阶段 1 的 `results/checkpoints`（已验证能发现 30 个 checkpoint），`--output-dir` 默认指向本阶段 `results/`，`<output-dir>/<时间戳>` 即新 run 目录。数据集需要 `/Dataset/deepfake_detection/DF40_all`（checkpoint config 里的 `dataset.root`）。
