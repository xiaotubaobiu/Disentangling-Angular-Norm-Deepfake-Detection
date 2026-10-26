# 2026-07-30 跨域阈值迁移（threshold transfer）

## 1. 研究问题

在 FF++ **validation** 上完成校准和阈值选择、把参数完全冻结后迁到 CDF，阈值还可靠吗？
分四问：迁移后损失多少 balanced accuracy（`R_transfer`）；源域的 FPR 预算（1% / 5%）是否还成立；温度 / Platt 校准能不能补救；失效来自整体平移、尺度变化、类条件漂移还是校准误差。

主分数是 `video_score = mean_t(signed_logit)`，`signed_logit = fake_logit − real_logit`。
未校准的 `mean_t(sigmoid(s))` 在 0.5 处切一刀只作为**错误基线**，不作主结果。

结论见 [`reports/threshold_transfer_final_report.md`](reports/threshold_transfer_final_report.md)，公式与统计设计见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 2. 输入依赖

全部是其它阶段的只读产物，路径写在 `configs/threshold_study.yaml`（仓库根相对）：

| 输入 | 来源 | 用途 |
|---|---|---|
| `.../2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet` | 阶段 2 | 目标域 CDF 的帧级 `signed_logit`（raw / direction） |
| `.../2026-07-27_pooling_saturation/results/val_cache/*__val.parquet` | 阶段 4 | 源域 **FF++ validation** 帧级 `s_raw` / `s_dir`（15 个 raw checkpoint） |
| `.../2026-05-05_paper_feature_geometry/results/checkpoints` | 阶段 1 | 仅在需要补跑 FF++ val 时使用 |

阶段 2 的 parquet 里 `ffpp` 是 **test** split，所以源域必须用阶段 4 的 val 缓存；两者都原地读取，不复制。
若某个 checkpoint 没有 val 缓存，`01` 可用 `--extract-missing --gpu <0-5>` 补跑，产物写入**本阶段** `results/val_cache/`，不动历史阶段。

## 3. Dry run（一条命令）

```bash
python experiments/2026-07-30_threshold_transfer/scripts/run_all.py --dry-run
```

CLIP / seed 42 / raw+direction / FSGAN+SadTalker，bootstrap 200 次，约 11 秒，纯 CPU。

## 4. Full run（一条命令）

```bash
python experiments/2026-07-30_threshold_transfer/scripts/run_all.py
```

3 backbone × 5 seed × 16 method × 2 mode，bootstrap 2000 次，约 8.5 分钟，纯 CPU（不占 GPU）。

## 5. 输出文件

```text
results/tables/00_input_audit.csv            34 项输入的存在性与规模
results/tables/01_source_videos.csv          每个 checkpoint 的源域视频数（100 real / 1600 fake）
results/tables/02_calibrators.csv            温度 T、Platt A/B 与源域 ECE/NLL/Brier
results/tables/03_source_thresholds.csv      6 条规则的阈值 + 源域操作点
results/tables/04_transfer_pooled.csv        pooled 迁移指标（含 oracle、R_transfer、固定 FPR 误差）
results/tables/04_transfer_per_method.csv    逐方法迁移指标（16 方法）
results/tables/04_bootstrap.csv              视频聚类 bootstrap（raw / direction / 配对差）
results/tables/04_target_video_scores.parquet 目标域视频级分数缓存（供 05 复用）
results/tables/05_score_shift.csv            漂移分解 + 主因判定
results/tables/05_class_distributions.csv    源/目标 × real/fake 的分布统计
results/val_cache/source_video_scores.parquet 源域视频级分数（由 01 生成）
results/logs/run_all_{dry,full}.log          编排日志
figures/05_*.png  figures/06_*.png           分布/BA-阈值曲线/漂移/regret/FPR 预算/ECE/逐方法热图
reports/input_audit.md                       输入审计报告
reports/threshold_transfer_final_report.md   最终报告（直接回答 6 个问题）
```

## 6. 关键指标定义

| 指标 | 定义 |
|---|---|
| `video_score` | `mean_t(signed_logit)`（`mean_logit` 聚合），视频键为 `(method_id, video_id)` |
| `t_BA` / `t_equal` | 源域上最大化 `(TPR+TNR)/2` / 最小化 `|TPR−TNR|`，**两个不同定义** |
| `t_FPR1` / `t_FPR5` | 源域 FPR ≤ 1% / 5% 的最小阈值 |
| `t_p50_temperature` | 温度校准下 `p=0.5` ⇔ `s=0`（对任意 `T` 成立，故阈值与温度无关） |
| `t_p50_platt` | Platt 校准下 `p=0.5` ⇔ `s=−B/A` |
| `R_transfer` | `BA_CDF(CDF oracle 阈值) − BA_CDF(源域阈值)`，≥ 0；只衡量阈值损失，不含排序损失 |
| 固定 FPR 迁移误差 | `FPR_CDF(t_FPR1) − 0.01`、`FPR_CDF(t_FPR5) − 0.05` |
| 漂移分解 | `shift_c`、`common_shift`、`differential_shift`、`scale_c`、Wasserstein-1、KS |
| 不确定度 | 视频聚类 bootstrap（2000 次，种子 20260730），raw/direction **配对**重采样 |

## 7. 数据泄漏禁止事项

- CDF **不得**用于拟合校准器、选阈值或选模型/种子。`02`、`03` 只读 FF++ val 缓存，源码里不出现目标域输入（`tests/test_no_leakage.py` 用 AST 静态检查 + 把目标域 loader 替换为抛异常的函数双重验证）。
- CDF oracle 阈值**仅作诊断**，用来定义 `R_transfer`，不得当作可部署结果。
- DFD 全程排除。
- 四条阈值规则对所有 cell 一律全报，不允许用目标域表现回头挑规则。
- 阈值贴近 `p=0.5` 不代表校准良好；`t_BA ≠ t_equal`。

## 8. 已知局限

1. FF++ val 来自阶段 4 的缓存：每方法 100 视频 × 每视频 8 帧的**子采样**，视频分数比 CDF（32 帧）更noisy。
2. 只覆盖 `raw` 训练的 15 个 checkpoint（只有它们同时有 raw 与 direction 视图）；`normtrain`、`length` 模式不在范围内。
3. CDF 每个 checkpoint 只有 178 个 real 视频，realised FPR 的分辨率约 0.6%，1% 预算无法更精细地验证。
4. oracle 阈值在同一份目标样本上选并评估，`R_transfer` 对 oracle 略偏乐观（目标域没有再切分）。
5. 只做了 FF++→CDF 一个方向；DFD 与其它目标域未验证。
