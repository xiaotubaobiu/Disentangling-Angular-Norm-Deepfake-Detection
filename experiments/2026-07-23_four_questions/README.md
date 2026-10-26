# 2026-07-23 四问题实证检验（four questions study）

时间跨度：2026-07-23 ~ 2026-07-24　分支：`study/four-questions`

## 做了什么

在阶段 2 的 `per_frame_scores.parquet` 上回答四个问题。全程排除 DFD，主分析用 raw-trained checkpoint。

- **Q1**：为什么归一化对 FSGAN 有增益、对 SadTalker 反而变差？→ rank-flip 恒等式 + H/B 解剖 + 部分归一化 γ 曲线。
- **Q2**：模长与真假的关系跨域是否稳定？→ delta/RTG/STG、方法级相关 + permutation + LOMO、混合效应。
- **Q3**：CLIP 的 length-only 高 AUC 可靠吗？→ micro/macro/video/equalized 四种口径 + 逐方法 + seed 稳定性。
- **Q4**：多方法对比学习（假设存在跨伪造方法的公共方向）有依据吗？→ 重抽特征 z，做 same-frame 位移对齐 vs 负对照。

## 主要结论

| 问题 | 结论 | 关键数值 |
|---|---|---|
| Q1 | 支持 | CDF 上 FSGAN H=0.0357 ≫ B=0.0065（ΔAUC +0.0292）；SadTalker 反向 B=0.0525 ≫ H=0.0047（ΔAUC −0.0478）。最优 γ 因方法而异（FSGAN≈0.14，SadTalker≈1.00）→ 否定一刀切硬归一化。**direction>raw 只在 video(mean-sigmoid) 层级成立**，帧级 CDF 上 raw 胜的比例 0.68 |
| Q2 | 部分支持 | RTG 均值 0.694，符号翻转比例 0.158；与归一化增益的相关性依赖 backbone（仅 DINOv2 spearman=0.526, p=0.036 显著；CLIP 负相关，EfficientNet 零相关）→ 有关联，不证因果 |
| Q3 | 支持 | CLIP CDF length-only：micro 0.852 / macro 0.852 / video 0.812（95% CI [0.779,0.842]）/ equalized 0.871；16 个方法全 > 0.5；对照 DINOv2 macro 0.63、EfficientNet 0.76。局限：parquet 里的 ffpp 是 test split，只用于 1-bit 符号选择，所以这是 in-domain 判别力的证据 |
| Q4 | **不支持** | same-frame chord alignment 一律 ≤ mismatch 负对照（6/6，如 clip/cdf 0.6757 vs 0.7367），公共方向不存在。按 spec 的 smoke-first 硬门，全量 15-checkpoint 抽取被跳过 |

**决策**：取 A（径向—角向自适应，per-method/per-input 的 γ，把 ‖z‖ 当门控特征而非硬归一化），放弃 B（多方法公共方向对齐）。详见 `reports/next_step_recommendation.md`。

## 目录内容

```text
scripts/
  00_inspect_project.py     # parquet/checkpoint schema 巡检 → reports/schema_report.md
  01_rank_flip_analysis.py  # Q1：rank-flip 恒等式、H/B、γ 曲线
  02_radial_stability.py    # Q2：delta/RTG/STG、相关性 + permutation + LOMO
  03_length_robustness.py   # Q3：length-only 四种口径 AUC
  04_audit_pairs.py         # Q4 前置：配对审计硬门（FS/FR 不得跨族配对）
  04b_q4_gate.py            # Q4 前置：特征抽取一致性门控
  04c_q4_run.py             # Q4：z 抽取 + 多方法对齐几何（--smoke / --full）
  05_generate_report.py     # 生成中期 / 最终报告（--midterm / --final）
  run_all.py                # 按顺序串起全流程（不含被门控跳过的 Q4 全量）
configs/study_config.yaml   # 数据源、方法族划分、采样、bootstrap、γ 网格、smoke 子集
results/tables/             # Q1_* / Q2_* / Q3_* / Q4_* 明细表
results/z_cache/            # Q4 抽取的 z 缓存（6 个 parquet）
results/logs/               # q1_run.log、Q2_correlations.md、run_all.log
figures/                    # Q1_* / Q2_* / Q3_* / Q4_* 图
reports/
  schema_report.md              # 数据/checkpoint 巡检
  midterm_report.md             # Q1-Q3 中期结论
  Q4_pairing_audit.md           # 配对审计
  Q4_consistency_gate.md        # 抽取一致性门控
  four_questions_final_report.md # 最终报告（主文档）
  next_step_recommendation.md    # 下一步建议与遗留问题
docs/                       # 设计与实施计划
tests/                      # 11 个单元测试（io / canonical / auc_pairs / length / radial / q4 / bootstrap / 采样 / 配置）
```

## 怎么重跑

```bash
cd <仓库根>
P=experiments/2026-07-23_four_questions

python $P/scripts/run_all.py              # 全流程（含 Q4 smoke，不含 Q4 全量）

python $P/scripts/00_inspect_project.py   # 单步示例
python $P/scripts/01_rank_flip_analysis.py
python $P/scripts/05_generate_report.py --final

python -m pytest $P/tests -q              # 本阶段测试
```

数据来源写在 `configs/study_config.yaml`：`parquet.frame` 指向阶段 2 的 `per_frame_scores.parquet`，`ckpt_root` 指向阶段 1 的 `results/checkpoints`（都是仓库根相对路径）；`paths.*` 是相对本阶段目录的输出路径。
