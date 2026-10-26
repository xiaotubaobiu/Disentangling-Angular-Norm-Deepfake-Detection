# 2026-07-27 池化与饱和机制研究（pooling / saturation）

时间跨度：2026-07-27 ~ 2026-07-28　分支：`study/pooling-saturation`

## 做了什么

追问阶段 1/3 留下的核心疑点：视频级 `mean_prob` 下 direction > raw，到底是特征方向真的更好（B），还是 sigmoid 饱和 + 概率池化造成的伪影（A）？还是聚合器降方差（C）或方法依赖（D）？

数据：FF++(test) + CDF（排除 DFD），raw-trained checkpoint × 3 backbone × 5 seed × 16 方法。所有校准参数（温度 T / robust scale / Platt）**只在 FF++ val 上拟合**，CDF 的 γ_best 仅作 oracle 诊断，不参与决策。

四组实验：

1. 聚合器对照：`mean_prob` / `mean_logit` / `sigmoid_mean_logit` / `median_logit` / `clip_prob` / `trimmed_logit` / `topk_logit`，配 Jensen gap 与 γ 扫描。
2. 饱和统计：饱和率、sigmoid 梯度、raw vs direction 的 logit 尺度差。
3. 温度 / 尺度校准：FF++ val 重推理（→ `results/val_cache/`）后拟合 T、robust scale、Platt，报 ECE/Brier，再看 direction 优势是否还在。
4. 视频内时序一致性 + 逐聚合器的 rank-flip H/B 分解。

## 主要结论

**判定：A 为主因 + D（方法依赖）。**

- 池化伪影（A）：7/32 个 method×域组合满足 `Delta_prob > 0` 且 `Delta_logit ≤ 0`（22%）；这些组合温度校准后 `Delta_temp` 均值 +0.0005，即优势基本归零。
- 饱和证据：fake 侧饱和率差（阈 0.10）raw − direction 均值 +0.9449，raw 明显更饱和。
- 降方差（C）：视频内 std(raw) − std(direction) 均值 +1.3568，direction 确实降方差。
- 真方向作用（B）：10/32 组合在无饱和的 `mean_logit` 下 direction 仍 > 0，温度后仍有 8/32，所以 B 不为零但不是主因。
- 方法依赖（D）：I_pool 在 FSGAN 为 +0.0167、SadTalker 为 −0.0252，符号相反、幅度大。

**方法建议**：视频级评估必须先做温度校准（FF++ val 拟合）再聚合，或直接用 `mean_logit` / `median` 这类无饱和聚合；不得把 `mean_prob` 上的优势解读为「方向特征更好」；暂停复杂的径向—角向训练，若要自适应就用样本级径向可靠性门控，不依赖测试时的 method label。

## 目录内容

```text
scripts/
  06_pooling_saturation_analysis.py  # 实验 1/2/3：聚合器、饱和统计、Jensen gap、γ 扫描
  07_fit_temperature_calibration.py  # FF++ val 重推理 + 温度/robust-scale/Platt 拟合，ECE/Brier
  08_pooling_rank_flip.py            # 逐聚合器的 rank-flip H/B 分解
  09_generate_pooling_report.py      # 视频内时序一致性 + A/B/C/D 最终报告
configs/pooling_saturation_config.yaml
results/tables/
  Q_pooling_aggregation_auc.csv        # 各聚合器的视频级 AUC
  Q_pooling_saturation_stats.csv       # 饱和率与 sigmoid 梯度
  Q_pooling_jensen_gap.csv             # Jensen gap
  Q_pooling_gamma_scan.csv             # γ 扫描
  Q_pooling_temperature.csv            # 拟合出的 T / scale / Platt 参数
  Q_pooling_calibrated_auc.csv         # 校准后 AUC
  Q_pooling_scale_matched_auc.csv      # 尺度对齐后 AUC
  Q_pooling_rank_flip_by_aggregator.csv
  Q_pooling_temporal_consistency.csv   # 92MB，视频内逐帧一致性明细
results/val_cache/                     # FF++ val 重推理缓存（parquet）
figures/Q_pooling_*.png                # 18 张：饱和、校准前后、可靠性图、逐聚合器 H/B、时序一致性等
reports/pooling_saturation_final_report.md   # 最终报告（主文档）
docs/                                  # 设计与实施计划
tests/                                 # 7 个单元测试（aggregators / saturation / temperature / temporal / video_eval / gamma / config）
```

## 怎么重跑

```bash
cd <仓库根>
P=experiments/2026-07-27_pooling_saturation

python $P/scripts/06_pooling_saturation_analysis.py           # 支持 --smoke 小子集先验证
CUDA_VISIBLE_DEVICES=0 python $P/scripts/07_fit_temperature_calibration.py   # 需要 GPU（FF++ val 重推理）
python $P/scripts/08_pooling_rank_flip.py
python $P/scripts/09_generate_pooling_report.py

python -m pytest $P/tests -q
```

数据来源写在 `configs/pooling_saturation_config.yaml`：`parquet_frame` 指向阶段 2 的 `per_frame_scores.parquet`，`ckpt_root` 指向阶段 1 的 `results/checkpoints`（仓库根相对路径）；`paths.*` 相对本阶段目录。
