# Existing-results figures for the 2026-09-28 paper revision

All plots use existing small CSV tables. They add no training, fitting, resampling, or inferential statistics. The 15 fixed checkpoints are the three raw-trained backbones with five seeds each. PNGs are rendered at 300 dpi; PDFs retain vector graphics and embedded TrueType fonts. The source tables are read-only and their SHA256 hashes are recorded before and checked after rendering.

Reproduce with:

```bash
python /home/z/project/deepfake_detection/projects/feature_geometry/reports/paper_rewrite_20260928/figures/render_figures.py
```

Machine-readable quantities and input provenance are in `figure_metrics.json` and `input_sha256.json`. No confidence intervals are shown in these figures. Model–method cells share test data and are not independent datasets.

## Figure 1 — cdf_normalization_effects

**中文图注。** CDF全部16种生成方法的推理归一化效应。每个点表示同一方法上15个固定模型的平均AUC(direction)−AUC(raw)，单位为百分点；三种设置分别为逐帧概率均值、logit均值和使用源验证集温度后的逐帧概率均值。温度参数在各检查点、各视图分别拟合。零线右侧表示direction的平均AUC较高。图中不含置信区间，均值正负不代表统计显著性。

**English caption.** Inference-normalization effects for all 16 generation methods in CDF. Each point is the equal-weight mean of AUC(direction) minus AUC(raw) over 15 fixed models, expressed in percentage points. The three settings use probability averaging, logit averaging, and probability averaging after source-validation temperature fitting separately for each checkpoint and view. Positive values favor direction. No confidence intervals are shown; the sign of a mean does not establish statistical significance.

Key checks: FSGAN = +2.9191 / −0.2647 / −0.0012 pp; SadTalker = −4.7818 / +0.3547 / +0.0215 pp, in probability / logit / source-temperature order. These are effects, not absolute AUCs.

## Figure 2 — aggregation_sensitivity

**中文图注。** 固定raw或direction视图后，从概率均值换为logit均值的AUC变化分布。各面板分别包含一个数据域下15模型×16方法的240个配置单元；FF++与CDF分列，raw与direction分行，同一行使用相同纵轴，不同行纵轴尺度不同。点为各单元，箱体为四分位区间，中线为中位数，须为最小至最大值；横向抖动仅用于显示。面板标注有符号均值和平均绝对变化。配置之间的离散程度不是置信区间，240单元也不视为独立数据集。

**English caption.** Distribution of AUC changes when switching from probability averaging to logit averaging within a fixed raw or direction view. Each panel contains 240 model–method cells (15 models × 16 methods). FF++ and CDF are shown separately; vertical scales are shared within each row and differ between the raw and direction rows. Dots show individual cells with visual-only horizontal jitter. Boxes show the interquartile range and median; whiskers span the minimum and maximum. Panel annotations report signed means and mean absolute changes. Configuration dispersion is descriptive and is not a confidence interval.

Key checks: mean absolute changes (raw / direction) are FF++ 0.0776 / 0.0020 pp and CDF 1.0673 / 0.0172 pp. CDF/raw has signed mean −0.0731 pp; 72/240 cells have absolute changes of at least 1 pp.

## Figure 3 — gamma_curves

**中文图注。** CDF/FSGAN与CDF/SadTalker在连续径向指数γ下的平均绝对视频AUC。横轴使用既有0、0.1、…、1网格，纵轴为AUC百分数；每个γ点先对相同15个模型等权平均。数学上γ=0与γ=1分别对应direction和raw；所绘两方法的缓存端点与主聚合表严格一致。圆环标记平均曲线在网格上的第一个最大值，不跨模型选择最高单行结果。两面板纵轴范围不同。最大值使用目标标签，仅作诊断，图中无置信区间。

**English caption.** Mean absolute video AUC along the radial-exponent path for FSGAN and SadTalker in CDF. At each existing grid point γ = 0, 0.1, …, 1, AUCs are first averaged equally over the same 15 models. The mathematical endpoints γ = 0 and γ = 1 correspond to direction and raw, and the cached endpoints for both plotted methods exactly match the main aggregation table. Rings mark the first grid maximum of each mean curve, without selecting the best individual model. The panels use different vertical ranges. These target-label optima are diagnostic; no confidence intervals are shown.

Key checks: FSGAN mean-curve maximizing γ values are 0.2 (probability) and 1.0 (logit); SadTalker values are 1.0 and 0.0, respectively. Y is absolute AUC (%), not AUC difference.

Across the complete gamma table, γ=0 differs from the main direction table in 3/960 entries, all CDF: clip/123/inswap/mean_prob = +0.001324994700013793 pp; dinov2/999/simswap/mean_prob = −0.0008764395519778034 pp; efficientnet/123/tpsm/mean_logit = −0.0008683118281394009 pp. The sign is gamma-table AUC minus main-table direction AUC. γ=1 matches all 960 raw entries exactly. The cause of the three small existing-table discrepancies has not been investigated here; source values were not changed. Exact keys and differences are recorded in `cross_table_audit.gamma_endpoint_audit` in the metrics JSON.

## Figure 4 — threshold_operating_points

**中文图注。** 源端最大BA阈值冻结后的平衡准确率与误报率。采用mean_logit分数，每个模型、每个视图在相同FF++验证集上分别选择使BA最大的阈值，然后固定应用于CDF。每个点平均15个模型；CDF将所有伪造方法合并为pooled目标集合。FF++源结果在用于阈值选择的验证集上计算，不是独立源测试结果。连接线仅连接源/目标均值，图中无置信区间。

**English caption.** Balanced accuracy and false-positive rate after freezing source max-BA thresholds. With mean-logit scores, each model and view selects its own threshold on the same FF++ validation set and applies it unchanged to CDF. Each point averages 15 models. CDF pools all fake methods. Source results are evaluated on the validation set used for threshold selection, not an independent source test set. Lines connect source and target means; no confidence intervals are shown.

Key checks (raw / direction): source BA 96.3396% / 96.3291%; target BA 61.7927% / 61.4126%; source FPR 2.4667% / 2.7333%; target FPR 74.8315% / 75.6554%. Across the 30 checkpoint–view cells, source/target BA = 96.33436% / 61.60262% and FPR = 2.6% / 75.24344%.

## Verification

- The rendering script completed successfully and validated input dimensions, unique model–method keys, domain/method/seed/backbone coverage, effect-column definitions, gamma-grid coverage, and the plotted endpoint matches.
- All four PNGs were opened using `view_image`; labels, ranges, legends, and figure notes were visually inspected. The distribution and threshold figures were revised to remove a rounded negative-zero label and an annotation overlap.
- All four source hashes were unchanged across rendering. No Obsidian/vault file, source CSV, or model asset was modified by this figure task.
