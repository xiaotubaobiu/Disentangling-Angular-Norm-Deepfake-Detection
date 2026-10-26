# DINO五种子已验收结果 / Accepted five-seed DINO results

*本材料仅含已独立验收的DINO正式终点结果；EfficientNet、H及全研究结论仍待完成。 / This material contains accepted DINO formal endpoints only; EfficientNet, H and full-study conclusions remain pending.*

各行使用注册种子7、42、123、2026、3407，报告训练种子均值±样本SD（ddof=1）。source_test、target_full与RQ4固定target_test保持分开。CAL先在每个模型内归约200个共同子集，再跨五模型汇总；该SD不等于CAL子集波动或条件TEST bootstrap区间。碰撞macro AUC按冻结规则保留NA。

Every row uses the five registered seeds and reports mean ± sample SD (ddof=1). Cohorts remain separate. CAL repeats are reduced within each model before aggregation across training seeds. This SD is distinct from CAL-subset spread and conditional TEST bootstrap intervals. Collision-grouping macro AUC remains NA as prespecified.

## AUC比较 / AUC comparisons

| Cohort | Grouping | Representation | Aggregation | Pooled AUC | Macro AUC |
|---|---|---|---|---:|---:|
| source_test | collision | l2 | mean_logit | 1.000000 ± 0.000000 | NA |
| source_test | collision | l2 | mean_probability | 1.000000 ± 0.000000 | NA |
| source_test | collision | raw | mean_logit | 1.000000 ± 0.000000 | NA |
| source_test | collision | raw | mean_probability | 0.999648 ± 0.000787 | NA |
| source_test | correct | l2 | mean_logit | 0.999648 ± 0.000091 | 0.999654 ± 0.000089 |
| source_test | correct | l2 | mean_probability | 0.999649 ± 0.000088 | 0.999655 ± 0.000086 |
| source_test | correct | raw | mean_logit | 0.999650 ± 0.000097 | 0.999656 ± 0.000095 |
| source_test | correct | raw | mean_probability | 0.999587 ± 0.000093 | 0.999592 ± 0.000093 |
| target_full | collision | l2 | mean_logit | 0.689746 ± 0.058947 | NA |
| target_full | collision | l2 | mean_probability | 0.689686 ± 0.058965 | NA |
| target_full | collision | raw | mean_logit | 0.665203 ± 0.100273 | NA |
| target_full | collision | raw | mean_probability | 0.638549 ± 0.087206 | NA |
| target_full | correct | l2 | mean_logit | 0.745415 ± 0.038028 | 0.742525 ± 0.035583 |
| target_full | correct | l2 | mean_probability | 0.745422 ± 0.038013 | 0.742532 ± 0.035569 |
| target_full | correct | raw | mean_logit | 0.709934 ± 0.084645 | 0.707721 ± 0.080862 |
| target_full | correct | raw | mean_probability | 0.716150 ± 0.078786 | 0.714017 ± 0.075237 |

## 同种子归一化效应 / Within-seed normalization effects

Δ为L2减Raw的pooled AUC差，交互为collision效应减correct效应；翻转在同一seed内比较两效应符号。 / Δ denotes L2 minus Raw pooled AUC; interaction is the collision effect minus the correct-grouping effect. Sign reversals are evaluated within each training seed.

| Cohort | Aggregation | Δ correct | Δ collision | Interaction | Seed reversals |
|---|---|---:|---:|---:|---:|
| source_test | mean_logit | -0.000001 ± 0.000009 | 0.000000 ± 0.000000 | 0.000001 ± 0.000009 | 0/5 |
| source_test | mean_probability | 0.000062 ± 0.000084 | 0.000352 ± 0.000787 | 0.000290 ± 0.000735 | 0/5 |
| target_full | mean_logit | 0.035481 ± 0.047339 | 0.024543 ± 0.047192 | -0.010938 ± 0.015472 | 0/5 |
| target_full | mean_probability | 0.029272 ± 0.042135 | 0.051137 ± 0.036608 | 0.021865 ± 0.014711 | 2/5 |

## 固定TEST校准操作点 / Calibration operating points on the fixed TEST

以下仅摘录预定主预算n=20与源真实池操作点；所有预算、默认点、oracle、逐seed结果及其TEST区间保留在完整绑定产物。采用声明的FP64 score域，不据此作总体风险保证。 / The prespecified n=20 budget and source-real operating points are shown. All budgets, defaults, oracles, per-seed results and their TEST intervals remain in the full bound artifacts. Decisions use the declared FP64 score space; these observations do not establish a population risk guarantee.

| Representation | Aggregation | Calibration | n | FPR | TPR | Pooled BA |
|---|---|---|---:|---:|---:|---:|
| l2 | mean_logit | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| l2 | mean_logit | target_real | 20 | 0.088093 ± 0.019582 | 0.368011 ± 0.090991 | 0.639959 ± 0.045499 |
| l2 | mean_probability | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| l2 | mean_probability | target_real | 20 | 0.088124 ± 0.019574 | 0.367968 ± 0.090943 | 0.639922 ± 0.045462 |
| raw | mean_logit | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| raw | mean_logit | target_real | 20 | 0.073814 ± 0.041030 | 0.295275 ± 0.156905 | 0.610730 ± 0.078340 |
| raw | mean_probability | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| raw | mean_probability | target_real | 20 | 0.071278 ± 0.041758 | 0.290221 ± 0.153673 | 0.609471 ± 0.075757 |

## 结果段落准备 / Results paragraph preparation

DINO五个正式终点在target_full正确分组和mean-logit下的macro AUC为Raw 0.707721 ± 0.080862、L2 0.742525 ± 0.035583。同seed归一化效应异号为0/5；五种子汇总均值不能替代逐seed效应或条件TEST区间。Raw与n=20目标真实校准的固定TEST FPR为0.073814 ± 0.041030，TPR为0.295275 ± 0.156905，pooled BA为0.610730 ± 0.078340。该结果反映本骨干的排序和操作点表现，不据此改动阈值选择、分数方向或H方案。完整跨骨干及训练历史结论仍待其余已注册实验。

Across the five DINO formal endpoints, target_full macro AUC under correct grouping and mean-logit aggregation was 0.707721 ± 0.080862 for Raw and 0.742525 ± 0.035583 for L2. Within-seed sign reversals numbered 0/5; an average across training seeds does not replace per-seed effects or conditional TEST intervals. With Raw and the n=20 target-real calibration budget, fixed-TEST FPR, TPR and pooled BA were 0.073814 ± 0.041030, 0.295275 ± 0.156905 and 0.610730 ± 0.078340. These observations describe this backbone and do not change threshold selection, score direction or the H protocol. Full cross-backbone and training-history conclusions await the remaining registered experiments.

## 来源 / Provenance

- Summary: `/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/analysis/D/dinov2_vit_b14/five_seed_summary`; completion SHA256 `455f9e7e9d2bdf7ac1c01504fb36d1c3b4f6609746b03a82d53b94cdef0f2508`.
- Independent audit: `/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/dinov2_vit_b14_five_seed_summary_independent_acceptance_20261024T1630.json`; SHA256 `89c9309cacae09bbed76a629e666b888e3042dae020ad73c187c494e5bb7c2dd`; 12 tables, 48,476 fields; maximum absolute difference 4.55e-13.
- Original conditional TEST intervals remain in supporting_artifacts.json; endpoints are not averaged across seeds.
