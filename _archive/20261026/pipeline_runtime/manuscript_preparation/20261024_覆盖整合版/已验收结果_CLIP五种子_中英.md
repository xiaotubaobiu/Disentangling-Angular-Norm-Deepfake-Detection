# CLIP五种子已验收结果 / Accepted five-seed CLIP results

*本材料仅含已独立验收的CLIP正式终点结果；其余D骨干、H及完整论文结论仍待完成。 / This material contains accepted CLIP formal endpoints only; the other D backbones, H and full-study conclusions remain pending.*

各行采用固定五种子7、42、123、2026、3407，报告训练种子均值±样本SD（ddof=1）。source_test、target_full与RQ4固定target_test分别报告。CAL先在每模型内归约200个共同子集，再跨五个模型汇总；该SD既不是CAL子集波动，也不是TEST bootstrap区间。碰撞分组macro AUC按预定规则保留NA。

Every row uses the five registered training seeds and reports mean ± sample SD (ddof=1). source_test, target_full and the fixed RQ4 target_test remain separate. CAL repeats are reduced within each model before aggregation across five models. This SD is distinct from CAL-subset spread and conditional TEST bootstrap intervals. Collision-grouping macro AUC remains NA as prespecified.

## AUC比较 / AUC comparisons

| Cohort | Grouping | Representation | Aggregation | Pooled AUC | Macro AUC |
|---|---|---|---|---:|---:|
| source_test | collision | l2 | mean_logit | 0.999985 ± 0.000023 | NA |
| source_test | collision | l2 | mean_probability | 0.999985 ± 0.000023 | NA |
| source_test | collision | raw | mean_logit | 1.000000 ± 0.000000 | NA |
| source_test | collision | raw | mean_probability | 0.999663 ± 0.000461 | NA |
| source_test | correct | l2 | mean_logit | 0.999547 ± 0.000146 | 0.999555 ± 0.000143 |
| source_test | correct | l2 | mean_probability | 0.999545 ± 0.000146 | 0.999552 ± 0.000143 |
| source_test | correct | raw | mean_logit | 0.999575 ± 0.000163 | 0.999582 ± 0.000159 |
| source_test | correct | raw | mean_probability | 0.999508 ± 0.000153 | 0.999516 ± 0.000149 |
| target_full | collision | l2 | mean_logit | 0.338014 ± 0.095109 | NA |
| target_full | collision | l2 | mean_probability | 0.337981 ± 0.095075 | NA |
| target_full | collision | raw | mean_logit | 0.272253 ± 0.063697 | NA |
| target_full | collision | raw | mean_probability | 0.272057 ± 0.064835 | NA |
| target_full | correct | l2 | mean_logit | 0.409049 ± 0.117439 | 0.405114 ± 0.115057 |
| target_full | correct | l2 | mean_probability | 0.409080 ± 0.117423 | 0.405144 ± 0.115042 |
| target_full | correct | raw | mean_logit | 0.296297 ± 0.039619 | 0.291299 ± 0.036949 |
| target_full | correct | raw | mean_probability | 0.303797 ± 0.038961 | 0.298783 ± 0.036427 |

## 同种子归一化效应 / Within-seed normalization effects

Δ为L2减Raw的pooled AUC差；交互为collision效应减correct效应。翻转按同seed两效应异号计数，不以均值替代，分母为可评估seed数。 / Δ is the L2-minus-Raw pooled AUC difference; interaction is the collision effect minus the correct-grouping effect. Reversals compare signs within the same seed, with the evaluable-seed denominator shown.

| Cohort | Aggregation | Δ correct | Δ collision | Interaction | Seed reversals |
|---|---|---:|---:|---:|---:|
| source_test | mean_logit | -0.000028 ± 0.000031 | -0.000015 ± 0.000023 | 0.000012 ± 0.000035 | 0/5 |
| source_test | mean_probability | 0.000037 ± 0.000025 | 0.000321 ± 0.000440 | 0.000285 ± 0.000422 | 0/5 |
| target_full | mean_logit | 0.112753 ± 0.112443 | 0.065760 ± 0.092772 | -0.046993 ± 0.066354 | 0/5 |
| target_full | mean_probability | 0.105283 ± 0.115741 | 0.065923 ± 0.095077 | -0.039360 ± 0.066602 | 0/5 |

## 固定TEST的校准操作点 / Calibration operating points on the fixed TEST

以下为预定主预算n=20与源真实池阈值比较；其他预算、默认操作点、逐seed值、oracle和逐seed TEST区间保留在原始完整汇总及绑定文件。阈值决策采用声明的FP64 score域；此表不提供总体风险保证。 / The prespecified n=20 budget is compared with source-real calibration. Other budgets, default points, per-seed estimates, oracle diagnostics and conditional TEST intervals remain in the full bound artifacts. Decisions use the declared FP64 score space; no population risk guarantee is inferred.

| Representation | Aggregation | Calibration | n | FPR | TPR | Pooled BA |
|---|---|---|---:|---:|---:|---:|
| l2 | mean_logit | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| l2 | mean_logit | target_real | 20 | 0.065371 ± 0.028500 | 0.073715 ± 0.043780 | 0.504172 ± 0.019343 |
| l2 | mean_probability | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| l2 | mean_probability | target_real | 20 | 0.065371 ± 0.028500 | 0.073712 ± 0.043776 | 0.504171 ± 0.019340 |
| raw | mean_logit | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| raw | mean_logit | target_real | 20 | 0.091969 ± 0.031114 | 0.036134 ± 0.011287 | 0.472082 ± 0.011994 |
| raw | mean_probability | source_real | 140 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.500000 ± 0.000000 |
| raw | mean_probability | target_real | 20 | 0.091629 ± 0.029570 | 0.036689 ± 0.010932 | 0.472530 ± 0.011497 |

## 结果段落准备 / Results paragraph preparation

在CLIP五个正式终点模型中，正确分组mean-logit的目标域macro AUC由Raw的0.291299 ± 0.036949变化为L2的0.405114 ± 0.115057。相应pooled AUC归一化效应在correct与collision分组分别为0.112753 ± 0.112443和0.065760 ± 0.092772，同seed符号翻转为0/5。这些观察保留了较低的跨域检测性能，也未在该配置下支持归一化效应异号这一现象。Raw n=20目标真实校准后的固定TEST FPR为0.091969 ± 0.031114，TPR为0.036134 ± 0.011287，pooled BA为0.472082 ± 0.011994；操作点变化不等于排序恢复。跨骨干/H结论须等待相应已注册实验。

Across the five formal CLIP endpoints, target_full macro AUC under correct grouping and mean-logit aggregation changed from 0.291299 ± 0.036949 for Raw to 0.405114 ± 0.115057 for L2. The corresponding pooled-AUC normalization effects were 0.112753 ± 0.112443 under correct grouping and 0.065760 ± 0.092772 under collision grouping, with 0/5 within-seed sign reversals. The low cross-domain performance is retained; this configuration does not support an opposite-sign normalization effect across grouping rules. With Raw and the n=20 target-real calibration budget, fixed-TEST FPR, TPR and pooled BA were 0.091969 ± 0.031114, 0.036134 ± 0.011287 and 0.472082 ± 0.011994. A changed operating point does not establish restored ranking. Cross-backbone and H conclusions require the remaining registered experiments.

## 来源 / Provenance

- Production summary: `/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/analysis/D/clip_vit_b16/five_seed_summary`; completion SHA256 `7b4d811bbe7f946c1d02011d86bf674027808f131ff23c9cc6071c4af3fe7add`.
- Independent audit: `/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/clip_vit_b16_five_seed_summary_independent_acceptance_20261024T1630.json`; SHA256 `a459978674827294f73e524e56a1c04518cb12a04b51eb5c8aae1f242576f6e9`; 12 tables, 48,476 fields; maximum difference 4.55e-13.
- Original per-seed bootstrap intervals remain bound in supporting_artifacts.json; their endpoints are not averaged across seeds.
