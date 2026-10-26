# T2-D：已验收正式训练预算 / Accepted formal training budgets

本表涵盖全部15个D正式模型：CLIP、DINOv2和EfficientNet各五个种子。各模型完成30轮，主终点为冻结最后轮；源域最佳权重单独保留。表中仅为训练计数，D真实缓存/分析及H的30个正式模型仍待完成。

This table covers all fifteen D formal models, with five seeds for each backbone. Each completed thirty epochs with the frozen last epoch as primary endpoint; source-best weights are retained separately. These are training counts. Real D caches and analyses, and all thirty H formal models, remain pending.

| Backbone | Seed | Epochs | Attempts | Successful updates | AMP skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP | 7 | 30 | 23970 | 23965 | 5 |
| CLIP | 42 | 30 | 23970 | 23965 | 5 |
| CLIP | 123 | 30 | 23970 | 23966 | 4 |
| CLIP | 2026 | 30 | 23970 | 23965 | 5 |
| CLIP | 3407 | 30 | 23970 | 23967 | 3 |
| DINOv2 | 7 | 30 | 23970 | 23968 | 2 |
| DINOv2 | 42 | 30 | 23970 | 23966 | 4 |
| DINOv2 | 123 | 30 | 23970 | 23966 | 4 |
| DINOv2 | 2026 | 30 | 23970 | 23967 | 3 |
| DINOv2 | 3407 | 30 | 23970 | 23967 | 3 |
| EfficientNet | 7 | 30 | 23970 | 23965 | 5 |
| EfficientNet | 42 | 30 | 23970 | 23966 | 4 |
| EfficientNet | 123 | 30 | 23970 | 23967 | 3 |
| EfficientNet | 2026 | 30 | 23970 | 23966 | 4 |
| EfficientNet | 3407 | 30 | 23970 | 23967 | 3 |

十五项合计359,550次尝试、359,493次成功更新和57次AMP跳步。每模型保留23,970次固定尝试，不补齐跳步。D开发另有10个完成模型，不混入本正式预算合计。身份、终态权重、source-best及源DEV独立复算依据见配套机器索引。训练完成不等于D分析或全部研究完成；T3–T8、最终跨种子性能和结论仍待真实合格缓存与完整矩阵。

Across the fifteen formal models, there were 359,550 attempts, 359,493 successful updates and 57 AMP skips, without makeup steps. Ten D development models are recorded separately. Identity, final-weight, source-best and source DEV recomputation evidence is bound in the accompanying machine index. Completed training is not completed D analysis or the whole study; final tables and conclusions await real eligible caches and the full matrix.
