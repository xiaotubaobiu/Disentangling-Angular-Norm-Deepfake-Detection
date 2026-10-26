# T2-D：已验收正式训练预算 / Accepted formal training budgets

本表涵盖已独立验收的五个CLIP、四个DINOv2和两个EfficientNet正式模型。各模型均完成30轮，主终点为冻结最后轮；源域最佳轮权重单独保存。仅列训练计数，不包含检测性能。一个DINOv2、三个EfficientNet及全部H正式模型仍未完成。

This table covers five CLIP, four DINOv2 and two EfficientNet formal models with independently accepted thirty-epoch terminal artifacts. The primary endpoint is the frozen last epoch; source-best weights are retained separately. These are training counts without detector-performance results. One DINOv2, three EfficientNet and all H formal models remain incomplete.

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
| EfficientNet | 7 | 30 | 23970 | 23965 | 5 |
| EfficientNet | 42 | 30 | 23970 | 23966 | 4 |

十一项合计263,670次尝试、263,626次成功更新和44次AMP跳步。每模型保留23,970次固定尝试，不补齐跳步。身份、终态权重、source-best及源域独立复算依据见配套机器索引。T3–T8、跨种子性能统计及研究结论仍待完整矩阵、真实缓存和独立分析。

Across these eleven models, there were 263,670 attempts, 263,626 successful updates and 44 AMP skips. Each model retains its fixed 23,970-attempt budget without makeup steps. Identity, final-weight, source-best and source-domain recomputation evidence is bound in the accompanying machine-readable index. Target-domain tables, seed-level performance summaries and study-wide conclusions await the full matrix, real caches and independent analyses.
