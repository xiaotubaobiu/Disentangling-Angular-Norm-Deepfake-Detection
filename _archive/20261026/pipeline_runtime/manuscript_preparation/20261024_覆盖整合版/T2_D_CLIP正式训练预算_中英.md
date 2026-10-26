# T2-D/CLIP：正式训练预算 / Formal training budgets

本子表报告D路线五个已完成CLIP模型的实际训练计数。主终点均为冻结最后轮；源域最佳权重另存，不能按目标结果替代。其他骨干及H矩阵、正式特征缓存和目标测试统计仍待完成；本表不报告检测性能。

This subtable reports actual training counts for the five completed D-branch CLIP models. Every primary endpoint is the frozen last epoch; source-best weights are separate and cannot be substituted using target outcomes. Other backbones, H, formal feature caches and target-test statistics remain incomplete. No detector performance is reported.

| Seed | Epochs | Attempts | Successful updates | AMP skips |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 30 | 23970 | 23965 | 5 |
| 42 | 30 | 23970 | 23965 | 5 |
| 123 | 30 | 23970 | 23966 | 4 |
| 2026 | 30 | 23970 | 23965 | 5 |
| 3407 | 30 | 23970 | 23967 | 3 |

每个模型保持23,970次尝试，不补齐因AMP跳步而减少的成功更新。初始权重、数据清单、配置、终态权重与source-best绑定见机器可读索引。30轮工件及源域DEV保存分数的独立复算已验收；这不等于目标域结果已完成。

Each run retains 23,970 attempts without makeup updates after AMP skips. Initialization, manifests, configuration, final weights and source-best bindings are recorded in the machine-readable index. Thirty-epoch artifacts and independent recomputation of saved source-DEV scores are accepted; this does not establish completed target-domain results.
