# 本轮辅助分析与独立核验源码快照

这里逐字节保留本轮实际使用的操作脚本。`source_manifest.json`列出原始产物路径和SHA256；这是来源快照，当前训练/评分入口不会导入这些文件。

| 源码 | 用途与限制 |
|---|---|
| `decompose_grouping_effect_rank_checked.py` | 在保存的400位十进制排序分数上，用精确有理数AUC credit重建正确/碰撞分组，将分组变化拆为权重和组内平均的对称贡献。固定模型的描述性分解。 |
| `extend_grouping_decomposition.py` | 对已验收D模型应用同一分解；完整五种子统计要求原严格五种子汇总存在。 |
| `extend_aggregation_baseline_analysis.py` | 重建Raw/L2×两聚合AUC恒等式及配对排序账本；记录FP64端点和数值负对照，不将端点计数当作Sigmoid饱和的独立因果证明。 |
| `verify_*_committed_analysis.py` | 检查已提交工件、模型/数据/缓存头身份并独立还原保存分数的指标；不替代原严格汇总中的原图/缓存payload完整复核。 |
| `verify_*_strict_summary_tables.py` | 在原严格汇总worker成功后，独立还原官方全部均值、样本标准差、计数、方向和NA。 |
| `verify_formal_*_training.py` | 从训练轨迹、恢复状态、抽样/更新事件及最佳EMA逐张量核验正式权重；完整性通过不等于源验证性能保证。 |

脚本中的产物根固定为实际执行位置`/home/z/workspace/deepfake_journal_20261102`。源数据、清单、特征缓存和回执依赖保留于该根，环境与配方见`configs/experiments/journal_20261102/`。扩展脚本从该根的`operations/`读取原分解函数。搬迁产物应另建重建清单并重新核验，不编辑原身份回执来冒充同一运行。

只在所需原严格汇总及独立验收完成后生成全种子结果；尚未完成的种子不能丢弃或用剩余种子平均。既有输出目录/回执保留，脚本不会以新结果覆盖旧分解目录。Obsidian发布脚本和作业控制器不在此目录，避免将本机文档写入和进程管理混作科学分析接口。
