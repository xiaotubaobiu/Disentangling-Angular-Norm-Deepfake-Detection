# 历史独立核验与导出脚本

这些脚本是本轮实际使用的operator工具，按原始字节归档；主科学入口仍是项目根目录run.py。它们不改变src/config/冻结输入，不替代正式训练管线。验收和导出脚本的SHA与final_independent_acceptance.json、final_report/supplemental_export_receipt.json逐项匹配。

脚本中的绝对路径指向本轮已完成目录，部分会写自身receipt或校验标记。当前完成目录不需要再次执行这些脚本。未来在隔离副本中复核时，先阅读脚本并显式改为新项目/输出路径；历史验收SHA对应的是这里归档的原字节，不冒充修改后运行。导出脚本保留原图，补充全范围ECDF与配对差，所有主分数/阈值均只读取。

主要范围：completed_model_verification（final/full Opt/AMP/RNG与225抽样数组）、completed_cache_verification（15×四角色块hash/清单逻辑全值）、bootstrap_support_verification（声明group_id抽样支持）、final_independent_acceptance（主流程重算、校准重建、bootstrap数组摘要/支持/配对代数、机制控制、匹配头分数、服务实际健康）、supplemental_exports（补充图表）、human_handoff（由接受的表生成结论与15实际配置）、full_readonly_progress（进程/锁/安全服务字段只读快照）。

所有2000项测试bootstrap AUC没有另一路逐项从score重算；保存数组、摘要、无效支持与配对代数已独立核对。最终独立验收不扩大未知祖先认证、TPR95支持或逐step AMP事件日志范围。
