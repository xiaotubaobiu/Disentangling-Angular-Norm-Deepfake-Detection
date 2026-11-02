# Deepfake Pipeline Reliability — 计划与既有实验归档

本分支保存2026-10-23的新研究计划，以及当前服务器已完成的31方法实验源码和结果证据。**新21方法/原视频预处理实验尚未开始；45个正式模型尚未训练。**

- [本机适配后的完整新实验计划](studies/pipeline_reliability/docs/plans/2026-10-23-pipeline-reliability-v2.md)：D路线15模型＋H路线30模型，源域开发另计，未来仅使用GPU0–3。
- [既有实验完整归档](studies/pipeline_reliability/docs/archive/2026-10-23-previous-experiments.md)：旧31方法15模型、八格AUC大表、阈值迁移、目标真实校准、已完成/未完成范围。
- [本机环境与材料证据](studies/pipeline_reliability/docs/plans/2026-10-23-pipeline-reliability-evidence.md)与[设计审查](studies/pipeline_reliability/docs/plans/2026-10-23-pipeline-reliability-review.md)。
- [当前已完成实验的源码](studies/pipeline_reliability/src/deepfake_detector/)与[测试](studies/pipeline_reliability/tests/)。这是31方法协议的实现，不是已接入的新45模型runner；旧配置使用GPU4–7，不可直接启动来执行新计划。
- [新增八格分析代码和结果](studies/pipeline_reliability/reports/video_grouping_8f_20261023/)与[目标真实校准代码和结果](studies/pipeline_reliability/reports/cdf_real_calibration_20261022_v1/)。
- [收到的五份原始设计文档](studies/pipeline_reliability/docs/reference_materials/2026-10-23/)；文档提及的独立参考源码和45份YAML未随本次附件提供。
- [快照文件清单与SHA256](studies/pipeline_reliability/SNAPSHOT_MANIFEST.json)。模型、原片、人脸图像和全量特征缓存继续保存在服务器，未上传到本分支。

## 与原仓库的关系

原master提交 `5cc0105477509566295ea6cd3c277cdcf4ede82e` 保持原样。根目录的旧训练/评价代码仍作为历史实现保留；其方法ID遗漏、训练/验证混合抽帧和测试概率合并问题见归档第12节。原README另保存在[历史说明](studies/pipeline_reliability/legacy_repository_readme.md)。其原有性能表述不作为新研究已验证结论。

当前工作区快照来自服务器 `deepfake_detector`，基础提交 `674fa4a362f2b7ce6382992238a9b975e7f9fe0a`，并纳入之后完成的两项CPU分析和本次文档。新增研究包位于 `studies/pipeline_reliability/`，未覆盖原master或删除历史提交。本分支只用于归档和审阅新方案，不自动运行任何实验。
