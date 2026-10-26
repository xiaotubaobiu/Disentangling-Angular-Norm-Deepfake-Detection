# 深度伪造检测：两篇论文工作区

日常阅读与运行从下面两个独立项目进入。两边各有真实的代码、配置、环境说明、测试和实验产物，可以单独搬走使用。

| 项目 | 研究范围 | 入口 |
|---|---|---|
| 论文一：特征几何、池化与阈值迁移 | 方向/长度、视频聚合、跨域阈值；2026-05 至 07 月的五个阶段 | [feature_geometry](projects/feature_geometry/README.md) |
| 论文二：Visual-Jev | H1 固定候选共享评分器、E2 语义决策验证；汇报 FF++、CDF | [visual_jev](projects/visual_jev/README.md) |

## 当前结果入口

- 旧论文：[阈值迁移最终报告](projects/feature_geometry/experiments/2026-07-30_threshold_transfer/reports/threshold_transfer_final_report.md)、[池化与饱和报告](projects/feature_geometry/experiments/2026-07-27_pooling_saturation/reports/pooling_saturation_final_report.md)。
- 新论文：[H1 报告](projects/visual_jev/experiments/2026-09-21_visual_jev_h1/reports/h1_validation_result.md)、[E2 最新中文汇报](projects/visual_jev/experiments/2026-09-22_semantic_visual_jev_e2/reports/e2_validation_summary_zh.md)。E2 三种子 FF++/CDF 42 组已完成，冻结总裁决为 `not_supported`。

## 使用方式

分别进入项目目录，按各自 `docs/ENVIRONMENT.md` 创建独立 Python 环境。两者保留相同历史包名，不能同时安装到一个环境。

```bash
cd projects/feature_geometry
python scripts/check_environment.py
python scripts/replay_threshold.py
```

```bash
cd projects/visual_jev
python scripts/check_environment.py
python scripts/replay_e2.py
```

以上命令分别从本工作区根目录开始；回放结果写到各项目的 `runs/`。旧论文为 CPU 小规模管线验证，新论文为冻结主域结果的 CPU 完整门槛回放与审计。

## 重合部分和历史材料

[shared/README.md](shared/README.md) 登记共同代码、外部数据和基线资产。代码在两边都有本地快照，运行不依赖父工作区或兄弟项目；图像数据仍使用外部数据目录。只克隆 Git 源码时，大型模型与预测产物需另外传输。

根目录原有的 `src/`、`configs/`、`tests/`、`experiments/` 保留为历史工作区，继续维持已完成实验的原路径。新研究在 `projects/` 内进行。旧导航与仓库整理文档见 [organization/](organization/README.md)，本次迁移来源可在各项目 `provenance/` 查看。

整理已通过测试、结果回放与哈希核验，详见 [拆分验收记录](organization/VALIDATION.md)。
