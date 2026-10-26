# 论文一：特征几何、池化与阈值迁移

研究问题：检测器的特征方向、特征长度、视频聚合和决策阈值分别怎样影响跨域结果。本项目含自己的代码、配置、测试及五阶段完整本地实验产物，可以整体搬走单独运行。

## 先读什么

| 顺序 | 内容 | 入口 |
|---|---|---|
| 1 | 当前结论：CDF 中方向优势的解释及阈值失效 | [最终阈值迁移报告](experiments/2026-07-30_threshold_transfer/reports/threshold_transfer_final_report.md) |
| 2 | 池化与 sigmoid 饱和效应 | [池化报告](experiments/2026-07-27_pooling_saturation/reports/pooling_saturation_final_report.md) |
| 3 | 特征假设与同帧位移审查 | [四个问题报告](experiments/2026-07-23_four_questions/reports/four_questions_final_report.md) |
| 4 | 分组口径修正与最初实验 | [原始结果修正](experiments/2026-05-05_paper_feature_geometry/reports/final_results_notes.md) |
| 5 | 实验全目录与原有使用说明 | [阶段索引](experiments/README.md) · [历史说明](docs/legacy_workspace_readme.md) |

旧结果中按裸 `video_id` 分组的数字已被后续审查修正。写论文时以带方法标识的视频分组和后续阶段结论为准；不能继续把历史 CDF 方向优势写成已成立的特征几何发现。

## 使用

[安装与环境](docs/ENVIRONMENT.md)。在本项目根目录：

```bash
python scripts/check_environment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/replay_threshold.py
```

最后一条会实际执行七步 CPU 小规模回放（CLIP、seed 42、两种伪造方法），输出到新的 `runs/threshold_*/`，包含表格、图和报告。它验证管线，不代替冻结的全规模论文结果。原五阶段脚本仍按自身相对路径工作。

重新训练入口为 `experiments/2026-05-05_paper_feature_geometry/scripts/train.py --config <配置>`，用 `torchrun` 启动多卡任务；配置和输出目录应为新实验另存。

## 目录和边界

- `src/diagnosis`、`src/study`、`src/study_pooling`、`src/study_threshold`：本论文的分析。
- `src/data`、`src/models`、`src/engine`、`src/evaluation`：本地训练评估基础模块。
- `experiments/`：2026-05-05 至 2026-07-30 的五阶段代码、配置、报告和数据产物。
- `provenance/`：导入文件哈希、共同代码版本、外部数据登记和原环境记录。
- `runs/`：本项目的新回放输出，可与冻结结果分开阅读。

H1/E2 不属于本论文。共同模块在本项目内有自己的快照，运行不依赖另一个项目或父工作区。不要在同一 Python 环境安装两个项目。
