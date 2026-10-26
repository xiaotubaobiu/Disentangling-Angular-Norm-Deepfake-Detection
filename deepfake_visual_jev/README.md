# 论文二：Visual-Jev 假设验证

研究问题：在冻结视觉表征下，候选条件共享评分器能否保持检测能力（H1），进一步加入问句、候选和多任务训练后能否形成可迁移的语义决策能力（E2）。本项目可独立安装和运行，只汇报 FF++、CDF。

## 先读什么

1. [E2 最新中文汇报](experiments/2026-09-22_semantic_visual_jev_e2/reports/e2_validation_summary_zh.md)：三种子 × 两主域 × 七条件，共 42 组，冻结结论 `not_supported`；G3 通过，G1/G2/G4/G5/G6 未通过。
2. [协议审查及局限](experiments/2026-09-22_semantic_visual_jev_e2/reports/protocol_audit_zh.md)：区分否定当前实现与否定整个研究方向。
3. [H1 验证报告](experiments/2026-09-21_visual_jev_h1/reports/h1_validation_result.md)：支持固定候选接口的非劣替换，不构成开放语义推理证明。
4. [H1 设计](docs/superpowers/specs/2026-09-21-visual-jev-h1-design.md) · [E2 设计](docs/superpowers/specs/2026-09-22-e2-semantic-visual-jev-design.md)。

E2 固定问答检测 AUROC：FF++ H1/E2 = 0.998531 / 0.998487；CDF = 0.911267 / 0.822181。研究写作必须保留这个跨域退化，不能仅凭候选置换通过宣称获得语义推理能力。

## 使用

[安装与环境](docs/ENVIRONMENT.md)。在本项目根目录：

```bash
python scripts/check_environment.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/replay_e2.py
```

回放只使用 CPU 和已有模型/预测文件：先验证输入字节并重定位路径元数据，再按冻结配置执行 10,000 次 bootstrap 门槛评估及 FF++/CDF 独立审计，输出到新的 `runs/e2_replay_*/`。完整指标、置信区间和结论应与冻结报告一致。

项目整体移动后也执行同一命令。原始绝对路径元数据保存在 `provenance/path_metadata_original/`，迁移日志在 `provenance/path_relocations.json`。模型、预测、语言表、锁定配置和 E2 评估源码不因迁移改变。

H1 训练入口为 `scripts/train_h1.py`，例如在为新实验另存配置与输出路径后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_h1.py --config <新实验的H1配置.yaml>
```

E2 训练/图像推理入口在 [experiments/2026-09-22_semantic_visual_jev_e2/scripts/](experiments/2026-09-22_semantic_visual_jev_e2/scripts)，原步骤见 [实验记录](experiments/2026-09-22_semantic_visual_jev_e2/README.md)。新训练须另存配置和输出目录；复制的冻结检查点无需重训。历史配置仍登记 DFD，三域缓存清单是原来源指纹的一部分；本项目默认回放只裁决 FF++、CDF，不运行 DFD 评估。

## 目录和边界

- `src/models/clip_visual_jev.py` 与 `clip_semantic_jev.py`：H1、E2 模型。
- `src/data/semantic_*`、`src/engine/semantic_trainer.py`、`src/evaluation/semantic_*` 和 `e2_*`：语言数据、训练、对照和裁决。
- `experiments/2026-09-21_visual_jev_h1`、`experiments/2026-09-22_semantic_visual_jev_e2`：正式实验与历史尝试。最终结果入口固定为 E2 `results/evaluation_primary_v2` 和 `results/summary`，旧 `evaluation`、smoke 目录只作历史追溯。
- `experiments/2026-05-05_paper_feature_geometry`：仅为 H1 复制的单个基线资产，[用途说明](experiments/2026-05-05_paper_feature_geometry/README.md)。保留原相对路径以保持配置哈希。
- `provenance/`：输入哈希、共同模块来源、环境、数据及路径迁移记录。
- `runs/`：新的回放输出。

基础模块是本项目自己的代码副本，没有运行时跨项目导入。两个项目使用不同 Python 环境。H1 的 `export_cached_baseline.py` 是历史缓存转换工具，需要显式提供逐帧 Parquet；日常评估可直接使用本地基线检查点与 H1 `evaluate_model.py`，已有基线预测也已随项目复制。
