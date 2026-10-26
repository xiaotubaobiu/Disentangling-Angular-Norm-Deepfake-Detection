# AntifakePrompt Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. No subagents permitted by user.

**Goal:** 微调并评估同一 AntifakePrompt 骨干上的官方词表评分与共享标量候选评分，回答第一阶段架构假设。

**Architecture:** 固定上游 AntifakePrompt 提交，复用模型实现，在独立实验中实现最小评分适配。共享数据清单和语言池；两个软提示可训练，scalar 另训练一个共享读出头。

**Tech Stack:** Python 3.11, PyTorch 2.6 FP16, transformers 4.28.1, upstream LAVIS, pandas/sklearn.

**Spec:** `docs/superpowers/specs/2026-09-28-antifakeprompt-design.md`

## Global Constraints

- 单智能体；仅 GPU 0–3；仅 FF++/CDF；旧实验原路径保留。
- 第一阶段 seed42、3 epochs×4096 例、有效 batch24；不使用测试集选模。
- 新实验根 `experiments/2026-09-28_e4_antifakeprompt`，下文路径均相对该目录。
- A0/A1/A2 初始 backbone 与 prompt 相同；固定提交与下载文件 SHA256。
- 不触碰主仓库既有未提交工作；只保存本任务新增文件。

## Review Focus

1. token loss 是否泄露问题或 soft prompt 标签：手算 loss fixture 与 upstream rank 对照。
2. 冻结权重是否阻断 prompt 梯度：小型真实网络反传及7B smoke。
3. 候选 padding/顺序导致错误：混合2/3候选及左/右padding测试。
4. 重复视频、真实视频跨方法引用：清单去重与跨划分交集必须为0。
5. 训练恢复/下载中断伪装完成：原子文件、状态和sha检查，smoke不可汇报。

### Task 1: 可重现上游加载与旧实验归档

**Files:** `upstream/`, `requirements.txt`, `scripts/prepare_assets.py`, `results/assets/manifest.json`, `ARCHIVE_INDEX.md`, `results/archive_manifest.json`.

**Interfaces:** 输出上游固定提交、Vicuna v1.3、EVA、InstructBLIP、官方prompt文件路径与hash；环境 `.venv`。

- [ ] 克隆上游到独立目录；归档 E2/E3 元信息与关键科学产物hash，不移动。
- [ ] 下载并校验公开权重，独立环境固定 transformers4.28.1。
- [ ] 验证：加载官方模型到 GPU0，跑1张真实图并记录参数/显存；预期有限分数及yes/no排名。

### Task 2: 两种评分接口及等价性

**Files:** `src/antifake_model.py`, `src/upstream_loader.py`, `tests/test_scoring.py`, `scripts/smoke_model.py`.

**Interfaces:** `AntifakeScorer(mode, assets).forward(images, questions, candidates, mask)` 输出 B×K logits；`training_loss(...,targets)` 输出标量；`trainable_state_dict()` 只含训练权重。

- [ ] 先写并运行关键失败测试：padding、mask、候选独立换序、mean token NLL、softprompt梯度。
- [ ] 实现最小适配，lm 模式调用原词表头、scalar 模式只调用语言 Transformer 后共享头。
- [ ] 在CPU小模型上通过测试；真实GPU模型A0 logits排序与upstream predict_class相同。
- [ ] 真实反向 smoke 证实prompt与head有限非零梯度、冻结参数无梯度。

### Task 3: 数据、训练与评估运行器

**Files:** `src/experiment_data.py`, `scripts/prepare_protocol.py`, `scripts/train.py`, `scripts/evaluate.py`, `configs/stage1.yaml`, `tests/test_protocol.py`.

**Interfaces:** 不可变train/val/test记录清单；共享原有 semantic dataset/evaluation 规范。训练输出最佳可训练state、逐epoch日志、配置与清单hash。评估输出逐视频预测、指标与覆盖审计。

- [ ] 测试数据去重、跨划分0重叠、GPU白名单和非smoke完整性拒绝。
- [ ] 按spec构造清单与预算，训练前冻结；实现训练/恢复和两域评估。
- [ ] 端到端少量真实帧 smoke；正式A1/A2顺序或独立GPU运行，A0评估同清单。

### Task 4: 配对判定与第一阶段报告

**Files:** `scripts/report.py`, `reports/stage1_report_zh.md`, `PROGRESS.md`, `results/summary/`.

**Interfaces:** 读取Task3逐视频预测，用10000次配对bootstrap产生两域检测/语义表、门槛和限制。

- [ ] 检查所有要求组、视频覆盖、种子/训练状态/退出码及hash。
- [ ] 汇总T1–T5和A2−A1，不把A0官方来源准确率当作本地基线。
- [ ] 自审代码与科学归因；用户禁止子智能体，所以无外部reviewer。
- [ ] 第一阶段报告完成后停止，不开始模型压缩/RL。

## 执行记录

设计与计划依据已授权goal继续实施；用户可随时纠正。未对旧未提交文件进行提交/重置。进度以实验PROGRESS及实际命令结果为准，不能凭计划勾选声称完成。
