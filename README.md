# Deepfake Detector — conference21 主线

`main` 是当前统一代码；`codex/conference-original` 保留会议原版提交 `5cc0105477509566295ea6cd3c277cdcf4ede82e`，内容不变。

本次把其余八个分支的**完整提交历史**合并进 main，以最后的 pipeline 代码作为当前工作树。每个旧分支的原始树都有 `archive/pre-main-20261102/<旧分支名>` 标签，可以完整恢复。不同历史项目没有相互覆盖成一套运行代码。对应关系见 [分支归档清单](docs/consolidation/branches-20261102.json)。

## 当前实验

入口：[run.py](run.py)，实现与详细说明：[studies/conference21](studies/conference21/README.md)。

| 设置 | CLIP | DINOv2 | EfficientNet |
|---|---|---|---|
| 骨干 | OpenAI ViT-B/16 | 官方 ViT-B/14 | ImageNet B0 |
| 输出维度 | 512 | 768 | 1280 |
| 学习率 | 1e-5 | 5e-5 | 1e-4 |
| 微调 | 视觉编码器全参数＋二分类头 | 同左 | 同左 |
| 每卡实际 batch | 128 | 128 | 128 |
| 最大 epoch | 15 | 15 | 15 |
| 训练输出 L2 | 关闭 | 关闭 | 关闭 |

训练：AdamW、CE、WD=5e-4、AMP、梯度裁剪1、会议版 epoch warmup1＋cosine、EMA=.999。每轮以 EMA 在源 DEV 上计算正确自然视频分组的 mean-probability **pooled video AUC**；严格提高才保存，连续5轮不提高早停。最终所有对照都加载 `best_model.pth`，不使用末轮权重。保持会议版的参数 EMA 和 live buffers 行为。

增强：DF40 官方增强（翻转、旋转、模糊、插值分支、颜色、JPEG40–100）；之后使用各骨干预训练对应的归一化。DINOv2 使用官方实现与官方权重；位置插值在 CPU 执行以支持确定性反向，保持官方公式和梯度。

数据：沿用最新21方法版本的 FS9＋FR12、源 TRAIN/DEV 划分、每自然视频8个不同RGB帧、重复RGB合并与加权替换采样。真实总概率.5，每种伪造方法.5/21。概率均衡不保证每个 epoch 的实际数量完全相等。实验种子为7、42、123、2026、3407，共15个模型。

对照：正确/错误分组 × Raw/L2 × mean-logit/mean-probability，共8组；错误分组将同 bare video ID 的各方法8帧合并，k方法即8k帧。保留 pooled/macro AUC、分组差值、数值分析、源域阈值迁移与目标真实样本校准，以及5种子汇总。H输入历史实验和旧31类实验保留在历史实现，不混入当前21类训练。

## 使用

本机独立环境：`/home/z/workspace/deepfake_conference21_environment_20261102/bin/python`。
独立输出：`/home/z/workspace/deepfake_conference21_20261102`。

```bash
# 仅准备清单和15个运行配置，不训练
python run.py prepare
python run.py status
# 指定0–3中的一张空闲GPU的UUID，再显式启动一个模型
CUDA_VISIBLE_DEVICES=GPU_UUID python run.py train --run-id D_formal_clip_vit_b16_seed7
# 训练完成后，统一加载该模型的最佳EMA做特征缓存和8组评价
CUDA_VISIBLE_DEVICES=GPU_UUID python run.py extract --run-id D_formal_clip_vit_b16_seed7
python run.py analyze --run-id D_formal_clip_vit_b16_seed7
python run.py summarize --backbone clip_vit_b16
```

GPU使用沿用0–3白名单和独占锁。每卡128是实际前向batch，不自动改成32＋梯度累计。若显存不足会失败，不能据此声称已完成batch128运行资格验证。

**交付状态：代码整合和CPU验证；新协议的正式训练、目标推理均未启动。** 旧表格不能当作本次训练结果。

## 保留的历史

- `studies/conference_original_snapshot/`：此前根目录代码快照；原会议版本的权威入口是独立会议分支。
- `studies/pipeline_reliability/`：此前31方法与21方法/H实验代码、计划及分析工具。
- `_archive/20261026/`：既有归档，不改写旧实验结果。
- 其他原始项目：通过分支清单中的归档标签完整访问；全部八个提交均为main祖先。
