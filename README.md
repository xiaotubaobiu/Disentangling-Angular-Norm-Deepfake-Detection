# Deepfake Detector：期刊实验主线

本仓库用于下一轮期刊实验的讨论与执行。当前代码尚未启动新一轮训练。

```text
src/journal_detector/      当前训练、数据准备、推理与评价代码
configs/                   当前15个运行配置和可重建的数据划分定义
tests/                     当前代码测试
third_party/               固定版本的官方DINOv2、DF40参考实现
legacy/31_methods/         第二阶段31方法实验的历史代码
legacy/21_methods/         第三阶段21方法实验的历史代码
docs/                     使用说明、版本沿革与核验记录
run.py                     统一命令入口
requirements.lock.txt      Python依赖版本
```

会议原版单独保存在 [`codex/conference-original`](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection/tree/codex/conference-original)，不在main重复保存。

## 当前配方

| 项目 | CLIP | DINOv2 | EfficientNet |
|---|---|---|---|
| 官方骨干 | OpenAI ViT-B/16 | Meta ViT-B/14 | ImageNet B0 |
| 特征维度 | 512 | 768 | 1280 |
| 学习率 | 1e-5 | 5e-5 | 1e-4 |
| 微调范围 | 视觉编码器全部参数＋二分类头 | 同左 | 同左 |
| 每卡实际batch | 128 | 128 | 128 |
| 最大轮数 | 15 | 15 | 15 |
| 训练输出L2 | 关闭 | 关闭 | 关闭 |

共用 AdamW、CE、WD=5e-4、AMP、梯度裁剪1、epoch级warmup1＋cosine、EMA=.999。每轮使用EMA计算源DEV正确视频分组的mean-probability pooled AUC，严格提高才保存；连续5轮不提高早停。所有测试加载最佳EMA `best_model.pth`。

使用DF40官方图像增强，随后使用各骨干对应的预训练输入归一化。DINOv2保留官方模型、权重与插值公式；位置插值在CPU执行并保留梯度，以支持PyTorch确定性反向。

数据与评价沿用最新21方法的FS9＋FR12协议：每自然视频8帧、独立源DEV、RGB去重、真实总采样概率.5、伪造每方法.5/21。三骨干×种子7/42/123/2026/3407，共15个模型。八组对照为正确/错误分组×Raw/L2×mean-logit/mean-probability，同时保留阈值迁移、目标真实样本校准、数值审计和五种子汇总。

## 操作

```bash
python run.py --help
python run.py status
python -m pytest tests -q
```

数据准备、预训练权重位置与逐步命令见 [使用说明](docs/期刊实验代码说明.md)。`prepare`只处理数据和配置；`train`、`extract`必须显式执行。正式实验尚未运行，实际batch128的GPU显存适配尚未验证。

当前代码不依赖历史实验的本地权重、预测缓存或结果目录。原始数据保留在 `/Dataset`；数据划分定义保存在 `configs/data_protocol/`，以后可以重新审计并构建清单。

## 实验沿革与文档

共三阶段已完成训练：会议阶段、31方法阶段、21方法阶段；本主线是准备中的下一轮，不是第四次已完成实验。历史代码目录仅供查阅，当前入口不会导入它们。完整历史提交与原始代码仍可通过归档标签恢复。

Obsidian主目录 `paper/deepfake-detector/` 只有：

- `31方法实验/`：实验计划、实验记录、实验结果、中文稿。
- `21 方法实验/`：实验计划、实验记录、实验结果、中文稿。

其中“实验结果”保留关键数值，“实验记录”区分观测与原因解释，“中文稿”保存各阶段可用的论文叙事。旧实验结果不混作下一轮结果。
