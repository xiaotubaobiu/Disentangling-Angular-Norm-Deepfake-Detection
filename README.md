# Deepfake Detector：期刊实验主线

本仓库用于正在执行的期刊实验。D路线研究推理归一化、视频聚合和阈值迁移，H路线研究完整人脸预处理历史。正式范围为D的15个模型＋已触发扩展的H的18个模型，共33个，完整结果仍在生成。

```text
src/journal_detector/      当前训练、数据准备、推理与评价代码
configs/                   基础模板、公开权重信息和可重建的数据划分定义
scripts/                   开发预算注册、验收及受控运行入口
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
| D最大预算，已冻结 | 2 | 30 | 2 |
| H最大预算 | 10，已冻结 | 待源域开发验收 | 15，已冻结 |
| 训练输出L2 | 关闭 | 关闭 | 关闭 |

共用 AdamW、CE、WD=5e-4、AMP、梯度裁剪1、epoch级warmup1＋cosine、EMA=.999。每轮使用EMA计算源DEV正确视频分组的mean-probability pooled AUC，严格提高才保存；连续5轮不提高早停。所有测试加载最佳EMA `best_model.pth`。

开发种子为1007、1042。D利用源DEV和隔离CDF-DEV开发总预算，正式种子7/42/123/2026/3407的训练轨迹只按源DEV选点。H只使用源DEV：在共同B输入上，以B/AB两模型平均AUC选择共同epoch和共同早停，正式种子7/42/123。H-DINO的30轮源学习仍不足，当前正在执行从公开初始化重新开始的60轮配对开发，正式预算尚未冻结。最大预算、实际停止轮数和最佳轮数分别记录。

使用DF40官方图像增强，随后使用各骨干对应的预训练输入归一化。DINOv2保留官方模型、权重与插值公式；位置插值在CPU执行并保留梯度，以支持PyTorch确定性反向。

数据与评价沿用最新21方法的FS9＋FR12协议：每自然视频8帧、独立源DEV、RGB去重、真实总采样概率.5、伪造每方法.5/21。三骨干×种子7/42/123/2026/3407，共15个模型。八组对照为正确/错误分组×Raw/L2×mean-logit/mean-probability，同时保留阈值迁移、目标真实样本校准、数值审计和五种子汇总。

CDF独立DEV为20真＋420假，21方法各20个伪造；关联闭包共890个候选视频，全部排除在正式排序和阈值校准池之外。正式排序队列157真＋12,037假；阈值实验单独使用64真CAL与93真＋12,037假固定TEST。关联规则、清单哈希和适用边界见使用说明。

H使用FF++ c23原视频训练/源评价，DFD c23跨域测试。B是dlib 68点对齐224；A是RetinaFace 5点对齐336；A→B在保存的A图像上重新完整运行B。B和A→B模型共享初始化、母帧、抽样及增强随机键，完成四格交叉评价，并报告处理失败率和共同成功队列。

## 操作

```bash
python run.py --help
python run.py status
python -m pytest tests -q
```

数据准备、预训练权重位置与逐步命令见 [使用说明](docs/期刊实验代码说明.md)，运行时线程和完整校验调度见[运行时校验](docs/execution/runtime-validation.md)。实际AMP batch128与恢复一致性已通过GPU资格检查。

本轮协议为`journal_detector-target-dev-20261102-v2`。`configs/journal.json`和`run.py`保留原15轮基础模板入口；本轮实际开发、epoch冻结和正式配置位于独立产物根，操作本轮运行时使用说明中的直接模块接口。本机产物根为`/home/z/workspace/deepfake_journal_20261102`，仅使用物理GPU0–3。

当前代码不依赖历史实验的本地权重、预测缓存或结果目录。原始数据保留在 `/Dataset`；数据划分定义保存在 `configs/data_protocol/`，以后可以重新审计并构建清单。

## 实验沿革与文档

共三阶段已完成训练：会议阶段、31方法阶段、21方法阶段；本主线是正在执行的期刊实验，完整正式结果待验收。历史代码目录仅供查阅，当前入口不会导入它们。完整历史提交与原始代码仍可通过归档标签恢复。

Obsidian主目录 `paper/deepfake-detector/` 按阶段保留：

- `31方法实验/`：实验计划、实验记录、实验结果、中文稿。
- `21 方法实验/`：实验计划、实验记录、实验结果、中文稿。
- `期刊实验_20261102/`：本轮实验计划、实验记录、实验结果、中文稿。

其中“实验结果”保留关键数值，“实验记录”区分观测与原因解释，“中文稿”保存各阶段可用的论文叙事。旧实验结果不混作下一轮结果。
