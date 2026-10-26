---
title: Deepfake Detector 既有实验归档（截至2026-10-23）
date: 2026-10-23
status: archived_completed_evidence
tags: [deepfake, DF40, experiment-archive, AUC, calibration]
---

# Deepfake Detector 既有实验归档（截至2026-10-23）

本页整合本机已有代码、固定协议、结果与验收凭据，作为新实验开始前的历史记录。归档过程仅整理文档，没有启动训练、推理或新增科学实验。小型证据保留在本仓库，模型与大规模缓存保留在原本机产物目录；本页不是模型、数据或缓存的完整备份。

**最关键的区分：已经完成的是31种伪造方法训练的15个模型。21方法指旧实验的视频评价子集，不是旧训练方法数。用户2026-10-23提供的新设计要求重新建立21方法训练路线与原视频预处理路线，共45个正式模型；这些新模型及新结果不在本归档中。**

## 1. 已完成、仅有方案、尚未执行

| 工作 | 状态 | 规模与用途 |
|---|---|---|
| DF40 v1.1 原始Raw训练 | 已完成、独立验收通过 | 31方法，3骨干×5种子=15模型，每模型15 epoch |
| 原head的Raw/L2与两种视频聚合 | 已完成 | 正常分组RL/RP/DL/DP；21视频方法与31方法图像评价分开 |
| 冻结特征匹配线性头 | 已完成 | 15个编码器×Raw/方向两头=30头；全部收敛 |
| 固定尺度、公共长度、排序转移与数值诊断 | 已完成 | 5个尺度、4种长度形式、3×3排序账本与数值ties |
| 源真实阈值迁移至FF/CDF | 已完成 | 140个源真实视频独立校准经验FPR≤5%，按流程冻结阈值 |
| CDF有限真实样本阈值校准 | 已完成 | 80视频池，97真实+4132伪造固定测试，200次共同嵌套抽样 |
| 正确/错误分组×Raw/L2×Logit/Prob八格 | 已完成、独立重算通过 | 现有8帧、21视频方法、15模型、2域；240个AUC格，零新训练/推理 |
| 旧16方法、每视频32帧的分组诊断 | 仅方案及预盘点 | 未补推理、未生成正式32帧结果；不得与8帧八格混用 |
| 经典FF++ c23完整视频下载 | 已完成 | 1000真实+四方法各1000伪造，共5000视频；尚未进入上述模型训练 |
| 新附件D路线21方法正式训练 | 尚未执行 | 新15个Raw模型，须重新建立训练/开发与冻结身份 |
| 新附件H路线预处理历史实验 | 尚未执行 | 2种训练历史×3骨干×5种子=30个Raw模型 |

旧结果是回顾性探索/验证证据：研究历史已看过CDF，不称前瞻盲测；所有31方法均参与旧源训练，不声称未见方法泛化。新计划不能把旧表改标签充作新21方法训练结果。

## 2. 旧训练协议与真实数据范围

源数据为DF40衍生FF域，目标为DF40衍生CDF域。它们不是刚下载的经典FF++四方法原视频实验，也不是原始Celeb-DF标准测试协议。

| 类别 | 方法数 | 本地固定方法名 |
|---|---:|---|
| FS（换脸） | 9 | fsgan、faceswap、simswap、inswap、blendface、uniface、mobileswap、e4s、facedancer |
| FR（重演） | 12 | fomm、facevid2vid、wav2lip、MRAA、one_shot_free、pirender、tpsm、lia、danet、sadtalker、mcnet、hyperreenact |
| EFS（整脸合成） | 10 | VQGAN、StyleGAN2、StyleGAN3、StyleGANXL、sd2.1、ddim、rddm、pixart、DiT、SiT |

方法名按冻结代码记录；`danet`为本地DaGAN命名，`ddim`不在这里未经核对重命名为其他官方方法。FE及e4e未进入旧实验；e4s属于FS并保留。FS/FR的21方法有自然视频单位，EFS是独立图像，不能把数字存储目录当视频。

骨干为CLIP ViT-B/16、DINOv2 ViT-B/14、EfficientNet-B0；固定种子为7、42、123、2026、3407。训练是**视觉编码器全参数微调，加新初始化二分类线性头**。CLIP只使用视觉部分；后续matched heads才冻结编码器。旧正式训练没有端到端L2训练分支。

| 参数 | 旧正式值 |
|---|---|
| 输入分辨率 | 224×224 |
| optimizer / weight decay | AdamW / 5×10⁻⁴ |
| LR：CLIP / DINOv2 / EfficientNet | 1×10⁻⁵ / 5×10⁻⁵ / 1×10⁻⁴ |
| 有效batch / 实际已通过的microbatch | 128 / 128 |
| 轮数与选点 | 固定15 epoch，使用最终第15轮 |
| scheduler / early stopping / EMA | 均无 |
| 精度与确定性 | FP16 AMP、TF32关闭、严格确定性；分析采用FP64分数 |
| 原训练设备协议 | 当时物理GPU4–7；这是历史配置，不代表新计划的设备安排 |

采样概率真实总计0.5、伪造总计0.5，每个伪造方法贡献0.5/31；视频方法内部先视频等权再帧等权，EFS内部图像等权。跨方法相同RGB输入合并为唯一训练输入，并累加各方法的sampling_mass，因此既不重复扩大epoch长度，也不丢失方法贡献。有放回抽样保证期望比例，不保证每轮或每batch精确等量。

完整审计覆盖1,514,777个路径，解码失败0。冻结训练保留265,527条方法关联并合并为250,000个唯一RGB输入；FF test 59,935行、CDF test 368,922行、源真实校准1,120帧。测试行数同时包括EFS图像和自然视频帧，不能直接当视频数。源校准为140个原FF-val真实视频，属于70个已知FF pair来源组；fake-cal=0。

已知FF来源与精确Byte/RGB内容实施隔离，未知音频驱动/EFS祖先仍为未知。62份官方修订JSON与本机副本逐字节/语义一致，不等于已认证本地图像压缩包完整来源链。CDF real `id10_0001`仅3个不同RGB帧，未进入8帧视频评价；这是不足帧，不是解码失败。

训练预算总计56,250,000次draw、439,650次optimizer尝试、439,581次成功更新、69次AMP skip；没有补步或回放。225份epoch抽样数组已核验，同seed三骨干共享预定抽样序列。每轮保存完整optimizer/AMP/RNG状态，可从完成epoch恢复；冻结旧代码未保存逐step AMP事件，不能补造各次skip的步号或当时scale。

代码证据：[模型](../../src/deepfake_detector/models.py)、[训练与采样](../../src/deepfake_detector/training.py)、[冻结协议](../../configs/v1.1.json)、[训练预算](../../reports/formal_training_budget_counts.csv)、[终态解析配置](../../reports/resolved_formal_run_configs.json)。

## 3. 正确的视频选帧、计分和AUC口径

视频选帧按`(domain, method_id, video_id)`分组，按数值帧号和路径排序，去除解码失败与重复RGB，至少8个不同RGB才纳入，再等间隔选8帧。因此方法A/001与方法B/001各自取8帧、各自计成视频，不相互借帧；真实视频跨方法只保留一池。

二输出分类头使用margin：`w=W_fake−W_real`、`b=b_fake−b_real`。Raw帧分数为`z=w·h+b`；方向分数为`z_dir=w·(h/max(||h||,1e−12))+b`，编码器和原head保持不变。L2不是对图像归一化，也不是训练新模型。

| 流程 | 输入特征 | 视频排序分数 |
|---|---|---|
| RL | Raw特征h | 8帧margin的算术平均 |
| RP | Raw特征h | 8帧sigmoid(margin)概率的算术平均 |
| DL | 单位方向u | 8帧方向margin的算术平均 |
| DP | 单位方向u | 8帧sigmoid(方向margin)概率的算术平均 |

`mean(sigmoid(z))`与`sigmoid(mean(z))`不同；mean-logit的AUC直接依据平均margin，聚合后sigmoid仅用于需要概率视图的指标。高精度实数下，聚合后正单调变换保序；浮点饱和产生ties属于另一个数值问题。

方法宏AUC先用每方法伪造与共享真实池计算AUC，再对方法等权平均；pooled AUC把全部视频作为各自独立的评价条目合并计算，每条视频等权。pooled不代表把同名视频平均成一个分数。相同共享真实池条件下，视频pooled等于按每方法伪造数量加权的方法AUC。

旧正确分组视频结果中，两种AUC接近：24个骨干×域×流程的五种子均值差最大约0.32个百分点，单seed差最大约0.52个百分点。此结论不适用于31方法图像评价：例如CDF CLIP Raw图像宏AUC78.70%，pooled66.20%，差12.50个百分点。八格分组诊断统一使用pooled，不能把宏AUC与错误分组pooled直接相减。

代码证据：[8帧选择](../../src/deepfake_detector/manifest.py)、[计分与视频聚合](../../src/deepfake_detector/experiments.py)、[逐seed主指标](../../reports/final_report/metrics_by_seed.csv)。

## 4. 正确分组正式结果

### 4.1 视频主表：21方法宏AUC

以下AUC取值0—1；±为同一骨干五个训练种子的样本标准差。三种骨干分别统计，不合并为15个同分布seed。


| 骨干 | 测试域 | RL 宏AUC | RP 宏AUC | DL 宏AUC | DP 宏AUC |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | 0.9934 ± 0.0014 | 0.9926 ± 0.0013 | 0.9925 ± 0.0014 | 0.9924 ± 0.0014 |
| CLIP ViT-B/16 | CDF | 0.8847 ± 0.0117 | 0.8818 ± 0.0114 | 0.6386 ± 0.1256 | 0.6389 ± 0.1254 |
| DINOv2 ViT-B/14 | FF | 0.8547 ± 0.0457 | 0.8494 ± 0.0416 | 0.8409 ± 0.0475 | 0.8409 ± 0.0474 |
| DINOv2 ViT-B/14 | CDF | 0.7435 ± 0.0500 | 0.7455 ± 0.0427 | 0.7295 ± 0.0518 | 0.7296 ± 0.0518 |
| EfficientNet-B0 | FF | 0.9866 ± 0.0025 | 0.9817 ± 0.0027 | 0.9838 ± 0.0022 | 0.9838 ± 0.0022 |
| EfficientNet-B0 | CDF | 0.8611 ± 0.0163 | 0.8537 ± 0.0146 | 0.8444 ± 0.0149 | 0.8448 ± 0.0147 |


CDF上，原head的推理L2在mean-logit宏AUC均值上使CLIP下降24.61个百分点、DINOv2下降1.40个百分点、EfficientNet下降1.67个百分点。结论限于这组Raw训练后的推理干预，不是所有归一化方法的普遍结论。

DINOv2在旧FF视频Raw mean-logit的宏AUC为85.47%，pooled为85.64%，低分不是宏/pooled切换造成。固定15轮旧配方没有源DEV优化充分性的正式准入规则；因此旧结果反映该预算下训练出的模型，不能被解释为DINOv2能力上限。新设计对此需另做源域开发检查。

### 4.2 源真实阈值迁移

每个骨干×种子×流程分别用同一140个源真实视频选择经验FPR≤5%的阈值，再冻结到FF/CDF。预测Fake条件为`score≥threshold`；不是统一使用概率0.5，也不是把Raw的同一数值阈值直接套L2。下表FPR、宏TPR、宏BA单位为%，±为百分点；BA=(1−FPR+宏TPR)/2。


| 骨干 | 测试域 | 流程 | FPR % | 宏TPR % | 宏BA % | 类平衡Brier |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | RL | 10.14 ± 2.64 | 98.36 ± 0.51 | 94.11 ± 1.16 | 0.2361 ± 0.0381 |
| CLIP ViT-B/16 | FF | RP | 10.57 ± 2.55 | 97.81 ± 0.47 | 93.62 ± 1.08 | 0.2070 ± 0.0344 |
| CLIP ViT-B/16 | FF | DL | 10.00 ± 2.37 | 98.29 ± 0.62 | 94.14 ± 0.96 | 0.1961 ± 0.0086 |
| CLIP ViT-B/16 | FF | DP | 10.14 ± 2.49 | 98.29 ± 0.62 | 94.07 ± 1.07 | 0.1961 ± 0.0086 |
| CLIP ViT-B/16 | CDF | RL | 90.17 ± 3.52 | 99.70 ± 0.16 | 54.76 ± 1.71 | 0.4958 ± 0.0020 |
| CLIP ViT-B/16 | CDF | RP | 87.34 ± 2.45 | 99.39 ± 0.25 | 56.02 ± 1.12 | 0.4853 ± 0.0068 |
| CLIP ViT-B/16 | CDF | DL | 91.30 ± 2.42 | 99.68 ± 0.21 | 54.19 ± 1.14 | 0.2679 ± 0.0016 |
| CLIP ViT-B/16 | CDF | DP | 91.19 ± 2.28 | 99.68 ± 0.22 | 54.25 ± 1.07 | 0.2679 ± 0.0017 |
| DINOv2 ViT-B/14 | FF | RL | 7.29 ± 1.37 | 55.62 ± 12.63 | 74.17 ± 6.17 | 0.4187 ± 0.0324 |
| DINOv2 ViT-B/14 | FF | RP | 8.43 ± 2.22 | 53.02 ± 12.59 | 72.30 ± 5.58 | 0.3765 ± 0.0368 |
| DINOv2 ViT-B/14 | FF | DL | 6.71 ± 2.06 | 41.76 ± 19.47 | 67.52 ± 8.91 | 0.2411 ± 0.0046 |
| DINOv2 ViT-B/14 | FF | DP | 6.71 ± 2.06 | 41.76 ± 19.46 | 67.52 ± 8.91 | 0.2411 ± 0.0046 |
| DINOv2 ViT-B/14 | CDF | RL | 10.28 ± 3.10 | 41.65 ± 12.45 | 65.68 ± 4.69 | 0.4475 ± 0.0204 |
| DINOv2 ViT-B/14 | CDF | RP | 10.96 ± 2.95 | 40.96 ± 11.16 | 65.00 ± 4.46 | 0.4119 ± 0.0241 |
| DINOv2 ViT-B/14 | CDF | DL | 9.15 ± 4.61 | 30.14 ± 18.70 | 60.49 ± 7.56 | 0.2463 ± 0.0018 |
| DINOv2 ViT-B/14 | CDF | DP | 9.15 ± 4.61 | 30.14 ± 18.70 | 60.50 ± 7.56 | 0.2463 ± 0.0018 |
| EfficientNet-B0 | FF | RL | 8.29 ± 0.81 | 96.69 ± 0.92 | 94.20 ± 0.43 | 0.2620 ± 0.0284 |
| EfficientNet-B0 | FF | RP | 10.14 ± 2.11 | 95.25 ± 0.88 | 92.55 ± 0.96 | 0.2224 ± 0.0237 |
| EfficientNet-B0 | FF | DL | 8.57 ± 1.52 | 95.67 ± 0.72 | 93.55 ± 1.03 | 0.1784 ± 0.0084 |
| EfficientNet-B0 | FF | DP | 8.71 ± 1.28 | 95.62 ± 0.75 | 93.45 ± 0.92 | 0.1781 ± 0.0083 |
| EfficientNet-B0 | CDF | RL | 75.14 ± 7.50 | 97.30 ± 1.20 | 61.08 ± 3.20 | 0.4808 ± 0.0063 |
| EfficientNet-B0 | CDF | RP | 75.37 ± 3.47 | 97.07 ± 0.79 | 60.85 ± 1.40 | 0.4664 ± 0.0073 |
| EfficientNet-B0 | CDF | DL | 72.09 ± 3.16 | 96.91 ± 0.70 | 62.41 ± 1.42 | 0.2780 ± 0.0054 |
| EfficientNet-B0 | CDF | DP | 71.98 ± 3.14 | 96.93 ± 0.68 | 62.48 ± 1.35 | 0.2775 ± 0.0054 |


高AUC与高误报可同时存在：AUC描述排序，冻结阈值描述某个工作点。DINOv2 Raw CDF FPR为10.28%，同时宏TPR仅41.65%；不能仅凭误报率就判断整体检测性能。原主实验只用源真实校准；后来的目标真实校准是第7节独立新增分析。未做概率校准；Brier改善不自动代表概率校准完成。TPR95因缺乏完整独立fake校准不可用。

### 4.3 图像/帧评价：31方法

每个自然视频8帧总权重为1，EFS每独立图像权重为1；本表与视频表的统计单位不同。


| 骨干 | 测试域 | 流程 | 宏AUC | FPR % | 宏TPR % | 宏BA % | 类平衡Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | raw | 0.9863 ± 0.0014 | 7.91 ± 1.05 | 96.25 ± 0.50 | 94.17 ± 0.39 | 0.2389 ± 0.0327 |
| CLIP ViT-B/16 | FF | direction | 0.9774 ± 0.0058 | 7.39 ± 0.79 | 95.49 ± 0.83 | 94.05 ± 0.19 | 0.1973 ± 0.0084 |
| CLIP ViT-B/16 | CDF | raw | 0.7870 ± 0.0218 | 83.16 ± 5.16 | 97.72 ± 0.48 | 57.28 ± 2.35 | 0.4887 ± 0.0052 |
| CLIP ViT-B/16 | CDF | direction | 0.5491 ± 0.1492 | 83.70 ± 5.31 | 97.64 ± 0.60 | 56.97 ± 2.37 | 0.2687 ± 0.0012 |
| DINOv2 ViT-B/14 | FF | raw | 0.8167 ± 0.0452 | 7.04 ± 0.97 | 49.77 ± 9.85 | 71.37 ± 4.78 | 0.4091 ± 0.0315 |
| DINOv2 ViT-B/14 | FF | direction | 0.7740 ± 0.0646 | 5.82 ± 1.06 | 29.69 ± 19.73 | 61.93 ± 9.38 | 0.2426 ± 0.0049 |
| DINOv2 ViT-B/14 | CDF | raw | 0.7109 ± 0.0365 | 9.28 ± 2.53 | 33.98 ± 9.51 | 62.35 ± 3.62 | 0.4311 ± 0.0197 |
| DINOv2 ViT-B/14 | CDF | direction | 0.6792 ± 0.0518 | 7.78 ± 4.05 | 21.31 ± 15.86 | 56.76 ± 6.24 | 0.2470 ± 0.0019 |
| EfficientNet-B0 | FF | raw | 0.9767 ± 0.0016 | 7.54 ± 1.01 | 92.47 ± 0.62 | 92.47 ± 0.57 | 0.2558 ± 0.0215 |
| EfficientNet-B0 | FF | direction | 0.9646 ± 0.0033 | 6.77 ± 0.56 | 85.43 ± 0.89 | 89.33 ± 0.49 | 0.1834 ± 0.0083 |
| EfficientNet-B0 | CDF | raw | 0.8201 ± 0.0192 | 57.67 ± 6.38 | 92.18 ± 1.95 | 67.25 ± 2.45 | 0.4731 ± 0.0062 |
| EfficientNet-B0 | CDF | direction | 0.7745 ± 0.0172 | 47.66 ± 6.12 | 83.93 ± 3.33 | 68.14 ± 1.89 | 0.2806 ± 0.0054 |


完整逐方法、逐seed及FS/FR/EFS拆分保留于[主结果目录](../../reports/final_report/)；图像CSV的legacy `real_videos`字段实际计real行数，不能当独立视频数。

## 5. 匹配头、尺度与数值控制

### 5.1 冻结表示后重新拟合30个匹配头

每个已训练编码器分别在Raw特征h、单位方向u上拟合一个新线性头，共30头。使用相同源训练行和sampling_mass、零初始化、全批L-BFGS-B、固定L2正则1e−4（bias不惩罚）、最大2000次迭代。全部收敛，实际20–672次。未运行可选正则敏感性网格，未用目标标签选参数。

下表仍为方法宏AUC（0—1）；L为mean-logit，P为mean-prob。重拟合后的阈值不复用原head数值。


| 骨干 | 测试域 | raw L | direction L | raw P | direction P | raw image | direction image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | 0.9910 ± 0.0026 | 0.9899 ± 0.0050 | 0.9923 ± 0.0016 | 0.9902 ± 0.0035 | 0.9790 ± 0.0060 | 0.9759 ± 0.0109 |
| CLIP ViT-B/16 | CDF | 0.7013 ± 0.1610 | 0.6083 ± 0.2243 | 0.7258 ± 0.1390 | 0.6184 ± 0.2148 | 0.6044 ± 0.1756 | 0.5332 ± 0.2359 |
| DINOv2 ViT-B/14 | FF | 0.8600 ± 0.0401 | 0.8457 ± 0.0403 | 0.8544 ± 0.0368 | 0.8439 ± 0.0398 | 0.8210 ± 0.0409 | 0.7766 ± 0.0625 |
| DINOv2 ViT-B/14 | CDF | 0.7540 ± 0.0538 | 0.7340 ± 0.0423 | 0.7553 ± 0.0436 | 0.7371 ± 0.0392 | 0.7194 ± 0.0405 | 0.6798 ± 0.0448 |
| EfficientNet-B0 | FF | 0.9867 ± 0.0025 | 0.9846 ± 0.0020 | 0.9823 ± 0.0027 | 0.9833 ± 0.0019 | 0.9763 ± 0.0023 | 0.9655 ± 0.0032 |
| EfficientNet-B0 | CDF | 0.8514 ± 0.0136 | 0.8018 ± 0.0085 | 0.8445 ± 0.0104 | 0.8102 ± 0.0081 | 0.7987 ± 0.0173 | 0.7213 ± 0.0073 |


这不是全参数L2训练，也不能证明方向表示对任意分类器都缺失信息。CDF上Raw与方向重拟合头的mean-logit差距仍存在；CLIP差距缩小伴随两边均值下降与较大种子SD，不应写成“重拟合救回方向表现”。完整结果：[matched_heads.csv](../../reports/final_report/matched_heads.csv)。

### 5.2 已完成的机制控制与不确定性

- 固定全局尺度`a∈{0.25,0.5,1,2,4}`；全部报告，不按CDF挑尺度。
- 仅用源训练sampling_mass加权范数中位数得到c，对比`h、h/c、u、c·u`。正比例缩放的图像/mean-logit排序控制、公共长度相应控制最大AUC偏差均为0。
- 同步正仿射变换分数与阈值时，所有预测差异为0；阈值的数值刻度变化本身不等于排序改善。
- 3×3真假对“错误/并列/正确”转移账本精确重建宏AUC差，见[pair_transitions.csv](../../reports/final_report/pair_transitions.csv)。
- 记录FP64 sigmoid精确端点、近端比例和ties。聚合后sigmoid最大绝对宏AUC数值差曾达到0.329086（AUC原始0—1尺度，即32.9086个百分点），见[scale_controls.csv](../../reports/final_report/scale_controls.csv)。该数值饱和现象不能与帧级sigmoid再平均的不交换性混成同一机制。
- 2000次测试来源组配对bootstrap与1000次源校准重采样分开。缺任一规定方法时整条宏重复记无效，不缩减方法集合；视频各2000次均有效，图像存在RDDM代理组覆盖不足导致的无效重复。

旧实验未做新附件要求的80/160/320位稳定精度递进分析，也没有以新21方法源DEV为Raw/方向头各自选择正则。旧控制可以支持提出机制假设，但不能充当这些新增分析的已完成证据。

## 6. 8帧正确/错误视频分组八格诊断

本诊断复用全部15个旧模型、21个FS/FR视频方法和已保存逐帧分数；**零新训练、零新推理、零GPU计算**。正常以`domain+method_id+video_id`分组；错误在单域内只用`video_id`，将不同伪造方法的同名ID合并。每个原视频始终8帧；k个方法碰撞时错误组平均8k帧，不是从混合池重抽总共8帧。两条件使用相同FP64平均器和同一逐帧预测。

### 6.1 全八组合大表

全部为pooled AUROC（%），±为五个训练seed样本SD。FF行表示DF40衍生FF域21方法，不是经典FF++四方法新实验。


| 测试域 | 骨干 | 正常 Raw·Logit | 正常 Raw·Prob | 正常 L2·Logit | 正常 L2·Prob | 错误 Raw·Logit | 错误 Raw·Prob | 错误 L2·Logit | 错误 L2·Prob |
|---|---|---|---|---|---|---|---|---|---|
| FF++ | CLIP | 99.35 ± 0.14 | 99.27 ± 0.13 | 99.25 ± 0.14 | 99.25 ± 0.13 | 99.46 ± 0.21 | 97.68 ± 0.37 | 99.24 ± 0.21 | 99.23 ± 0.21 |
| FF++ | DINOv2 | 85.64 ± 4.54 | 85.07 ± 4.14 | 84.23 ± 4.72 | 84.23 ± 4.72 | 81.77 ± 4.88 | 72.85 ± 4.60 | 78.71 ± 4.78 | 78.69 ± 4.79 |
| FF++ | EfficientNet | 98.69 ± 0.24 | 98.19 ± 0.27 | 98.41 ± 0.22 | 98.41 ± 0.22 | 98.85 ± 0.31 | 94.76 ± 0.38 | 97.95 ± 0.27 | 97.90 ± 0.27 |
| CDF | CLIP | 88.63 ± 1.16 | 88.36 ± 1.13 | 63.98 ± 12.56 | 64.02 ± 12.54 | 79.40 ± 1.66 | 75.16 ± 2.18 | 64.45 ± 8.01 | 64.46 ± 7.99 |
| CDF | DINOv2 | 74.66 ± 4.98 | 74.86 ± 4.22 | 73.24 ± 5.16 | 73.24 ± 5.15 | 65.08 ± 4.87 | 60.82 ± 3.13 | 63.40 ± 4.57 | 63.39 ± 4.57 |
| CDF | EfficientNet | 86.33 ± 1.59 | 85.61 ± 1.44 | 84.44 ± 1.59 | 84.48 ± 1.56 | 77.94 ± 0.80 | 68.95 ± 1.68 | 74.42 ± 1.39 | 74.41 ± 1.38 |


### 6.2 实际评价单位数量

| 测试域 | 真实视频（两分组不变） | 正常伪造视频 | 错误合并伪造组 | 多方法错误组 | 单方法组 | 最大方法数/错误组 | 共用帧行数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FF | 140 | 2884 | 280 | 271 | 9 | 19 | 24192 |
| CDF | 177 | 12876 | 1958 | 648 | 1310 | 19 | 104424 |

错误分组同时改变了评价单位数量和权重。原来含19方法的同名ID对应19个独立方法视频，合并后只占1个伪造组；不能把这种AUC变化当成模型本身提升。真实视频、单方法组对应分数不变，跨类别ID碰撞预检通过。

### 6.3 归一化相对结论的反转

定义`Δ_correct=AUC(正确,L2)−AUC(正确,Raw)`，`Δ_wrong=AUC(错误,L2)−AUC(错误,Raw)`；指定反转需同一模型的`Δ_correct<−1e−10`且`Δ_wrong>1e−10`。以下差值单位为百分点。

| 域 | 骨干 | 聚合 | 正常Δ | 错误Δ | 指定方向反转seed数 |
|---|---|---|---:|---:|---:|
| CDF | CLIP | mean-logit | −24.64 | −14.95 | 0/5 |
| CDF | DINOv2 | mean-logit | −1.42 | −1.67 | 0/5 |
| CDF | EfficientNet | mean-logit | −1.89 | −3.51 | 0/5 |
| CDF | CLIP | mean-prob | −24.34 | −10.70 | 1/5 |
| CDF | DINOv2 | mean-prob | −1.62 | +2.58 | 4/5 |
| CDF | EfficientNet | mean-prob | −1.13 | +5.45 | 4/5 |
| FF | CLIP | mean-prob | −0.02 | +1.54 | 3/5 |
| FF | DINOv2 | mean-prob | −0.83 | +5.83 | 3/5 |
| FF | EfficientNet | mean-prob | +0.21 | +3.14 | 1/5 |

FF三个骨干的mean-logit指定反转也均为0/5。CDF中DINOv2/EfficientNet的mean-prob出现五种子均值方向反转，各4/5模型满足指定判据；CLIP均值未反转，但1/5个体反转。错误分组常常让Raw与L2的绝对AUC一起下降，只是Raw下降更多，形成表观L2相对优势。因此“归一化相对优势增加”不等于“错误分组使所有绝对AUC升高”。

结果支持：错误评价单位结合概率平均，能在当前固定模型上改变Raw/L2的相对评价结论。它不能单独证明已经删除旧模型的历史提升原因，也不能单独证明Sigmoid饱和的净贡献；旧训练阶段混合抽帧的影响未被这项评价干预隔离。

本次共有240个AUC格（8条件×15模型×2域），其中错误分组新增120格。正常分组与原保存结果AUC误差0；独立持久化视频表重算240格的最大AUC误差1.11×10⁻¹⁶。完成凭据记录67项测试通过、独立复核通过；本归档仅引用已有验收，不重跑实验。

证据：[完整八格CSV](../../reports/video_grouping_8f_20261023/auc_cells.csv)、[逐seed配对差](../../reports/video_grouping_8f_20261023/auc_by_seed.csv)、[五seed汇总](../../reports/video_grouping_8f_20261023/summary_by_backbone.csv)、[组计数](../../reports/video_grouping_8f_20261023/group_counts.json)、[独立核验](../../reports/video_grouping_8f_20261023/independent_verification.json)、[完成凭据](../../reports/video_grouping_8f_20261023/complete.json)。

## 7. 已有CDF目标真实校准：与源域迁移实验分开保存

这是2026-10-22独立授权的回顾性分析，复用旧15模型的8帧mean-logit，Raw/L2原head均不重训。80个CDF真实视频作校准池（48 Celeb-real、32 YouTube-real），其余固定测试为97真实+4132伪造视频，保留21方法。按视频去重，排除共享既有来源组、Byte/RGB及`idN`名称标记；名称仅是保守排除代理，不认证人物身份或未知驱动祖先。

所有六行共用该固定测试集，因此“0个CDF校准样本”的行也不能直接与第4节177真实/12876伪造的完整测试指标混比。n=10/20/40/80为80池200次共同无放回排列的嵌套前缀，所有骨干、seed和Raw/L2共用每次ID序列；n=80每次是同一集合。

阈值仅用真实校准分数选择经验FPR≤5%，无目标伪造参与阈值选择。n=10的经验FPR步长10%，只能允许0个误报；n=20/40/80最多1/2/4个误报。测试FPR不保证为5%。阈值在原生logit域比较，固定概率0.5行等价于mean-logit≥0，不是logit≥0.5。

每格依次为**宏BA / FPR / 宏TPR（%）**。n>0先在每模型内平均200次抽样，再平均5个模型；模型种子SD与校准抽样波动分别保留在原CSV，不能把5×200当1000个独立模型。



### CLIP ViT-B/16

| CDF真实校准视频数 | Raw：BA / FPR / TPR | L2：BA / FPR / TPR |
|---|---:|---:|
| 固定概率阈值0.5 | 50.00 / 100.00 / 100.00 | 50.10 / 99.79 / 100.00 |
| 0（FF++源阈值） | 53.80 / 92.16 / 99.77 | 52.86 / 94.02 / 99.74 |
| 10 | 75.76 / 10.63 / 62.16 | 51.72 / 9.28 / 12.72 |
| 20 | 78.71 / 10.69 / 68.12 | 51.82 / 10.25 / 13.89 |
| 40 | 78.54 / 8.33 / 65.41 | 51.61 / 7.29 / 10.51 |
| 80 | 78.20 / 6.80 / 63.20 | 50.53 / 6.60 / 7.66 |


### DINOv2 ViT-B/14

| CDF真实校准视频数 | Raw：BA / FPR / TPR | L2：BA / FPR / TPR |
|---|---:|---:|
| 固定概率阈值0.5 | 51.14 / 96.49 / 98.78 | 51.53 / 95.67 / 98.73 |
| 0（FF++源阈值） | 66.31 / 9.69 / 42.31 | 60.37 / 9.90 / 30.63 |
| 10 | 63.32 / 9.07 / 35.71 | 59.95 / 9.67 / 29.58 |
| 20 | 64.29 / 9.65 / 38.23 | 60.73 / 10.24 / 31.70 |
| 40 | 63.11 / 8.24 / 34.46 | 59.32 / 8.23 / 26.86 |
| 80 | 62.37 / 7.84 / 32.57 | 58.26 / 7.84 / 24.36 |


### EfficientNet-B0

| CDF真实校准视频数 | Raw：BA / FPR / TPR | L2：BA / FPR / TPR |
|---|---:|---:|
| 固定概率阈值0.5 | 51.02 / 97.94 / 99.98 | 51.12 / 97.73 / 99.98 |
| 0（FF++源阈值） | 60.76 / 76.08 / 97.60 | 61.65 / 73.81 / 97.11 |
| 10 | 72.22 / 9.26 / 53.70 | 69.97 / 11.28 / 51.23 |
| 20 | 75.45 / 9.37 / 60.27 | 72.75 / 12.42 / 57.92 |
| 40 | 74.88 / 7.30 / 57.06 | 72.19 / 9.96 / 54.34 |
| 80 | 74.44 / 6.39 / 55.26 | 71.89 / 10.31 / 54.08 |


Raw n=0→20时，CLIP宏BA从53.80%到78.71%，EfficientNet从60.76%到75.45%，DINOv2从66.31%到64.29%。目标真实校准对不同骨干的收益不一致；对已有CDF探索背景必须如实披露。改变阈值不会改变同一固定测试集合的AUC，但此处重新限制测试集合会改变所评价总体，不能由阈值表推回完整CDF AUC。

完成凭据记录24,060条重复/基线记录独立重算、最大指标误差0、全部哈希读回通过。证据：[完整说明](../../reports/cdf_real_calibration_20261022_v1/结果与方法.md)、[逐seed表](../../reports/cdf_real_calibration_20261022_v1/by_seed.csv)、[分割清单](../../reports/cdf_real_calibration_20261022_v1/video_split.csv)、[完成凭据](../../reports/cdf_real_calibration_20261022_v1/complete.json)。

## 8. 32帧旧16方法方案：只有方案，没有正式结果

[32帧方案](../superpowers/plans/2026-10-23-video-grouping-auc-reversal.md)拟使用当前15模型、旧仓库默认16方法、CDF域，每原视频32个有效不同路径（不额外RGB去重）并以mean-prob对比Raw/L2及正确/错误分组。它需要补充8帧缓存以外的新推理，尚未实施。

只读预盘点为153个真实视频、8438个正确伪造视频、1612个错误伪造组，其中647多方法组、965单方法组，最大14方法/组，274912帧。它们是计划时预盘点，未冻结为正式执行输入，不是原历史实验总体。错误组有k方法时将平均32k帧，不会在每组重新抽总共32或8帧。新附件以8帧同缓存机制诊断为主，不自动授权启动旧32帧计划。

## 9. 原视频资产与本机保留位置

2026-10-22新增经典FaceForensics++ c23完整视频下载已经完成：

| 相对`/Dataset/deepfake_detection/FaceForensics++`的目录 | 视频数 |
|---|---:|
| `original_sequences/youtube/c23/videos` | 1000 |
| `manipulated_sequences/Deepfakes/c23/videos` | 1000 |
| `manipulated_sequences/Face2Face/c23/videos` | 1000 |
| `manipulated_sequences/FaceSwap/c23/videos` | 1000 |
| `manipulated_sequences/NeuralTextures/c23/videos` | 1000 |
| 总计 | 5000 |

总字节9,041,543,739（约9.04GB十进制）。下载凭据检查覆盖完整manifest、无缺失/partial标记、5000条均有视频流与正时长；该验证为ffprobe检查，不是每一帧完整解码。下载数据尚未用于前述旧模型或完成预处理历史实验。既有DFD属于另一原视频资产，具体容量与健康状态由新计划的环境盘点重新核实。

| 本机路径 | 内容与用途 |
|---|---|
| `/home/z/project/deepfake_detector` | 源码、协议、测试、小型报告镜像 |
| `/home/z/workspace/deepfake_detector_v11_20261007` | 旧实验实际artifact根目录；项目`artifacts`为其符号链接 |
| `.../runs/<backbone>_seed<seed>/` | epoch15 final模型、完整状态、history、逐轮采样数组 |
| `.../features/<backbone>_seed<seed>/` | 冻结身份绑定的训练/校准/测试特征缓存 |
| `.../analysis/<backbone>_seed<seed>/` | 逐帧/逐视频分数、matched heads、阈值及机制分析 |
| `.../data/` | 审计、冻结清单、角色与来源处置 |
| `/Dataset/deepfake_detection/DF40_all/dataset_json_update` | 旧实验62份修订索引JSON |
| `/Dataset/deepfake_detection/FaceForensics++` | 新下载经典FF++ c23原视频 |
| `/Dataset/deepfake_detection/FaceForensics++/_download_ffpp_c23_20261022/verification.json` | 原视频5000文件完整性与ffprobe凭据 |
| `/home/z/project/deepfake_detector/reports/cdf_real_calibration_20261022_v1` | CDF真实校准完整独立分析包 |
| `/home/z/project/deepfake_detector/reports/video_grouping_8f_20261023` | 八格分组诊断完整独立分析包 |

上表`...`均指`/home/z/workspace/deepfake_detector_v11_20261007`。仓库发布保存源码、小型证据与本地路径索引，不能被理解为大模型权重、原数据和全量缓存已上传。

## 10. 与新附件设计的衔接及证据边界

用户新附件为README、交付边界、英文稿、中文稿、中文实验协议。按附件协议，新正式模型为D路线21视频方法Raw训练15个，加H路线经典FF++四方法两种训练历史Raw训练30个，共45个；八种评分条件复用模型，不分别训练8套。新H路线涉及FF++与DFD的预处理历史，新D路线需要重新划源TRAIN/DEV、训练与冻结；旧31方法模型均不计入新45模型。

可复用的是经过核验的模块和经验：来源/内容审计、正确视频键、模型封装、采样质量控制、margin与两种聚合、独立校准、缓存身份校验、配对统计及CPU分组诊断。新训练方法集合、DEV划分、优化配方、处理历史、阈值池或精度协议发生变化时，必须产生新清单/配置/模型身份；不能覆盖旧产物或把旧数字更名。

尚未由旧证据解决的问题：新21方法结果；DINOv2在源DEV合格预算下的结果；原视频预处理历史效应；高精度重算揭示的Sigmoid数值误差净贡献；未知源身份/驱动/生成祖先独立性；已删除旧模型的历史提升唯一成因。归档中没有把这些未知项预写成有利结论。

## 11. 最小证据索引与验收身份

| 证据 | 仓库相对链接 |
|---|---|
| 主实验结果总报告 | [final_results.md](../../reports/final_results.md) |
| 源码复现入口与步骤 | [README](../../README.md)、[manual_reproduction.md](../../reports/manual_reproduction.md) |
| 冻结配置 / 来源隔离授权 | [v1.1.json](../../configs/v1.1.json)、[source_isolation_authorization.json](../../configs/source_isolation_authorization.json) |
| 冻结数量 | [frozen_unit_counts.csv](../../reports/frozen_unit_counts.csv) |
| 主指标 / 方法明细 | [metrics_by_seed.csv](../../reports/final_report/metrics_by_seed.csv)、[metrics_by_seed_method.csv](../../reports/final_report/metrics_by_seed_method.csv) |
| 主实验独立终态验收 | [final_independent_acceptance.json](../../reports/final_independent_acceptance.json) |
| 训练预算 / 完整缓存凭据 | [formal_training_budget_counts.csv](../../reports/formal_training_budget_counts.csv)、[completed_run_cache_receipts.json](../../reports/completed_run_cache_receipts.json) |
| 官方JSON与本机内容核对 | [official_local_verification.md](../../reports/official_local_verification.md) |
| 尺度 / 长度 / 排序转移 | [scale_controls.csv](../../reports/final_report/scale_controls.csv)、[length_controls.csv](../../reports/final_report/length_controls.csv)、[pair_transitions.csv](../../reports/final_report/pair_transitions.csv) |
| 匹配头 | [matched_heads.csv](../../reports/final_report/matched_heads.csv) |
| bootstrap支持与结果 | [bootstrap_support_structure.json](../../reports/bootstrap_support_structure.json)、[test_bootstrap.csv](../../reports/final_report/test_bootstrap.csv)、[calibration_bootstrap.csv](../../reports/final_report/calibration_bootstrap.csv) |
| 目标真实校准独立包 | [结果与方法](../../reports/cdf_real_calibration_20261022_v1/结果与方法.md) |
| 八格分组诊断独立包 | [结果与方法](../../reports/video_grouping_8f_20261023/结果与方法.md) |
| 32帧未执行计划 | [方案](../superpowers/plans/2026-10-23-video-grouping-auc-reversal.md) |

原计算源码SHA256：`7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5`。主实验独立验收时间为2026-10-09T13:12:44.428643+00:00，状态`INDEPENDENT_FINAL_ACCEPTANCE_PASS`；CDF真实校准完成时间为2026-10-22T13:10:10.356075+00:00；8帧分组诊断完成时间为2026-10-23T04:24:47.732541+00:00，状态`COMPLETE_INDEPENDENTLY_VERIFIED`。

本页归档于2026-10-23，数值引用上述持久化报告与凭据。新计划、旧实验归档、原始冻结证据分别保存；本页不改写原结果，不以新的解释替代原始记录。

## 12. 更早GitHub实现的审计背景

远端仓库`xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection`原master提交为`5cc0105477509566295ea6cd3c277cdcf4ede82e`，与当前本机31方法正确实现不是同一套代码。原提交会保留在Git历史中，新分支增量保存当前工作。

旧`data/index_ffpp.py`将FrameRecord.video_id取为裸pair_id；`data/builders.py`训练/验证汇集方法后按裸ID抽帧，使用拼接列表上的均匀位置采样，合并组总共8帧，不是每方法8帧，也不是随机保证每方法等量。旧FF/CDF测试的build_eval_loader则加载JSON列出的全部测试帧，通常每个方法视频32帧，不经过同一8帧抽样步骤。

旧`evaluation/sample_eval.py`按裸video_id累加逐帧fake概率并除以帧数，因此测试中同ID跨k方法会把约32k帧平均成一个假视频分数。二输出softmax fake概率等于sigmoid(fake−real margin)，所以是mean-probability，不是mean-logit。不同完整pair_id不因共享第一个编号就自动合并。旧DFD只有一种操纵类型，不具有跨16方法的同一问题；外部测试另有命名空间规则，不能概括为所有数据集都以相同方式混合。

旧归一化脚本共享同一checkpoint/原head和帧，但沿用裸ID评价，故可以受该缺陷影响。原训练/验证抽样缺陷还可能改变模型及选点；当前八格只控制评价分组，不能把所有历史影响都归为评价或Sigmoid。用户更早删除的本地实验/权重没有被本次重新找到；其他Visual-JEV项目归档不视为其替代物。
