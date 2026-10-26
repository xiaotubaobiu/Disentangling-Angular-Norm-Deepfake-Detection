# AntifakePrompt 候选决策验证设计

## 用户目标与边界

暂时归档 E2/SmolVLM 路线，使用用户指定的 AntifakePrompt 做小幅候选评分改造、微调和 FF++/CDF 测试。保留预训练视觉、Q-Former 与语言 Transformer，考察不使用词表输出头的受限候选决策。仅 GPU 0–3，禁止子智能体。旧文件保持原路径，不覆盖冻结实验。

本次是架构实验。用户已授权实现、微调、测试，执行方法为本会话独立完成；不重复设计审批。新的设计与计划可供审阅纠正。旧主仓库包含未提交项目拆分，不能用 HEAD 工作树替代当前源码或整理其他改动；新代码隔离在 E4 实验目录，上游代码单独克隆并固定提交。

## 已查明的上游事实

- 上游 https://github.com/nctu-eva-lab/AntifakePrompt，固定提交 `e261aa6ee4c398f63a36c08b7ebad3455dc1428e`。
- 视觉 EVA ViT-g、InstructBLIP Q-Former、Vicuna 7B v1.3；骨干冻结，Q-Former 与 LLM 各训练一个伪词向量。
- 官方 `test.py` 调用 `predict_class`，按候选答案 token NLL 排序，已经不进行逐词生成。不能把本实验包装为“第一次移除生成循环”。细读实现确认 reduction=none 再沿包含masked位置的整段序列求均值；同批候选的分母相同，所以排序等价于答案 NLL 求和。
- 官方包含 FaceForensics++ 检测研究；README 未列 Celeb-DF。官方图像生成检测成绩不能替代本项目两域视频实验。
- 官方公开 prompt checkpoint 可获取；其名称与 YAML 中 90k 训练统计存在不一致。采用 checkpoint 内实际 `datasets.textinv.vis_processor.eval` 的 mean=[0.4874,0.4694,0.4305]、std=[0.2854,0.2802,0.3005]，不使用上游遗留default统计，不声称精确复现论文表格。

## 方案选择

1. 推荐并采用：直接复用上游模型实现，提供最小加载器和两种评分接口，保留官方 token NLL 作为对照。适配旧 transformers 环境，避免模型转换带来额外变量。
2. Hugging Face InstructBLIP 重实现：依赖更新但需映射全部权重与 soft prompt，增加等价性负担，暂不采用。
3. 更换为另一规模模型：不能孤立检验输出结构，本阶段不做。

## 实验组

- A0 official：官方公开 COCO+SD3/SD2IP prompt checkpoint，官方问题 `Is this photo real [*]?` 与 yes/no，用 upstream predict_class 做排序等价性检查。全 T1–T5 使用冻结原版 backbone 与 prompt 的扩展问法，明确不是官方 benchmark。
- A1 lm_prompt：从 A0 同一权重初始化，保留词表头，只微调两个软提示向量。训练正确候选的 token CE（含 EOS，沿用上游训练目标）；推断按不含 EOS 的候选 NLL 总和排序，与官方排序等价。候选概率统一由这些分数softmax得到，去掉上游随batch padding改变的公共分母；此概率是明确新增的归一化定义，官方只返回rank，不声称是官方校准概率。
- A2 scalar_prompt：从 A0 同一权重初始化，删除 LM vocab head，保留全部 Transformer，两个软提示向量及 LayerNorm+Linear 共享标量头可训练。每条候选独立输入相同图像、问题及候选文本，末个非 padding token 隐状态评分；候选间 softmax、正确候选 CE。

主比较 A2−A1 控制预训练骨干、数据、语言池、初始化、训练例数、选模域。训练目标与读出方式共同变化，因此结论关于这个完整受限决策替换，不能再拆成单一头结构因果。A0 用于检测能力起点，H1/E3 只作历史背景。

## 第一阶段预算与数据协议

先实施可复现的单种子（42）架构验证，完成后汇报，不自动压缩语言层或开展 RL。更多种子属于后续确认实验。

- DF40 派生 FF++ 训练与验证，CDF 只用于最终测试；两个测试域包含同样 16 方法，不是官方原始 benchmark，不是未见方法测试。
- 从已有物理划分建新清单；方法、视频、帧路径一起标识。训练/验证/测试不得重叠。重用语言 bank 的 train 与 held-out 隔离。
- 训练：最多 3 epochs，每 epoch 4096 例；在 camera/swap/reenact 三家族均衡抽样，T1–T5 均匀确定性分配；每训练视频最多 2 帧。A1/A2 同序列同目标，采用相同224px bicubic与上述归一化，不加随机数据增强。优化器 AdamW，prompt lr=1e-5、head lr=1e-4、weight_decay=0.05。有效 batch=24（epoch最后16例按实际例数归一化），FP16、gradient checkpointing；不因 CDF 成绩调参。
- 验证：FF++ val，每个伪造方法最多 8 个视频、真实最多 32 个视频，各取 1 帧；固定 train 语言池的全 T1–T5 变体。按各组自身正确候选 CE 选最佳 epoch；不以不同组 CE 的数值相互比较。
- 最终固定真假：两域所有测试视频，每视频 4 个均匀帧（不足4全部使用）。同时评 A0 官方 Yes/No 问题与三组的固定 H1 候选描述。
- T1–T5/控制：每域每伪造方法最多 32 个视频、真实全部视频，各4帧，按视频键 hash 选择，预测前冻结清单。seen、held-out、counterfactual、meaningless candidates/questions、negation、composition、length-matched controls、permutation。结果明确标注子集，不能直接填入旧 E3 全量66组表。
- 精确抽样规模、文件 hash、预处理与环境版本在训练前落盘；性能测量后可以下调 microbatch、提高累积次数保持 batch，不能根据测试标签改变科学协议。

## 证据与判定

报告 A0/A1/A2 的两域 AUROC、BA、真实/伪造召回、Brier/ECE；T1–T5 的 seen/held-out、否定与组合诊断。候选换序映射回语义后的预测一致率应为100%，最大概率差报告实测值。用配对物理视频 bootstrap 10000次报告 A2−A1 区间。

阶段性支持至少需要 A2 相对 A1 两域固定真假 AUROC 的单侧95%下界 ≥−0.01，并满足旧语义门槛（T2 BA≥0.80、极性≥0.95、T4 BA≥0.65、家族排序≥0.80、T5 macro BA≥0.60、held-out 单任务下降≤0.05、宏平均下降≤0.02、文本控制增益≥0.10）。一个种子及诊断子集只能提供该协议下的初步证据，不能声称普遍证明；未通过时报告失败或不确定而不偷偷扩展预算。

另报总/可训练参数、实际峰值显存、延迟和吞吐。仅删词表头不意味着已压缩7B语言骨干，也不直接等于降低推理能力。

## 工程验证与保存

测试关键风险：候选 mask 与换序、padding 汇聚、答案 token loss 边界/归一化、软提示梯度贯穿冻结骨干、原版 rank 等价、断点只保存可训练权重、无数据泄漏、显式 GPU 白名单。先真实模型前向/反向 smoke，再训练，smoke 与正式结果分目录并不可汇报成实验指标。

下载按公开固定 revision 保存并校验；独立 Python 环境，不降级旧项目环境。不执行上游全部安装脚本。旧实验新增归档索引和重要产物 hash 清单，不移动、不删数据。
