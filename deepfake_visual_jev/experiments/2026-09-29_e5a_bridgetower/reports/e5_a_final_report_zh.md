# E5-A BridgeTower 最终报告

**E5-A STATUS: FAIL**。Stage B killer test 未通过：G2_polarity。按冻结协议立即停止，未执行 Stage C 或 E5-B/C/D。

## 1. 研究目标与模型

检验经过视觉语言判别式预训练的 BridgeTower，能否用原生 ITM head 对完整“问题＋候选”文本逐一共享评分，同时支持视觉真假与问题极性、未见表达语义。

- 检查点：`BridgeTower/bridgetower-base-itm-mlm`，revision `6cfb318bb08588808368a9ab1fe26609bc148f1f`；本地权重 SHA256 见 `results/protocol/model_loading_audit.json`。
- 实现：Hugging Face `BridgeTowerForImageAndTextRetrieval`，每个候选独立通过同一模型，取 ITM match logit 后做候选 softmax 和交叉熵。没有候选专属 head、生成器或新增大 MLP。
- 总参数：326,350,082；可训练参数：115,806,722。冻结 `vision_model` 和 `text_model`；训练 cross-modal/bridge 模块及原生 ITM head。
- 原仓库配置缺少 `model_type`；以官方 BridgeTower 类分别加载图像处理器、RoBERTa tokenizer 和 ITM 模型。固定版本 RoBERTa slow BPE tokenizer 的词表与 BridgeTower 附带词表逐项相同，merges 固定于 `e2da8e2f811d1448a5b465c236feacd80ffbac7b`；71 个正式问题候选文本的最长输入为 31 token。权重加载只有未使用的 MLM 权重，没有缺失的 ITM 权重；真实图像前向和候选换序检查已通过。

## 2. 数据与训练

- 仅用 DF40 派生 FF++ 来源域，不使用 CDF。训练 9,962、验证 156、测试 3,049 个物理视频；测试中真实 140、伪造 2,909。
- 训练/验证/测试物理视频交集均为 0；FF++ 语义子集属于完整测试集。逐文件 SHA256 与详细审计见 `results/protocol/physical_split_audit.json`。
- 仅 seed 42。T1–T5 轮换抽样，训练任务比例约各 20%；T1–T3 在真实/伪造间平衡，T4 的 Yes/No 目标各半，T5 的 camera/swap/reenact 各约三分之一。每轮 4096 个训练输入，逐轮计数见 `results/protocol/sampling_audit.json`，详细配置见 `configs/e5_a.yaml`。
- 只按 FF++ 验证候选交叉熵选模；最佳值 0.298354。训练历史见 `results/training/history.json`；无跳过优化更新。前期的 FP16 非有限梯度、Fast tokenizer 逐字符分词，以及 T4 目标答案不平衡，均在测试前发现；相关日志与未完成训练隔离保存。正式训练从原始权重重新开始，使用正确 BPE、T4 正反目标均衡和 FP32；验证选模规则及门槛未变。

## 3. Stage B 结果

| 条件 | T2 视频变体平衡准确率 | 极性一致率 | 候选换序语义一致率 |
|---|---:|---:|---:|
| seen | 94.08% | 95.88% | 未测试 |
| held-out | 82.38% | 77.01% | 100.00% |

门槛：held-out T2 平衡准确率 ≥70%；同视频真假互反问句极性一致率 ≥90%；候选换序后语义预测一致率 =100%，且最大语义概率误差 ≤1e-05（浮点数容差）。二分类随机参考为约 50%。T2 seen 与 held-out 均明显高于二分类随机参考，说明存在有效的视觉真假判别信号；极性互反关系只有约四分之三成立，主要失败点是问题极性语义的一致使用。

测试以物理视频聚合帧概率，再按视频×语言变体计分；held-out 换序在 97,568 个帧级请求上核验，最大概率误差为 0。逐帧预测与机器结果在 `results/stage_b/`。

未见明显实现异常：四个评估分片均正常退出，完整覆盖 3,049 个物理测试视频；检查点和源数据哈希匹配；换序概率一致。加载时未使用的 MLM 权重符合 ITM 任务头切换，原生 ITM 权重无缺失。此审计不能证明模型学会了所有语义关系，极性 gate 的失败是实测结果。

## 4. Gate 与退出

| Gate | 状态 |
|---|---|
| G1 / T2 held-out | PASS |
| G2 / polarity | FAIL |
| G3 / T1/T3 | 未评估，Stage B 已失败 |
| G4 / T4 | 未评估，Stage B 已失败 |
| G5 / T5 | 未评估，Stage B 已失败 |
| G6 / permutation | PASS |

**E5-A STATUS: FAIL。** 可以退出，因为 Stage B 中至少一个预先规定的 killer gate 已失败；协议要求停止后续实验，不允许事后调参救模型。该结论仅针对当前 FF++ qualification protocol 与固定训练范围，不能唯一定位视觉表征或语言交互的全部原因。
