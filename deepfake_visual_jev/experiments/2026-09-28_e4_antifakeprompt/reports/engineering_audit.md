# 工程与协议核查

本文件记录第一阶段正式结果前完成的核查，不是科学结论。

- 上游源码固定e261aa6，未修改网络代码；局部namespace loader仅避免导入无关LAVIS数据集/扩散依赖。
- EVA官方特征提取为39层；公共checkpoint含第40层和分类头，按上游行为忽略，其他缺失/额外权重拒绝加载。
- InstructBLIP trimmed权重覆盖Q-Former、query tokens、视觉LayerNorm和LLM projection；Vicuna单独加载。官方prompt的两个向量及附带position/rotary buffers均载入。
- `predict_class`实际上按候选答案NLL排序，而非开放式生成。原版mean包含被mask位置，因此只返回rank；E4候选概率明确采用sum answer log likelihood的softmax，保持rank、去掉动态padding公共分母。真实GPU smoke确认与上游rank一致。
- Scalar模型删除lm_head，保留Transformer、Q-Former和视觉塔；训练的只有两个soft prompt和共享LayerNorm+Linear。bias的零梯度符合共享softmax平移不变性，不误判为断梯度。
- 真实7B前向/反向、两组优化器更新、A0/A2所有12条件smoke完成；smoke显式标记reportable=false。
- EVA特征原生float32，缓存保留float32而不量化。缓存校验数据物理ID、图片stat、输入/预处理指纹、dtype/shape和完整SHA256；损坏与缺失ID测试通过。
- 无意义长度控制同时匹配Q-Former BERT和Vicuna Llama的完整输入长度。只依据分词性质扩展中性词表（包括单token词和BERT/Llama长度差相反的whimsical），不读取标签或试验成绩。最终22个替换池已冻结并预检。
- 原始换序审计会把2候选/3候选拆为不同batch，FP16下混入batch数值敏感性。保留原始G3结果；另有保持batch组成不变的补充诊断，能够检测真实位置偏差，不能用它覆盖原始未通过结果。
- 单种子统计直接以物理视频分层bootstrap，不重复复制seed42伪装为三种子；每个视频的全部语言变体共享抽样权重。
- 最新本阶段开发测试：25项通过，含统计配对身份检查及partial审计不能启动最终GPU测试的回归用例。最终仍需核验训练完成记录、72组正式评估、bootstrap、资源测试和报告。

用户禁止子智能体；所有代码、检查和自审在本会话完成。GPU计算进程及Python线程调度不是智能体。

追加物理路径审计：训练真实视频有16个方法别名（9,376个real索引键实际为586个视频）；验证32个real索引键实际为28视频。所有训练/验证/测试物理视频目录交集为0，测试/语义数据无重复。决定保留已冻结的相同两组采样及选模权重，在报告中明确真实计数；不把这些训练别名当独立证据或bootstrap样本，也不根据已出现的成绩事后改选checkpoint。详见results/protocol/physical_audit.json。

补充候选长度审计：固定真假描述在Vicuna中分别为6/10个答案token（排除BOS/EOS），官方yes/no为1/1。全部20组候选的输入长度保存于results/summary/candidate_length_audit.json。求和语言似然可能带来长度偏好；这只是解释候选，不能由长度统计证明因果，而且等长官方Yes/No同样表现较弱。此次不做后验长度归一化或更改冻结评分规则。
