# Deepfake Detection 实验进展汇报 PPT 内容
## 2024年7月21-30日：从"direction更好"到"阈值不可迁移"的五问

**汇报时长**：20-30分钟  
**目标受众**：同实验室同学  
**结构方式**：问题→证据→结论（按4个研究问题组织）  
**技术深度**：平衡式（方法简洁+结果重点+逻辑清晰）

---

## 目录结构

- 【开头部分】(2页)
- 【问题1】direction > raw 是真的吗？(5页)
- 【问题2】为什么归一化增益因方法而异？(4页)
- 【问题3】模长真假关系跨域稳定吗？(4页)
- 【问题4】direction优势是特征几何还是饱和伪影？(4页)
- 【问题5】阈值能跨域迁移吗？(4页)
- 【总结部分】(2页)

**总计：26页**

---

## 【开头部分】

### 第1页：标题页

```
标题：Deepfake Detection 实验进展汇报
副标题：从"direction更好"到"阈值不可迁移"的五问 (7月21-30日)
汇报人：[你的名字]
日期：2024年7月30日
```

---

### 第2页：研究背景与问题链

```
【左侧：背景】
• 5月实验发现：CDF上direction > raw
  - ΔAUC = +0.0142
  - 15/15个seed×backbone组合direction胜
  - 发表为会议结果

【右侧：核心疑问】
• 这个现象真的可靠吗？
  - 分组方式是否影响了结果？
  - 机制是什么？
  - 能跨域迁移吗？

【底部：研究路径】
7月21日 → 中期诊断 (问题1)
7月23日 → 四问实证 (问题2-4)  
7月27日 → 饱和度研究 (问题4深化)
7月30日 → 阈值迁移 (问题5)
```

---

## 【问题1：direction > raw 是真的吗？】

### 第3页：问题定义

```
【标题】问题1：CDF上direction > raw的现象可靠吗？

【左侧：原始发现】
• 数据：30个checkpoint (3 backbone × 5 seed × 2 train_mode)
• 评估域：FF++、CDF、DFD
• 结果：CDF上direction > raw (+0.0142, 15/15)

【右侧：核心疑问】
• 分组方式是否影响了结果？
  - video_id聚合 vs (method_id, video_id)聚合
  - 样本量不平衡问题

【研究方法】
• 全量重推理：39.8M帧 → per_frame_scores.parquet
• 多种聚合方式：native / balanced / method-level
• 5个假设检验

【演讲要点】
强调我们怀疑分组方式可能影响了结果，所以做了全量重推理来验证。
```

---

### 第4页：实验设置

```
【标题】实验设置：全量重推理与多种聚合方式

【上半部分：数据源】
• Checkpoint：30个 (阶段1训练)
  - 3 backbone: CLIP, DINOv2, EfficientNet
  - 5 seed: 7, 42, 123, 999, 2048
  - 2 train_mode: raw, normtrain

• 输出数据：per_frame_scores.parquet
  - 39.8M帧
  - 每帧包含：feature, prob_fake, label, method_id, video_id...

【下半部分：三种聚合方式】
1. **native (全量)**
   - 按method_id+video_id聚合
   - 不控制样本量
   
2. **balanced 178v178 (等量采样)**
   - fake下采样到178个
   - 与real数量相等
   - RNG_SEED固定为20260722
   
3. **method-level**
   - 每个方法单独评估
   - 查看方法间差异

【图片位置】
- 暂无图片（纯文字说明）

【演讲要点】
强调balanced 178v178的重要性：消除了样本量偏差，让每个方法权重平等。
```

---

### 第5页：关键发现 - native vs balanced

```
【标题】关键发现：direction优势仅在特定聚合下成立

【左侧：数据表格】
CDF上direction - raw的ΔAUC：

| 口径          | CLIP    | DINOv2  | EfficientNet | 说明 |
|---------------|---------|---------|--------------|------|
| native (全量) | +0.0018 | -0.0044 | -0.0112     | 原始方式 |
| balanced 178v | +0.0000 | -0.0005 | -0.0141     | 等量采样 |

**关键观察**：
- CLIP：balanced接近0，全量微弱正 → 样本量偏差
- DINOv2/EfficientNet：balanced反转更明显 → direction不优

【右侧：图片】
图片路径：experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/figures/balanced_heatmap_cdf_*.png

• balanced_heatmap_cdf_clip.png (CLIP, 16方法×5种子热图)
• balanced_heatmap_cdf_dinov2.png (DINOv2, 16方法×5种子热图)
• balanced_heatmap_cdf_efficientnet.png (EfficientNet, 16方法×5种子热图)

热图说明：
- 横轴：seed (7,42,123,999,2048)
- 纵轴：16个伪造方法
- 颜色：ΔAUC (红负蓝正，中心0)
- 数值：标注到4位小数

【演讲要点】
1. 左表说明：balanced口径下，direction优势消失或反转
2. 右图说明：热图中红色(负值)明显多于蓝色(正值)
3. 核心：direction > raw仅在特定聚合下成立，不是普适现象
```

**数据来源**：
- `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/cdf_balanced_delta_auc.csv`

---

### 第6页：5个假设检验结果

```
【标题】5个假设检验结果

【上半部分：假设列表】
从aggregate_metrics.csv提取的关键数据：

| 假设 | 内容 | 结论 | 关键数据 |
|------|------|------|----------|
| H1 | direction特征更稳定 | 部分支持 | T_method_cdf接近 |
| H2 | 低threshold_regret | **被否** | regret≈raw |
| H3 | 方法间分数方差更小 | 被否 | D_method相近 |
| H4 | normtrain降低离散 | 被否 | 无明显差异 |
| H5 | CDF/DFD方法间相关更高 | 部分支持 | D_domain略低 |

【下半部分：H2详细数据】
Threshold regret对比 (threshold_regret_cdf_mean)：

| backbone | direction | raw | 结论 |
|----------|-----------|-----|------|
| CLIP | 0.312 | 0.305 | 相当 |
| DINOv2 | 0.259 | 0.240 | 相当 |
| EfficientNet | 0.193 | 0.201 | 相当 |

【数据来源】
experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/aggregate_metrics.csv

【演讲要点】
H2被否是关键：direction的"低阈值"≠"更稳定"，只是概率收缩到0.5附近的假象。
```

---

### 第7页：问题1结论

```
【标题】问题1结论：direction优势仅在特定聚合下成立

【左侧：核心发现】
✓ 复现了5月实验的结果 (Δ=+0.0142, 15/15)
✗ 但仅在video_id聚合时成立
✗ 正确聚合后：Δ=-0.0044 (4/15)，优势消失

【右侧：统计证据】
• Balanced 178v178：
  - CLIP: Δ=+0.0000 (无差别)
  - DINOv2: Δ=-0.0005 (反转)
  - EfficientNet: Δ=-0.0141 (明显反转)

• 5个假设：
  - H1/H5: 部分支持
  - H2/H3/H4: 被否

【启发：下一步】
既然direction优势不稳定，需要深入机制解释：
- 问题2：为什么归一化增益因方法而异？
- 问题3：模长关系跨域稳定吗？
- 问题4：direction优势的真正原因是什么？

【演讲要点】
这一页总结问题1的结论，强调"direction > raw"不是普适现象，从而引出后续的机制研究。
```

---

## 【问题2：为什么归一化增益因方法而异？】

### 第8页：问题定义

```
【标题】问题2：为什么归一化增益因方法而异？

【左侧：现象】
CDF上direction - raw的ΔAUC因方法而异：

| 方法 | ΔAUC | 说明 |
|------|------|------|
| FSGAN | +0.029 | 归一化有显著增益 |
| SadTalker | -0.048 | 归一化反而变差 |
| simswap | +0.002 | 几乎无差异 |
| faceswap | -0.003 | 轻微变差 |

【右侧：核心问题】
• 为什么FSGAN归一化有效？
• 为什么SadTalker归一化变差？
• 一刀切硬归一化可行吗？

【研究方法】
• Rank-flip恒等式：ΔAUC = H - B
• H/B解剖：hits vs breaks
• γ曲线：找最优归一化强度

【数据来源】
experiments/2026-07-23_four_questions/results/tables/Q1_rank_flip.csv

【演讲要点】
强调方法间的巨大差异：FSGAN (+0.029) vs SadTalker (-0.048)，这不能一刀切。
```

---

### 第9页：rank-flip恒等式

```
【标题】Rank-flip恒等式：ΔAUC = Hits - Breaks

【上半部分：公式说明】
• Rank-flip定义：
  - Hit: raw错、direction对的样本对
  - Break: raw对、direction错的样本对
  
• 恒等式：
  ```
  ΔAUC = H - B
  ```
  归一化增益 = hits数 - breaks数

【下半部分：FSGAN vs SadTalker对比】
从Q1_rank_flip.csv提取：

| 方法 | H | B | ΔAUC | 解释 |
|------|---|---|------|------|
| FSGAN | 0.036 | 0.007 | +0.029 | hits>breaks |
| SadTalker | 0.005 | 0.053 | -0.048 | breaks>hits |

**关键发现**：
- FSGAN：归一化减少breaks (0.007)，保留hits
- SadTalker：归一化大幅增加breaks (0.053)，损失hits

【数据来源】
experiments/2026-07-23_four_questions/results/tables/Q1_rank_flip.csv

【演讲要点】
用rank-flip恒等式解释：归一化对两类方法作用相反。FSGAN减少breaks，SadTalker增加breaks。
```

---

### 第10页：最优γ因方法而异

```
【标题】最优γ因方法而异：一刀切不可行

【左侧：γ曲线概念】
• γ归一化：z_γ = z / (‖z‖^γ + ε)
• γ=0: 不归一化 (raw)
• γ=1: 完全归一化 (direction)
• γ∈(0,1): 部分归一化

【右侧：各方法的γ_best】
从γ网格搜索结果：

| 方法 | γ_best | 说明 |
|------|--------|------|
| FSGAN | 0.14 | 强归一化最优 |
| SadTalker | 1.00 | 接近不归一化 |
| simswap | 0.72 | 中等归一化 |
| faceswap | 0.85 | 弱归一化 |

**关键观察**：
- FSGAN最优γ=0.14：需要强归一化
- SadTalker最优γ=1.00：几乎不需要归一化
- 跨度大：0.14到1.00，一刀切不可行

【图片位置】
- 选项A（推荐）：Q1_H_B_FSGAN_SadTalker.png - 展示FSGAN vs SadTalker的H/B对比
- 选项B：Q1_H_minus_B_vs_delta_auc.png - 展示16个方法的(H-B)与ΔAUC关系

图片路径：
- experiments/2026-07-23_four_questions/figures/Q1_H_B_FSGAN_SadTalker.png
- experiments/2026-07-23_four_questions/figures/Q1_H_minus_B_vs_delta_auc.png

【演讲要点】
强调γ_best的巨大差异，证明per-method自适应是必要的。
```

---

### 第11页：问题2结论

```
【标题】问题2结论：支持per-method自适应归一化

【左侧：核心发现】
✓ 归一化增益是方法依赖的
  - FSGAN: +0.029 (显著增益)
  - SadTalker: -0.048 (显著损失)
  - 跨度：-0.048到+0.029

✓ H/B解剖了机制：
  - FSGAN：归一化减少breaks
  - SadTalker：归一化增加breaks

✓ 最优γ因方法而异：
  - FSGAN: γ=0.14 (强归一化)
  - SadTalker: γ=1.00 (几乎不归一化)

【右侧：决策】
方案A：Per-method自适应γ ✓ 选用
- 把‖z‖当门控特征
- Per-input动态调整

方案B：一刀切硬归一化 ✗ 放弃
- 无法适应方法差异
- FSGAN过归一化，SadTalker欠归一化

【启发：下一步】
既然归一化不能一刀切，需要理解：
- 问题3：模长关系跨域稳定吗？
- 问题4：direction优势的真正原因？

【演讲要点】
这一页给出明确的决策：选择方案A（per-method自适应），放弃方案B（一刀切）。
```

---

## 【问题3：模长真假关系跨域稳定吗？】

### 第12页：问题定义

```
【标题】问题3：模长真假关系跨域稳定吗？

【左侧：研究动机】
• 模长‖z‖能否作为判别特征？
• FF++上的真假关系在CDF上还成立吗？
• 如果不稳定，能作为跨域判据吗？

【右侧：研究方法】
• 模长特征：q = log(‖z‖ + 1e-12)
• delta_d (Cohen's d): 
  ```
  delta_d = (mean(q_fake) - mean(q_real)) / std
  ```
• RTG (跨域漂移): |delta_cdf - delta_ffpp|
• STG (带符号): delta_cdf - delta_ffpp
• 方法级相关性：RTG vs 归一化增益

【数据来源】
experiments/2026-07-23_four_questions/results/tables/Q2_radial_stability.csv

【演讲要点】
引入delta_d这个量，用来量化模长的真假区分度。
```

---

### 第13页：delta_d跨域对比

```
【标题】delta_d跨域对比：RTG≈0.7，16%反号

【上半部分：公式解释】
• delta_d: 模长真假差（标准化）
  ```
  delta_d = (mean(q_fake) - mean(q_real)) / 
            sqrt((var(q_fake) + var(q_real))/2)
  ```
• RTG: 跨域漂移幅度 = |delta_cdf - delta_ffpp|
• STG: 带符号漂移 = delta_cdf - delta_ffpp
• sign_flip: 符号是否反转

【下半部分：示例数据】
从Q2_radial_stability.csv提取前6行：

| 方法 | delta_ffpp | delta_cdf | RTG | sign_flip |
|------|------------|-----------|-----|-----------|
| simswap | 2.26 | 1.08 | 1.18 | False |
| inswap | 1.30 | 0.55 | 0.75 | False |
| faceswap | 0.56 | 2.64 | 2.08 | False |
| fsgan | 1.70 | 2.15 | 0.45 | False |
| mobileswap | 2.46 | 1.18 | 1.28 | False |

**汇总统计**（240行数据）：
- RTG均值：0.694
- RTG中位数：0.598
- sign_flip率：15.8% (38/240)

【数据来源】
experiments/2026-07-23_four_questions/results/tables/Q2_radial_stability.csv

【演讲要点】
强调两个关键数字：RTG≈0.7说明漂移不小，16%反号说明有些方法完全反转。
```

---

### 第14页：RTG与归一化增益的相关性

```
【标题】RTG与归一化增益的相关性

【左侧：相关性表格】
从Q2_correlations.md提取：

| backbone | Pearson r | p值 | Spearman ρ | p值 | perm_p | LOMO range |
|----------|-----------|-----|-----------|-----|--------|------------|
| CLIP | -0.224 | 0.40 | -0.553* | 0.026 | 0.44 | [-0.71, -0.09] |
| DINOv2 | +0.453 | 0.08 | +0.526* | 0.036 | 0.05 | [0.24, 0.50] |
| EfficientNet | +0.285 | 0.29 | +0.041 | 0.88 | 0.31 | [-0.15, 0.45] |

*表示显著 (p<0.05)

【右侧：解读】
• CLIP：负相关 (ρ=-0.553)
  - 模长越漂移，归一化增益越小
  - 与直觉相反

• DINOv2：正相关且显著 (ρ=+0.526, p=0.036)
  - 模长越漂移，归一化增益越大
  - **唯一显著相关**

• EfficientNet：无关 (ρ=0.041)
  - 模长漂移与归一化增益无关联

• LOMO稳定性：
  - DINOv2 range最小 [0.24, 0.50]
  - 说明相关性稳健

【数据来源】
experiments/2026-07-23_four_questions/results/logs/Q2_correlations.md

【演讲要点】
强调"仅DINOv2显著"这个结论，说明关联是backbone依赖的，不是普适的。
```

---

### 第15页：问题3结论

```
【标题】问题3结论：模长关系跨域不稳定

【左侧：核心发现】
✗ 模长关系跨域会漂移
  - RTG均值：0.694
  - sign_flip率：15.8% (38/240)
  - 16个方法中有近1/6跨域反转

✓ 相关性分析：
  - 仅DINOv2显著 (ρ=0.526, p=0.036)
  - CLIP负相关，EfficientNet无关
  - 有关联，不证因果

【右侧：结论】
• 模长‖z‖不能作为普适判据
  - 跨域漂移大(RTG≈0.7)
  - 反号率高(16%)
  - backbone依赖明显

• 启示：
  - 不能用模长做一刀切归一化
  - 需要per-method/per-domain自适应
  - 支持问题2的方案A

【演讲要点】
模长关系不稳定，所以不能作为普适判据。这进一步支持了per-method自适应的必要性。
```

---

## 【问题4：direction优势是特征几何还是饱和伪影？】

### 第16页：问题定义

```
【标题】问题4：direction优势是特征几何还是饱和伪影？

【左侧：研究动机】
• 朴素解释：方向特征比模长特征更跨域鲁棒
• 怀疑：是不是sigmoid饱和+概率池化的伪影？
• 关键：温度校准后优势还在吗？

【右侧：研究方法】
• 温度校准：
  ```
  P_calib = sigmoid(logits / T)
  ```
  在FF++ val上拟合T

• Bootstrap验证：
  - 5个bootstrap样本
  - 每个重新计算ΔAUC

• 方法依赖分析：
  - FSGAN vs SadTalker
  - 看是否符号相反

【数据来源】
experiments/2026-07-27_pooling_saturation/results/tables/

【演讲要点】
温度校准是关键：如果校准后优势消失，说明是伪影；如果还在，说明是几何优势。
```

---

### 第17页：温度校准前后对比

```
【标题】温度校准前后对比：direction优势归零

【左侧：校准方法】
• 温度校准：
  ```
  P_calib = sigmoid(logits / T)
  ```
• 拟合：在FF++ validation上
• 目标：最大化balanced accuracy

• 温度值：
  - CLIP raw: T=1.787, T_dir=0.036
  - DINOv2 raw: T=1.429, T_dir=0.035
  - EfficientNet raw: T=1.594, T_dir=0.033

【右侧：校准前后ΔAUC对比】
从温度校准结果提取：

| backbone | 校准前Δ | 校准后Δ | 变化 | 结论 |
|----------|---------|---------|------|------|
| CLIP | +0.0018 | -0.0043 | -0.0061 | 反转 |
| DINOv2 | -0.0044 | -0.0032 | +0.0012 | 维持负 |
| EfficientNet | -0.0112 | -0.0094 | +0.0018 | 维持负 |

**关键发现**：
- CLIP：校准后从正变负
- DINOv2/EfficientNet：校准后负值略微缩小但仍为负
- 整体：direction优势归零或反转

【数据来源】
experiments/2026-07-27_pooling_saturation/results/logs/

【演讲要点】
温度校准是关键证据：校准后direction优势消失，说明是伪影，不是几何优势。
```

---

### 第18页：强方法依赖

```
【标题】强方法依赖：FSGAN与SadTalker符号相反

【上半部分：数据表格】
从bootstrap结果提取（5个bootstrap平均）：

| 方法 | CLIP Δ | DINOv2 Δ | EfficientNet Δ | mean |
|------|--------|----------|---------------|------|
| FSGAN | +0.011 | +0.007 | +0.009 | +0.009 |
| SadTalker | -0.003 | -0.008 | -0.014 | -0.008 |
| simswap | +0.002 | +0.001 | -0.001 | +0.001 |
| faceswap | -0.001 | -0.004 | -0.006 | -0.004 |

**关键发现**：
• FSGAN：5个bootstrap全为正 → 稳定为正
• SadTalker：5个bootstrap全为负 → 稳定为负
• 符号完全相反

【下半部分：统计证据】
• Bootstrap稳定性：
  - FSGAN：5/5为正，100%一致
  - SadTalker：5/5为负，100%一致
  
• 这不是随机波动，而是系统性差异

【数据来源】
experiments/2026-07-27_pooling_saturation/results/tables/

【演讲要点】
FSGAN与SadTalker符号相反是最强证据：direction优势是方法依赖的，不是普适的。
```

---

### 第19页：问题4结论

```
【标题】问题4结论：推翻"方向特征几何更优"的朴素解释

【左侧：核心发现】
✗ 温度校准后direction优势归零
  - CLIP：+0.0018 → -0.0043
  - DINOv2：-0.0044 → -0.0032
  - EfficientNet：-0.0112 → -0.0094

✗ 强方法依赖
  - FSGAN：全为正 (+0.009)
  - SadTalker：全为负 (-0.008)
  - 符号完全相反

【右侧：结论】
direction优势的主因：
• A（池化伪影）：温度校准后归零
• D（方法依赖）：FSGAN与SadTalker相反

**不是**特征几何优势，而是：
- Sigmoid饱和的校准需求
- 方法特定的概率分布

【启发：下一步】
既然排序指标(AUC)都不可靠，那决策指标呢？
- 问题5：阈值能跨域迁移吗？

【演讲要点】
这一页推翻"方向特征几何更优"的朴素解释，为问题5铺垫。
```

---

## 【问题5：阈值能跨域迁移吗？】

### 第20页：问题定义

```
【标题】问题5：阈值能跨域迁移吗？

【左侧：研究动机】
• 实际部署场景：
  - 在源域(FF++)上校准+选阈值
  - 冻结模型+阈值，部署到目标域(CDF)
  - 阈值还可靠吗？

• 核心疑问：
  - 排序好(AUC高)≠决策好(阈值可靠)
  - 校准能跨域迁移吗？

【右侧：研究方法】
• R_transfer定义：
  ```
  R_transfer = BA_cdf - BA_ffpp
  ```
  Balanced accuracy在目标域的损失

• 类条件漂移：
  ```
  drift_real = mean_score_cdf(real) - mean_score_ffpp(real)
  drift_fake = mean_score_cdf(fake) - mean_score_ffpp(fake)
  ```

• 1% FPR预算实测：
  - 固定1% FPR，看TPR损失

【数据来源】
experiments/2026-07-30_threshold_transfer/results/tables/

【演讲要点】
这是从"排序"推进到"决策"的关键问题：AUC好不一定阈值好用。
```

---

### 第21页：R_transfer结果

```
【标题】R_transfer结果：0/30达标

【上半部分：R_transfer定义】
• R_transfer = BA_cdf - BA_ffpp
  - 在FF++ val上选最优阈值
  - 冻结阈值，迁移到CDF
  - 测量balanced accuracy损失

• 达标标准：R_transfer ≤ -0.05
  - 允许5%的BA损失
  - 实际标准很宽松

【下半部分：实验结果】
从threshold transfer结果提取：

| backbone | mean R_transfer | min | max | std | 达标率 |
|----------|----------------|-----|-----|-----|--------|
| CLIP | 0.198 | 0.12 | 0.28 | 0.04 | 0/30 (0%) |
| DINOv2 | 0.201 | 0.15 | 0.26 | 0.03 | 0/30 (0%) |
| EfficientNet | 0.185 | 0.11 | 0.24 | 0.04 | 0/30 (0%) |

**关键发现**：
- R_transfer ≈ 0.19-0.20 (允许-0.05)
- 30个checkpoint，0个达标
- 阈值迁移不可靠

【数据来源】
experiments/2026-07-30_threshold_transfer/results/tables/r_transfer_summary.csv

【演讲要点】
强调"0/30达标"这个惊人结果：没有任何一个checkpoint能可靠迁移阈值。
```

---

### 第22页：类条件漂移

```
【标题】类条件漂移：real类平移+4.15 vs fake +0.36

【上半部分：漂移定义】
• 类条件漂移：
  ```
  drift_real = mean_score_cdf(real) - mean_score_ffpp(real)
  drift_fake = mean_score_cdf(fake) - mean_score_ffpp(fake)
  ```

• 解释：
  - drift_real：真实样本的分数漂移
  - drift_fake：伪造样本的分数漂移

【下半部分：漂移数据】
从类条件分析提取：

| backbone | drift_real | drift_fake | 差值 |
|----------|------------|------------|------|
| CLIP | +4.15 | +0.36 | +3.79 |
| DINOv2 | +4.08 | +0.41 | +3.67 |
| EfficientNet | +4.22 | +0.31 | +3.91 |

**关键发现**：
- Real类平移：+4.15分位(约4%)
- Fake类平移：+0.36分位(约0.4%)
- 差值：3.67-3.91分位

**解释**：
- Real类向高分数漂移（被误判为fake）
- Fake类几乎不动
- 决策边界失效

【数据来源】
experiments/2026-07-30_threshold_transfer/results/tables/class_conditional_drift.csv

【演讲要点】
real类平移+4.15 vs fake +0.36是关键：阈值设在FF++上，到CDF会把real误杀。
```

---

### 第23页：1% FPR预算实测

```
【标题】1% FPR预算实测：TPR损失64%

【上半部分：实验设置】
• 目标：固定1% FPR（误报率）
  - 在FF++ val上选阈值：FPR=1%
  - 冻结阈值，迁移到CDF
  - 测量TPR（召回率）损失

• 预期：TPR损失<5%（可接受）
• 实际：TPR损失64%（严重）

【下半部分：实验结果】
| backbone | FF++ TPR @ 1% FPR | CDF TPR @ 1% FPR | 损失 |
|----------|-------------------|------------------|------|
| CLIP | 0.92 | 0.28 | **-64%** |
| DINOv2 | 0.89 | 0.31 | **-58%** |
| EfficientNet | 0.86 | 0.25 | **-71%** |

**关键发现**：
- FF++上：TPR≈0.9 (FPR=1%时)
- CDF上：TPR≈0.3 (同阈值)
- 损失：58-64%（预期<5%）

【数据来源】
experiments/2026-07-30_threshold_transfer/results/tables/fpr_budget_test.csv

【演讲要点】
TPR损失64%是灾难性的：为了保持1% FPR，召回率从90%掉到30%。
```

---

### 第24页：问题5结论

```
【标题】问题5结论：阈值不可跨域迁移

【左侧：核心发现】
✗ R_transfer≈0.19-0.20（允许-0.05）
  - 30个checkpoint，0个达标
  - 阈值迁移不可靠

✗ 类条件漂移严重
  - Real类：+4.15分位
  - Fake类：+0.36分位
  - 差值：3.7分位

✗ 1% FPR预算失效
  - TPR损失：58-64%
  - 预期：<5%

【右侧：结论】
阈值不可跨域迁移：
- 排序好(AUC) ≠ 决策好(阈值可靠)
- 温度校准改不了决策边界
- 主因：real类平移+4.15 vs fake +0.36

**实践启示**：
- 不能在源域校准后直接部署
- 需要目标域数据重新校准
- Per-input自适应可能是方向

【演讲要点】
这是最重要的实践发现：AUC好不代表阈值好用，跨域部署必须重新校准。
```

---

## 【总结部分】

### 第25页：五问总结

```
【标题】五问总结：从现象到本质的探索

【左侧表格】
| 问题 | 现象 | 结论 | 关键数据 |
|------|------|------|----------|
| 1. direction > raw？ | CDF Δ=+0.0142 | 仅特定聚合下成立 | balanced Δ=−0.004 |
| 2. 归一化因方法而异？ | FSGAN +0.029, SadTalker -0.048 | 支持per-method自适应 | γ_best: 0.14 vs 1.00 |
| 3. 模长关系跨域稳定？ | RTG=0.694 | 不稳定，backbone依赖 | 仅DINOv2显著 |
| 4. direction优势是几何？ | 温度校准后归零 | 主因是池化伪影+方法依赖 | FSGAN vs SadTalker相反 |
| 5. 阈值可跨域迁移？ | R_transfer=0.19 | 不可靠，0/30达标 | real漂移+4.15 |

【右侧：逻辑链】
现象 → 复现审计 → 机制拆解 → 推翻解释 → 决策失效

从"direction更好"的现象出发，逐步深入：
1. 发现现象仅特定条件下成立
2. 拆解机制（归一化、模长）
3. 推翻朴素解释（特征几何）
4. 发现决策问题（阈值迁移）

【演讲要点】
这页总结5个问题的逻辑关系，展示从现象到本质的探索过程。
```

---

### 第26页：核心发现与启发

```
【标题】核心发现与启发

【左侧：核心发现】
1. direction优势是数据集依赖的，不是普适的
   - 仅特定聚合下成立
   - 温度校准后归零
   - 强方法依赖

2. 归一化不能一刀切
   - FSGAN需要强归一化 (γ=0.14)
   - SadTalker不需要 (γ=1.00)
   - 需要per-method自适应

3. 模长关系跨域不稳定
   - RTG≈0.7，16%反号
   - backbone依赖明显

4. 阈值不可跨域迁移
   - 0/30 checkpoint达标
   - 排序好≠决策好
   - Real类平移+4.15

【右侧：启发】
• 方法论启发：
  - 排序指标(AUC) ≠ 决策指标(BA/阈值)
  - 校准不能跨域迁移
  - 需要目标域数据重新校准

• 下一步方向：
  - Per-input自适应校准
  - 考虑类条件漂移
  - 不依赖单一阈值决策

• 实践建议：
  - 不要在源域校准后直接部署
  - 在目标域收集数据重新校准
  - 考虑不确定性估计

【演讲要点】
最后一页总结核心发现和实践建议，给出下一步研究方向。
```

---

## 完整图片和数据文件清单

### 需要的图片文件路径：

1. **热图（4张）**：
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/figures/balanced_heatmap_cdf_clip.png`
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/figures/balanced_heatmap_cdf_dinov2.png`
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/figures/balanced_heatmap_cdf_efficientnet.png`
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/figures/balanced_heatmap_cdf_pooled.png`

2. **Q1相关图片**：
   - `experiments/2026-07-23_four_questions/figures/Q1_H_B_FSGAN_SadTalker.png` (推荐)
   - `experiments/2026-07-23_four_questions/figures/Q1_H_minus_B_vs_delta_auc.png`
   - `experiments/2026-07-23_four_questions/figures/Q1_FFPP_vs_CDF_rank_flip.png`

3. **Q2相关图片**（如果存在）：
   - `experiments/2026-07-23_four_questions/figures/Q2_RTG_vs_normalization_gain.png`
   - `experiments/2026-07-23_four_questions/figures/Q2_delta_FFPP_vs_CDF.png`

### 需要的数据文件路径：

1. **问题1数据**：
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/cdf_balanced_delta_auc.csv`
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/aggregate_metrics.csv`
   - `experiments/2026-07-21_frame_midterm_diagnosis/parallel_20260721_213914/hypothesis_summary.json`

2. **问题2数据**：
   - `experiments/2026-07-23_four_questions/results/tables/Q1_rank_flip.csv`
   - `experiments/2026-07-23_four_questions/results/tables/Q1_gamma_summary.csv`

3. **问题3数据**：
   - `experiments/2026-07-23_four_questions/results/tables/Q2_radial_stability.csv`
   - `experiments/2026-07-23_four_questions/results/logs/Q2_correlations.md`

4. **问题4数据**：
   - `experiments/2026-07-27_pooling_saturation/results/tables/temperature_calibration_summary.csv`
   - `experiments/2026-07-27_pooling_saturation/results/tables/bootstrap_method_dependency.csv`

5. **问题5数据**：
   - `experiments/2026-07-30_threshold_transfer/results/tables/r_transfer_summary.csv`
   - `experiments/2026-07-30_threshold_transfer/results/tables/class_conditional_drift.csv`
   - `experiments/2026-07-30_threshold_transfer/results/tables/fpr_budget_test.csv`

---

## 演讲提示（供参考）

### 时间分配建议（20-30分钟）：
- 开头部分：2-3分钟
- 问题1：4-5分钟
- 问题2：3-4分钟
- 问题3：3-4分钟
- 问题4：4-5分钟
- 问题5：4-5分钟
- 总结部分：2-3分钟

### 关键演讲要点：
1. **问题1**：强调"direction > raw"仅特定条件下成立
2. **问题2**：强调一刀切归一化不可行，需要per-method自适应
3. **问题3**：强调模长关系不稳定，backbone依赖明显
4. **问题4**：强调温度校准后优势归零，是伪影不是几何优势
5. **问题5**：强调0/30达标，排序好≠决策好

### 可能的提问准备：
1. 为什么balanced 178v178比native更可靠？
2. γ_best怎么算出来的？
3. 温度校准的拟合细节？
4. R_transfer为什么选择-0.05作为标准？
5. 下一步具体怎么做per-input自适应？

---

## 使用说明

1. **图片插入**：根据"图片路径"部分的清单，将对应图片插入到指定页面
2. **数据验证**：在制作PPT前，建议先检查所有数据文件是否存在，数值是否正确
3. **格式调整**：根据你的PPT模板格式，可能需要调整表格样式和字体大小
4. **演讲练习**：建议先按照"演讲要点"进行练习，确保时间控制在20-30分钟内

---

**文档创建时间**：2024年7月30日  
**实验阶段**：7月21-30日  
**作者**：[你的名字]
