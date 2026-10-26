# 四问题实证检验 — 最终报告

> DFD 排除。结论由数据驱动。

## 问题一:为什么 FSGAN 归一化提升、SadTalker 下降?

- **结论:支持**

**核心数值(video mean-sigmoid 视角,CDF 域):**

| 方法 | 域 | H | B | ΔAUC (dir−raw) |
|---|---|---|---|---|
| fsgan | ffpp | 0.0029 | 0.0013 | +0.0016 |
| fsgan | cdf | 0.0357 | 0.0065 | +0.0292 |
| sadtalker | ffpp | 0.0030 | 0.0020 | +0.0010 |
| sadtalker | cdf | 0.0047 | 0.0525 | -0.0478 |

- **H 与 B**:FSGAN 在 CDF 上 H=0.0357 ≫ B=0.0065(归一化有益);SadTalker 反向,B=0.0525 ≫ H=0.0047(保留模长更好)。
- **FF++ 与 CDF 差异**:FF++ 上两方法 ΔAUC 都接近 0(FSGAN +0.0016,SadTalker +0.0010)——in-domain 训练让方向修正无空间;分化只见于 CDF (FSGAN +0.0292 vs SadTalker -0.0478)。
- **backbone 与 seed 稳定性**:CDF video 跨 16 方法 direction 胜的比例=0.45;FSGAN ΔAUC seed-std=0.009,SadTalker seed-std=0.015(方向稳定,符号一致)。
- **反例**:SadTalker 在 CDF video 上 ΔAUC=-0.0478(raw 胜,FSGAN 结论不普适);且帧级 CDF 上 raw>dir 比例=0.68(direction>raw 并非帧级现象)。

**方法学发现:direction>raw 仅在 video(mean-sigmoid) 层级成立。** 最优 γ 因方法而异:FSGAN γ_best_video=0.14 (ρ_mean=-0.135),SadTalker γ_best_video=1.00 (ρ_mean=-0.049)。这否定「一刀切硬归一化」,要求 per-method/per-input 自适应。

**图表:**
- ![Q1_H_B_FSGAN_SadTalker.png](../figures/Q1_H_B_FSGAN_SadTalker.png)
- ![Q1_FFPP_vs_CDF_rank_flip.png](../figures/Q1_FFPP_vs_CDF_rank_flip.png)
- ![Q1_H_minus_B_vs_delta_auc.png](../figures/Q1_H_minus_B_vs_delta_auc.png)

## 问题二:模长真假关系是否跨域稳定?

- **结论:部分支持**

- **delta_ffpp / delta_cdf**:|Δlogit| 在 FF++ 上均值=1.055,CDF 上均值=1.084(CDF 跨域差距更明显)。
- **RTG**:径向真假差距 RTG 均值=0.694,sign_flip 比例=0.158(部分方法真假模长关系翻转,方向不稳定)。
- **与 normalization gain 的相关性(按 backbone,方法级 n=16,自 Q2_correlations.md):**

| backbone | pearson | spearman | perm_p | LOMO range |
|---|---|---|---|---|
| clip | -0.224 (p=0.404) | -0.553 (p=0.026) | 0.436 | [-0.71,-0.09] |
| dinov2 | 0.453 (p=0.078) | 0.526 (p=0.036) | 0.052 | [0.24,0.50] |
| efficientnet | 0.285 (p=0.285) | 0.041 (p=0.880) | 0.310 | [-0.15,0.45] |

仅 DINOv2 显著正相关(spearman=0.526, p=0.036,LOMO [0.24,0.50]);CLIP 负相关,EfficientNet 零相关。→ 关联存在但 backbone 依赖,**不证因果**。
- **反例**:CLIP 负相关、EfficientNet 零相关;即便 DINOv2 也仅方法级 16 个点的关联,无法支撑「模长真假关系跨域稳定」的强主张。

**图表:**
- ![Q2_delta_FFPP_vs_CDF.png](../figures/Q2_delta_FFPP_vs_CDF.png)
- ![Q2_RTG_vs_normalization_gain.png](../figures/Q2_RTG_vs_normalization_gain.png)

## 问题三:CLIP 的 length-only 高 AUC 是否可靠?

- **结论:支持**

- **micro / macro / video / equalized**(CLIP,CDF,5-seed 均值):micro=0.852, macro=0.852, video_mean=0.812, equalized=0.871。video 95% CI=[0.779,0.842]。
- **方法级**:CLIP CDF per-method length-AUC 范围=[0.68,0.96],跨 16 方法均高于 0.5(无方法失败)。
- **seed 稳定性**:CLIP CDF macro 跨 5 seed 的 std=0.017(均值 0.852),稳定。
- **对照**:DINOv2 macro=0.63(≈0.63),EfficientNet macro=0.76——仅 CLIP 的模长含强可迁移信号。
- **定向局限**:parquet 的 ffpp 实为 FF++ **test split**;Q3 仅用其做 **1-bit in-domain 符号选择**(详见 length.py),非跨域定向。故「length-only 高 AUC」是 in-domain 长度判别力的证据,不是 out-of-domain 泛化证据。

**图表:**
- ![Q3_micro_macro_video_auc.png](../figures/Q3_micro_macro_video_auc.png)

## 问题四:多方法对比学习是否具有依据?

- **结论:不支持**

**same-frame alignment vs negative-control(same_chord_all vs mismatch_chord):**

| backbone | domain | same_chord_all | mismatch_chord | same−mismatch | residual_chord | pca_top1 |
|---|---|---|---|---|---|---|
| clip | ffpp | 0.8835 | 0.8891 | -0.0056 | 0.8044 | 0.2167 |
| clip | cdf | 0.6757 | 0.7367 | -0.0610 | 0.6573 | 0.2182 |
| dinov2 | ffpp | 0.7045 | 0.7660 | -0.0615 | 0.6696 | 0.0816 |
| dinov2 | cdf | 0.5955 | 0.6696 | -0.0742 | 0.5832 | 0.0841 |
| efficientnet | ffpp | 0.6506 | 0.7249 | -0.0743 | 0.5797 | 0.0665 |
| efficientnet | cdf | 0.5519 | 0.6254 | -0.0735 | 0.5358 | 0.0724 |

6/6 组合中 same_chord_all ≤ mismatch_chord 的数量=6/6。同帧对齐**未超过**「随机错配」负对照 → 不存在帧特定的公共 fake-fake 方向。
- **residual-after-w**:投影掉线性头方向后的残差仍高度共线(residual_chord 列,均值=0.638),即残差里仍藏着与分类头无关的方向,但它们并不构成跨方法的单一公共方向。
- **PCA top-1**:均值=0.123(CLIP≈0.22 偏高,DINOv2/EfficientNet≈0.07–0.08 极低)——伪影方向在低维上不集中,无主成分可抓。
- **CDF 泛化**:CDF 域 same_chord_all 均值=0.608(< ffpp 的 0.746),跨域进一步衰减。
- **leave-one-method-out**:见 Q4_leave_one_method_out*.csv;LOFO 对齐在各 held-out 方法上分散,无单一方法主导公共方向(任何方法剔除后公共方向不崩溃也不增强)。
- **哪些方法共享**:否——method-cosine 矩阵(Q4_method_cosine_matrix__*.csv)显示方法两两 chord 余弦无稳定高聚簇,face-swap 与 face-reenactment 族**不应**混入同一公共方向(配对审计 §Q4_pairing_audit.md 亦要求 FS/FR 分开)。

**图表:**
- ![Q4_same_frame_vs_random_alignment.png](../figures/Q4_same_frame_vs_random_alignment.png)
- ![Q4_PCA_explained_variance.png](../figures/Q4_PCA_explained_variance.png)
- ![Q4_residual_after_classifier_proj.png](../figures/Q4_residual_after_classifier_proj.png)
- ![Q4_leave_one_method_out.png](../figures/Q4_leave_one_method_out.png)
- ![Q4_method_cosine_matrix__clip_raw_s42__cdf.png](../figures/Q4_method_cosine_matrix__clip_raw_s42__cdf.png)

## 最终决策

- **决策:A — 继续径向—角向自适应方法;放弃 B(同帧多方法对比学习)**

**依据:**
- **Q1**:最优 γ 因方法而异(FSGAN γ≈0.14 vs SadTalker γ≈1.00)→ 应做**自适应加权**,而非硬归一化或全保留。
- **Q3**:CLIP 模长含真实可迁移信号(macro≈0.85,跨方法稳定)→ **应保留模长作为加权特征**,而非删除。
- **Q4**:同帧多方法公共方向在 6/6 组合中不优于负对照,PCA top-1 低,residual-after-w 不构成单一方向 → **不引入全局 fake-fake 对齐**(放弃 B)。

三者一致指向 A:径向(模长)按方法/输入自适应加权 + 角向(方向)保留分类头几何,不叠加多方法公共方向监督。

## 偏差与局限(Provenance)

> 本节记录 4 项对 spec 的已接受偏离与已知局限。所有偏离均不影响宏观结论。

- **Q1 §4.2 video-balanced frame 采样口径**:spec 要求两套口径(native frame / video-balanced)。实现中 `view=="frame"`=native(全量帧),`view=="video"`=spec §4.3 的 mean-sigmoid 视频聚合。§4.2 的「先等权抽视频再等帧采样」独立口径未单独输出(其函数 `sample_pairs_video_balanced` 仅在 Q3 equalized 中使用)。结论不受影响(video 视图已含长视频主导检验的等价信息),但记录此合并。
- **Q3 定向用 FF++ test**:parquet 的 ffpp 仅为 FF++ test split(train/val 特征未落盘),无法按 spec §6.1 回退到 train。改为用 FF++ test 做 1-bit in-domain 符号选择并显著标注。这是对 spec 回退规则的已接受偏离(1-bit,不影响 macro/video/equalized 结论)。
- **Q4 一致性门控阈值**:gate 实际 `tol=1e-3`(spec §7.1 step2 仅要求「≈0」);实测相对误差 ~1e-6,远低于阈值。记录阈值为 1e-3(非 1e-6)。
- **Q4 多 seed 稳定性未测**:Q4 smoke(clip/dinov2/efficientnet × seed42)在 6/6 组合为负,按 spec 的 smoke-first 硬门跳过了 15-checkpoint 全量抽取,故 §7.6 判据②(多 seed 稳定性)未形式化测试。「不支持」结论依据判据①(6/6 same≤mismatch)、④(CDF 衰减)与 residual/PCA 诊断,已充分;但多 seed 稳定性留作后续(见 next_step)。
