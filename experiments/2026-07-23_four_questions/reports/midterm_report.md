# 中期报告(Q1/Q2/Q3)

> DFD 全程排除。Q1/Q2/Q3 基于 raw-trained checkpoint。

> 重要方法学发现:direction>raw 现象只存在于 video(mean-sigmoid) 层级;帧级 raw>dir。

> Q3 定向局限性:parquet 的 ffpp 为 FF++ test split,定向用其做 1-bit 符号选择(详见 length.py)。

## Q1 rank-flip + 翻转解剖 + γ 曲线

- identity 自检:max identity_error=2.34e-06(应≈0,通过)

- fsgan (CDF, video):H=0.0357 B=0.0065 delta_auc=+0.0292
- sadtalker (CDF, video):H=0.0047 B=0.0525 delta_auc=-0.0478
- CDF video 跨方法:direction 胜的比例=0.45

- fsgan gamma_best_video=0.14 (rho_mean=-0.135)
- sadtalker gamma_best_video=1.00 (rho_mean=-0.049)

## Q2 径向稳定性

- RTG 均值=0.694, sign_flip 比例=0.158

```
# Q2 method-level correlations (RTG vs dir_minus_raw_cdf)

## backbone=clip
- pearson=-0.224 (p=0.404), spearman=-0.553 (p=0.0263), perm_p=0.436, n_methods=16
- LOMO pearson range=[-0.709,-0.089]

## backbone=dinov2
- pearson=0.453 (p=0.0778), spearman=0.526 (p=0.0362), perm_p=0.0519, n_methods=16
- LOMO pearson range=[0.242,0.503]

## backbone=efficientnet
- pearson=0.285 (p=0.285), spearman=0.041 (p=0.88), perm_p=0.31, n_methods=16
- LOMO pearson range=[-0.148,0.447]

```

## Q3 length 稳健性(CDF, 5-seed 均值)

| backbone | micro | macro | video_mean | equalized | video_CI |
|---|---|---|---|---|---|
| clip | 0.852 | 0.852 | 0.812 | 0.871 | [0.779,0.842] |
| dinov2 | 0.635 | 0.631 | 0.639 | 0.646 | [0.595,0.681] |
| efficientnet | 0.761 | 0.760 | 0.700 | 0.744 | [0.663,0.735] |

- CLIP per-method length AUC 范围:[0.68,0.96]

## 初步决策(A 径向—角向自适应方法)

- Q1 γ 分化(FSGAN vs SadTalker):是
- Q3 CLIP length macro>0.7:是
- Q2 RTG↔增益:backbone 依赖(仅 DINOv2 显著正相关)→ 仅关联证据,不证因果

**初步判断:**
- **倾向支持 A(自适应保留模长)**:Q1 证明最优 γ 因方法而异(不能一刀切全删/全留),Q3 证明 CLIP 模长含真实可迁移信号 → 应做径向—角向**自适应加权**而非硬归一化。
- 最终 A/B/C/D 决策待 Q4(是否叠加多方法公共方向监督)。
