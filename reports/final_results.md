# DF40 v1.1 正式实验结论与交付

15 个正式模型、四角色完整缓存、30 个匹配头与 A–D 分析全部完成，并通过独立终态验收。**本轮主要发现是排序质量与源阈值迁移表现分离，且计分效应依赖骨干、种子和读出。** CDF 视频 RL 的 CLIP / DINO / EfficientNet 宏 AUC 均值分别为 0.8847 / 0.7435 / 0.8611，而 FPR 为 90.17% / 10.28% / 75.14%，对应宏 TPR 为 99.70% / 41.65% / 97.30%。这些阈值各自仅用源真实校准池冻结，源 FPR 为5%，没有用 CDF 调阈值。较高 AUC 不等于该源阈值可控制目标误报，较低 FPR 也须与检出率一起解释。

按原 head 做方向归一化后，CDF 视频 mean-logit 宏 AUC 的五种子均值在三个骨干均下降：CLIP 约24.61pp、DINO 1.40pp、EfficientNet 1.67pp；这不是对其他训练算法或任意归一化方法的结论。DINO 的 RL→DL 降低 FPR 约1.13pp，同时降低宏 TPR 约11.51pp，属于误报与检出的权衡。raw 下概率平均相对 logit 平均的效应在 CLIP/EfficientNet 为负、DINO 均值为正，没有在本轮中获得跨骨干通用的聚合赢家。

本轮是固定协议的**回顾性验证**：研究历史已看过 CDF，不称为前瞻盲测。所有31方法均参与源训练，不声称未见方法泛化。三个骨干各自报告5种子的点、均值、样本SD、范围，不合并为15个同分布种子；以下 ± 均表示跨训练种子的样本SD。

## 视频主表：21 FS/FR 方法

| 骨干 | 测试域 | RL 宏AUC | RP 宏AUC | DL 宏AUC | DP 宏AUC |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | 0.9934 ± 0.0014 | 0.9926 ± 0.0013 | 0.9925 ± 0.0014 | 0.9924 ± 0.0014 |
| CLIP ViT-B/16 | CDF | 0.8847 ± 0.0117 | 0.8818 ± 0.0114 | 0.6386 ± 0.1256 | 0.6389 ± 0.1254 |
| DINOv2 ViT-B/14 | FF | 0.8547 ± 0.0457 | 0.8494 ± 0.0416 | 0.8409 ± 0.0475 | 0.8409 ± 0.0474 |
| DINOv2 ViT-B/14 | CDF | 0.7435 ± 0.0500 | 0.7455 ± 0.0427 | 0.7295 ± 0.0518 | 0.7296 ± 0.0518 |
| EfficientNet-B0 | FF | 0.9866 ± 0.0025 | 0.9817 ± 0.0027 | 0.9838 ± 0.0022 | 0.9838 ± 0.0022 |
| EfficientNet-B0 | CDF | 0.8611 ± 0.0163 | 0.8537 ± 0.0146 | 0.8444 ± 0.0149 | 0.8448 ± 0.0147 |

RL=raw帧logit平均，RP=raw帧sigmoid概率平均；DL/DP使用单位方向特征并保持原 head。视频固定8个不同RGB帧。完整五种子与min/max见 [metrics_by_seed.csv](final_report/metrics_by_seed.csv)、[metrics_five_seed_summary.csv](final_report/metrics_five_seed_summary.csv)。

## 各流程的源 FPR5 冻结阈值

FPR、宏TPR、宏BA单位为百分比；± 的单位为百分点。每个骨干×种子×流程独立使用同一源real校准池确定阈值，原样迁移。宏BA=(宏TPR+1−FPR)/2。Real=0/Negative，Fake=1/Positive，score越大越像Fake，预测Fake条件为score≥阈值。

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

FPR的真实集合在同域/同模型/同流程下共享，不按伪造方法重复计为独立测量。未做概率校准；这里是源real的阈值校准，Brier改善不能证明概率已校准。TPR95缺少完整独立fake校准覆盖，已明确标记不可用。

## 图像/帧表：31 方法

每个自然视频的8帧总权重1，EFS每独立图像权重1；EFS数字存储桶不是自然视频。图像/帧和视频指标分开。

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

逐方法及FS/FR/EFS结果见 [metrics_by_seed_method.csv](final_report/metrics_by_seed_method.csv)、[image_FS_FR_EFS_by_seed.csv](final_report/image_FS_FR_EFS_by_seed.csv)。native/real/fake/class-balanced Brier完整值保留于主CSV；legacy `real_videos` 字段在图像流程里实际计real行数，不能当作独立视频数。

## 配对计分效应与机制控制

下表为宏AUC差（pp）的均值±SD；末尾为正/负/零的种子次数。Δraw=RP−RL，Δdir=DP−DL，Δnorm,L=DL−RL，Δnorm,P=DP−RP，交互=Δdir−Δraw。5/5同号是本轮观察，不能推广为普遍规律或未经多重校正的显著性结论。

| 骨干 | 测试域 | Δraw | Δdir | Δnorm,L | Δnorm,P | 交互 |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | -0.084 ± 0.054; 0/5/0 | -0.000 ± 0.002; 1/4/0 | -0.099 ± 0.072; 0/5/0 | -0.015 ± 0.077; 2/3/0 | +0.084 ± 0.052; 5/0/0 |
| CLIP ViT-B/16 | CDF | -0.291 ± 0.135; 0/5/0 | +0.033 ± 0.030; 5/0/0 | -24.612 ± 12.113; 0/5/0 | -24.287 ± 12.157; 0/5/0 | +0.324 ± 0.133; 5/0/0 |
| DINOv2 ViT-B/14 | FF | -0.531 ± 0.458; 0/5/0 | +0.004 ± 0.005; 3/2/0 | -1.384 ± 1.379; 0/5/0 | -0.849 ± 1.344; 2/3/0 | +0.535 ± 0.457; 5/0/0 |
| DINOv2 ViT-B/14 | CDF | +0.198 ± 0.941; 3/2/0 | +0.008 ± 0.010; 4/1/0 | -1.403 ± 1.137; 1/4/0 | -1.594 ± 1.469; 0/5/0 | -0.190 ± 0.934; 2/3/0 |
| EfficientNet-B0 | FF | -0.490 ± 0.129; 0/5/0 | -0.000 ± 0.008; 2/3/0 | -0.277 ± 0.174; 0/5/0 | +0.213 ± 0.245; 4/1/0 | +0.490 ± 0.126; 5/0/0 |
| EfficientNet-B0 | CDF | -0.740 ± 0.701; 1/4/0 | +0.044 ± 0.036; 5/0/0 | -1.673 ± 2.150; 1/4/0 | -0.890 ± 1.981; 1/4/0 | +0.784 ± 0.703; 4/1/0 |

固定尺度a=[0.25,0.5,1,2,4]全部报告；c仅由源训练sampling_mass加权长度中位数估计。h、h/c、u、c·u和不同聚合均保留，未按CDF选择尺度。正比例缩放图像/mean-logit排序控制最大AUC偏差=0，统一长度缩放的相应最大偏差=0；同步正仿射变换阈值的全部预测差异为0。3×3 pair转移账本与直接宏AUC差一致，完整表在 [pair_transitions.csv](final_report/pair_transitions.csv)。

聚合后sigmoid理论上保序，但FP64精确端点饱和可产生新ties；本轮记录的post-mean sigmoid最大绝对宏AUC数值差为0.329086。这项实际数值效应单独记录，不能把它与帧级sigmoid和多帧平均不交换混为一谈。近端比例、精确端点比例、ties和所有尺度结果见 [scale_controls.csv](final_report/scale_controls.csv)，长度控制见 [length_controls.csv](final_report/length_controls.csv)。

## 30 个冻结特征匹配头

所有30头收敛，实际迭代范围20–672，上限2000；同一源训练行与sampling_mass，零初始化，全批L-BFGS-B，固定L2=1e-4、不惩罚bias，骨干和normalization状态均冻结。raw头使用h，direction头使用u；未运行可选λ敏感性网格，未用目标标签选择参数。以下为各匹配头的五种子宏AUC；这些重拟合头的结果不复用原head的数值阈值。

| 骨干 | 测试域 | raw L | direction L | raw P | direction P | raw image | direction image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | FF | 0.9910 ± 0.0026 | 0.9899 ± 0.0050 | 0.9923 ± 0.0016 | 0.9902 ± 0.0035 | 0.9790 ± 0.0060 | 0.9759 ± 0.0109 |
| CLIP ViT-B/16 | CDF | 0.7013 ± 0.1610 | 0.6083 ± 0.2243 | 0.7258 ± 0.1390 | 0.6184 ± 0.2148 | 0.6044 ± 0.1756 | 0.5332 ± 0.2359 |
| DINOv2 ViT-B/14 | FF | 0.8600 ± 0.0401 | 0.8457 ± 0.0403 | 0.8544 ± 0.0368 | 0.8439 ± 0.0398 | 0.8210 ± 0.0409 | 0.7766 ± 0.0625 |
| DINOv2 ViT-B/14 | CDF | 0.7540 ± 0.0538 | 0.7340 ± 0.0423 | 0.7553 ± 0.0436 | 0.7371 ± 0.0392 | 0.7194 ± 0.0405 | 0.6798 ± 0.0448 |
| EfficientNet-B0 | FF | 0.9867 ± 0.0025 | 0.9846 ± 0.0020 | 0.9823 ± 0.0027 | 0.9833 ± 0.0019 | 0.9763 ± 0.0023 | 0.9655 ± 0.0032 |
| EfficientNet-B0 | CDF | 0.8514 ± 0.0136 | 0.8018 ± 0.0085 | 0.8445 ± 0.0104 | 0.8102 ± 0.0081 | 0.7987 ± 0.0173 | 0.7213 ± 0.0073 |

CDF上raw与direction重拟合后的mean-logit均值差仍存在；CLIP的差距缩小也伴随raw与direction两边均值下降和较大的种子SD，不能称为方向头救回了目标表现。这里只比较当前表示、线性读出与固定正则化条件，不能推出信息论损失或任何方法使用长度均无用，也不是归一化端到端训练算法对比。所有头的完整五种子、目标指标与收敛状态见 [matched_heads.csv](final_report/matched_heads.csv)。

## 不确定性与图表

2000次测试来源组配对bootstrap与1000次源校准重采样分开，均条件于固定模型、清单、阈值和已声明分组；不合并成总体95%保证。缺任一规定方法时整条宏指标重复无效，不平均缩水方法集合。FF图像宏重复无效次数按seed[7,42,123,2026,3407]为[39,39,39,34,37]，CDF图像为[274,264,247,288,284]，原因是RDDM代理组支持稀少；视频各2000次均有效。未知EFS/音频驱动祖先及CDF/EFS代理组不被认证为人口独立性。详见 [bootstrap_support_structure.json](bootstrap_support_structure.json)、[test_bootstrap.csv](final_report/test_bootstrap.csv)、[calibration_bootstrap.csv](final_report/calibration_bootstrap.csv)。

原6幅图逐byte保留。补充3幅逐种子AUC/FPR配对差图和3幅全观测范围的real相对源阈值加权ECDF（PNG/PDF），全部五种子点保留。原直方图固定bins为[-4,4]，实际被该窗口遗漏的最大加权质量为5.08%，逐panel覆盖见 [relative_score_plot_coverage.csv](final_report/relative_score_plot_coverage.csv)；完整ECDF补充了范围外的分布，不改动任何计分或阈值。

- CLIP ViT-B/16：[主指标稳定性](final_report/clip_vit_b16_five_seed_stability.png)、[逐种子配对差](final_report/clip_vit_b16_paired_flow_differences.png)、[real全范围ECDF](final_report/clip_vit_b16_real_scores_relative_full_range_ECDF.png)。
- DINOv2 ViT-B/14：[主指标稳定性](final_report/dinov2_vit_b14_five_seed_stability.png)、[逐种子配对差](final_report/dinov2_vit_b14_paired_flow_differences.png)、[real全范围ECDF](final_report/dinov2_vit_b14_real_scores_relative_full_range_ECDF.png)。
- EfficientNet-B0：[主指标稳定性](final_report/efficientnet_b0_five_seed_stability.png)、[逐种子配对差](final_report/efficientnet_b0_paired_flow_differences.png)、[real全范围ECDF](final_report/efficientnet_b0_real_scores_relative_full_range_ECDF.png)。

## 数据协议、执行证据与复现

作者修订62份官方JSON与本机索引逐byte/语义一致；独立解析实际路径、realpath/inode和像素证明本地存在内容复用，不能仅由JSON结构同构或同seed basename下结论，也不把尚未认证原图压缩包来源链的本地重复归咎作者原档案。原始数据只读，发布角色与逐条来源/去重处置保留；原v1.1设计SHA保持不变，执行记录另写。已知FF来源隔离、选中RGB角色间零交集；未知祖先范围明确披露。官方说明的跨方法训练/测试规则不构成像素唯一性声明，EFS数字桶属于兼容目录，见 [官方/本机核对](official_local_verification.md)、[作者issue14](https://github.com/YZY-stack/DF40/issues/14#issuecomment-2432880783)、[作者issue16](https://github.com/YZY-stack/DF40/issues/16#issuecomment-2428452229)。

全量1514777路径Byte/RGB/解码审计0失败。冻结train250000、FFtest59935、CDFtest368922、real-cal1120；cal来自140个原FF-val视频/70已知FF pair组，fake-cal=0。训练保留265527关联贡献并合并重复RGB输入，共享SG2/3内容保留两方法贡献，不删除方法。所有31方法包括e4s，e4e/FE排除；CDF从未用于训练、校准或选模型/阈值。CDF real id10_0001只有3个不同RGB，不足8帧而未进入视频主表，不是解码失败。

15模型均为epoch15 final，224/实际与有效batch128、AdamW WD5e-4、三个预定LR、五个固定seed，无EMA/早停/调度。总预算56250000 draws、439650 attempts、439581 successful updates、69 AMP skips。225份真实epoch sampler数组逐一精确重建，同seed的三骨干共享预定抽样序列。Opt/AMP/Python/NumPy/CPU/CUDA RNG完整终态保留；冻结代码未记录逐step AMP skip事件，具体步号和当时scale不可恢复，未补造、补步或重放。小型随机CUDA Opt/AMP/RNG续接一致性已验证；恢复粒度为完成epoch，正式健康作业未人为中断。

CPU回归60项曾通过，三骨干实际GPU资格、隔离pilot及15正式run均通过。两次前置execution失败（3974097预检、3977426 DINO资格）与旧工件保留，旧freeze/错误来源注释/路径resolver修订尝试也保留在归档和原账本；DINO严格确定性CUDA AA插值反传限制在正式队列前修复为CPU FP32可微插值，保留原37×37学习位置网格，随后真实GPU资格与全部DINO正式训练通过。未把pilot或旧实验结果填入主表。

独立验收UTC=2026-10-09T13:12:44.428643+00:00：从raw缓存重算六主流程score/probability、行/标签/权重、源阈值和全部主指标；校准重采样阈值独立重建，测试bootstrap数组/区间/无效次数/配对代数核对；匹配head分数、终态模型/完整状态SHA、阶段输出、负对照和原6PNG核对通过。终态679977个选中路径size/mtime与原审计基线一致，属于变化筛查，不是新的逐Byte/RGB证明。

原supervisor3984143自然结束，正式workers全部退出0，原队列与执行锁已释放，无重复启动或新observer/schedule。原onecat-vllm相同完整ID及unless-stopped policy恢复，实际健康HTTP200；GPU0–3/Jev和onecat-gateway未干预。最初stop超时退出137、OOM=false已按实记录，不称为无异常退出。

可复现源码和固定输入/初始化/配置/环境SHA全部保留。计算代码SHA=`7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5`。完整模型、Opt/AMP/RNG、raw缓存、逐样本score和bootstrap数组在`/home/z/workspace/deepfake_detector_v11_20261007`；项目`artifacts`提供快捷链接。小型报告/图表镜像在`reports/final_report`，供本地阅读和源码包携带。15份完整配置由原canonical config与实际保存identity解析，见 [resolved_formal_run_configs.json](resolved_formal_run_configs.json)，明确标记为终态解析，不冒充训练前日志。

[独立验收凭据](final_independent_acceptance.json) · [源文件终态stat核对](final_selected_source_state_check.json) · [15缓存凭据](completed_run_cache_receipts.json) · [训练预算](formal_training_budget_counts.csv) · [手动复现指南](manual_reproduction.md)

复现使用独立checkout/新artifact_root，固定官方修订副本、源候选和初始权重；`prepare`会写该项目主config。单一CLI仍为`run.py`，见手动指南。原始设计与旧项目/旧论文第三条均未改写，本轮没有更新回放。无执行阻碍；TPR95、未知祖先认证和逐step AMP日志缺口按实披露。
