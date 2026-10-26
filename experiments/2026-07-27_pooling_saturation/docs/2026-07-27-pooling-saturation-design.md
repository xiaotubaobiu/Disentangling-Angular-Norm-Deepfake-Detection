# Pooling/Saturation 机制研究 — 设计文档

- 日期:2026-07-27
- 范围:`/home/z/project/deepfake_detection`(分支 `study/four-questions` 之上,或新建分支)
- 上游规范:用户提供的「Pooling/Saturation 实验」全文(本设计为其执行规格化)
- 复用:前序四问题研究的 `src/deepfake_detection/study/` 库(config/io/canonical/auc_pairs/flip_dissect/q4_extract)

## 0. 科学动机(一句话)
前序 Q1 发现:direction>raw 现象**只在视频级 mean(prob_fake) 下成立**,帧级反而 raw>dir。本研究判定该现象究竟来自 **(A) sigmoid 饱和 + 概率聚合** / **(B) 真实径向—角向作用** / **(C) 视频内时间一致性提升** / **(D) 混合且方法依赖**。

## 1. 已确认前提(审计结论)
- `prob_fake == sigmoid(signed_logit)`,max err ~8.9e-8(浮点精度)—— `mean(prob_fake)` 与重算 `mean(sigmoid(s))` 等价。
- raw/direction 行数相等;对齐键须含 `method_id`(real 跨方法复制)—— 已由 `dedup_reals`(infer_mode 感知)+ `_wide` 解决。
- parquet `ffpp` = FF++ **test**(140 real 视频),无 train/val 预测。
- **FF++ val 可恢复且无泄漏**:`build_val_loader` 切 FF++ train 的 videos [586:686](每方法 100),与 train(0:586)、test(parquet 140)均不重叠。manifest `dataset_json_update/{method}_ff.json`,图像在 `/Dataset/deepfake_detection/DF40_all/DF40_train/...` 已确认存在。
- 15 个 raw-trained checkpoint 齐全;`extract_checkpoint` 经一致性门控验证(相对误差 ~1e-6)—— 一次前向得 z → 推导 `s_raw=w·z+b`、`s_dir=w·u+b`。

## 2. 全局口径
- DFD 全程排除(`domain!='dfd'`,弃 `dfd_native`/`DFD_*`)。
- 只用 raw-trained checkpoint;raw/direction 同 checkpoint、同头、同帧。
- s = fake_logit − real_logit;p = sigmoid(s);r=‖z‖;u=z/r。
- γ 部分归一化:`z_γ = r^γ u`;`s_γ = r^γ·(w·u) + b`,**b 从 checkpoint 真实读取**(w=w_fake−w_real, b=b_fake−b_real),不假设 b=0。
- 按 backbone×seed×method×domain 计算;汇总 mean±std。
- GPU 仅 0–5;报告中文,列名/表头英文。

## 3. 实验矩阵
| 实验 | 数据源 | 产出 |
|---|---|---|
| 1 聚合器对照(mean_prob/mean_logit/sigmoid_mean_logit/median/clip/trimmed/top25)| parquet | `Q_pooling_aggregation_auc.csv` + `Delta_prob/Delta_logit/I_pool=Delta_prob−Delta_logit` + 4 图 |
| 2 饱和统计(logit 分位、sat_001/005/010、sigmoid_grad=p(1−p),分 real/fake + 视频级)| parquet | `Q_pooling_saturation_stats.csv` + 3 图;按 backbone Pearson/Spearman/permutation |
| 3 Jensen gap `J=mean σ(s)−σ(mean s)`(J_raw/J_dir/Delta_J)| parquet | `Q_pooling_jensen_gap.csv` + 3 图 |
| 4 温度校准(T_raw/T_dir 在 FF++ **val** 拟合帧级 NLL,video-balanced/real-fake 平衡;Platt 辅助)| **GPU val 补跑** + parquet 评估 | `Q_pooling_temperature.csv`、`Q_pooling_calibrated_auc.csv` + 3 图(含 reliability) |
| 5 尺度匹配对照(median/IQR robust 标准化,在 val 拟合,在 test/CDF 评估)| **GPU val** + parquet | `Q_pooling_scale_matched_auc.csv` |
| 6 各聚合器下 rank-flip H/B + identity(mean_prob/mean_logit/median/trimmed/temp/standardized)| parquet + 实验4 温度 | `Q_pooling_rank_flip_by_aggregator.csv` + 1 图 |
| 7 视频内时间一致性(std/MAD/IQR/q90−q10/P(s>0)/相邻帧差/自相关/离群比例)| parquet | `Q_pooling_temporal_consistency.csv` + 2 图 |
| γ 扫描对照 | parquet + checkpoint w,b | `Q_pooling_gamma_scan.csv` + 2 图(FSGAN/SadTalker × 聚合器) |

## 4. 关键正确性自检(必须通过)
1. **`AUC(mean_logit) == AUC(sigmoid(mean_logit))`** 机器精度(sigmoid 单调)。
2. **rank-flip 恒等 `AUC_dir−AUC_raw == H−B`** 在每个聚合器下、同一批 fake-real 视频 pair 上成立(identity_error≈0)。
3. 温度 T>0;Platt A>0;均在 FF++ val 拟合,**禁止用 test/CDF 拟合任何参数**。
4. clip 值{2,4,6,8}、γ 网格、top25%、trim10% 全部固定;CDF 上 γ_best 仅称 oracle diagnostic。
5. bootstrap:video-cluster paired,≥2000 次,raw/dir/各聚合器同样本;BH 多重比较;帧不作独立样本。

## 5. A/B/C/D 判定逻辑(数据驱动,不预设)
- **A(sigmoid 饱和+概率聚合)**:`Delta_prob>0` 但 `Delta_logit≤0`,且温度/尺度校准后 `Delta_temp≈0`,且 Raw 饱和率显著>Direction,且 `I_pool` 与饱和相关。
- **B(真实径向—角向)**:mean_logit、校准后概率聚合下 Direction 仍稳定提升;γ 最优不随聚合器剧变。
- **C(时间一致性)**:Direction 显著降低视频内方差/离群,且该改善与 mean_logit AUC 提升相关,而饱和/Jensen 不能解释。
- **D(混合且方法依赖)**:FSGAN 与 SadTalker 在多数聚合器下仍呈相反最优 γ/相反 H-B,且上述机制各解释一部分。
- 必须报告所有聚合器下结论是否一致、校准前后是否一致、帧级 vs 视频级是否一致(不得隐藏不一致)。

## 6. 下一步方法建议(由结论唯一决定)
- 情况1(A):校准 + 更合理视频聚合;暂停径向—角向复杂训练。
- 情况2(B):继续径向—角向自适应训练。
- 情况3(C):temporal consistency-aware pooling / 稳健视频聚合。
- 情况4(D):样本级径向可靠性门控(不得用测试时未知的 method label)。

## 7. 执行顺序
1. `reports/pooling_saturation_schema_report.md`(对齐检查 + val 可恢复性 + prob_fake 验证)。
2. dry run:CLIP/seed42/FSGAN+SadTalker/FF+++CDF → 先报 mean_prob vs mean_logit 是否一致。
3. 验证公式后扩展 3×5×16×FF+++CDF;补跑 15 ckpt FF++ val(实验4/5)。
4. `reports/pooling_saturation_final_report.md`(规范结构)+ 下一步建议。

## 8. 交付
脚本 `scripts/06..09`;配置 `configs/pooling_saturation_config.yaml`;表 `outputs/tables/Q_pooling_*.csv`;图 `outputs/figures/Q_pooling_*.png`;报告 `reports/pooling_saturation_*.md`。

## 9. 硬性禁止
不读 DFD;不用 CDF/test 拟合温度/Platt/尺度/γ;不据 CDF 选聚合器;不把 mean-prob 直接解释为方向更好;不把 H-B 恒等当因果;不只报 FSGAN/SadTalker;不隐藏帧级与视频级不一致;不因预期不符改口径。

## 10. 偏差记录(可接受,执行中若触发即记入报告)
- FF++ val 用项目 `build_val_loader` 切片(非官方 140 清单)—— 已与用户确认接受;leak-free,与 checkpoint 训练时 val 一致。
- 若 val 抽取遇阻(图像缺失/manifest 缺):实验4/5 报"校准数据不足",不偷偷用 CDF;A/B/C/D 依据实验1/2/3/6/7 仍可判定。
