# E3：T1–T5 两域完整结果与基线说明

本表汇总已经完成的第一阶段，未启动新模型训练或新评估。全部数字来自审计后的 3 种子（42、7、2048）、FF++/CDF、11 条件共 66 组；表内采用均值 ± 种子样本标准差，以百分数显示，标准差不是置信区间。

## 原始 VLM 基线是否做过

没有原始、未改造 SmolVLM-256M-Instruct 的正式深伪检测基线；也没有保持原始输出方式的任务微调对照或同系列更大模型对照。当前已有的 H1 是 CLIP 视觉真假基线，E3 是移除词表输出头、增加共享标量头后微调语言模块的模型。工程 backbone smoke 标记为 smoke=true/reportable=false，不能当原始 VLM 的科学基线。

因此当前结果只能否定具体的 E3 模型、适配与训练组合在本协议下同时达标；不能定位为模型太小、原始 VLM 不适合深伪检测、共享评分头不适合，或微调损伤语言能力。模型容量/架构不匹配是合理待验证解释。

## 指标与任务

train/seen/held-out 是语言表达集合；seen 测试仍使用测试视频，不是训练集成绩。held-out 使用未见语言表达；部分任务两种集合的变体数量不同，不能把全部差异归因于措辞。T1/T3 主指标为 AUROC，T2/T4 为 BA（平衡准确率），T5 为三类宏召回率（macro BA）。各任务的数值不可直接当作同一种准确率平均。

| 任务 | 含义 | 主指标 |
| --- | --- | --- |
| T1 | 真假描述二选一 | auroc |
| T2 | 真假问句 Yes/No（含极性） | balanced_accuracy |
| T3 | 多组真假措辞二选一 | auroc |
| T4 | 换脸/表情驱动家族 Yes/No | balanced_accuracy |
| T5 | 原片/换脸/表情驱动三分类 | macro_balanced_accuracy |

## Seen / held-out 主结果

| 任务 | FF++ seen | FF++ held-out | CDF seen | CDF held-out |
| --- | --- | --- | --- | --- |
| T1 | 87.43 ± 0.98 | 85.23 ± 0.45 | 81.22 ± 0.72 | 80.34 ± 0.65 |
| T2 | 77.92 ± 0.73 | 59.08 ± 2.31 | 66.09 ± 1.20 | 54.55 ± 2.73 |
| T3 | 87.03 ± 1.04 | 85.35 ± 1.27 | 80.66 ± 0.72 | 79.54 ± 0.61 |
| T4 | 72.52 ± 0.38 | 48.45 ± 5.08 | 68.36 ± 0.45 | 50.36 ± 2.06 |
| T5 | 71.99 ± 1.26 | 43.98 ± 6.40 | 61.85 ± 1.89 | 41.31 ± 5.04 |

## Seen / held-out：统一 BA 口径

T1/T2/T3 用映射后的真假概率在 0.5 阈值判断，T4 用 Yes 概率在 0.5 阈值判断，T5 用三类最大概率。二分类随机水平为 50%，三分类宏召回率随机水平为 33.33%。这些 BA 是补充描述，不改变预注册的 T1/T3 AUROC 门槛。

| 任务 | FF++ seen | FF++ held-out | CDF seen | CDF held-out |
| --- | --- | --- | --- | --- |
| T1 | 79.97 ± 1.49 | 77.94 ± 1.09 | 66.88 ± 1.44 | 66.23 ± 1.12 |
| T2 | 77.92 ± 0.73 | 59.08 ± 2.31 | 66.09 ± 1.20 | 54.55 ± 2.73 |
| T3 | 79.15 ± 1.68 | 72.28 ± 3.68 | 66.69 ± 2.09 | 57.83 ± 0.33 |
| T4 | 72.52 ± 0.38 | 48.45 ± 5.08 | 68.36 ± 0.45 | 50.36 ± 2.06 |
| T5 | 71.99 ± 1.26 | 43.98 ± 6.40 | 61.85 ± 1.89 | 41.31 ± 5.04 |

## FF++ 全条件主指标

以下仍按 T1/T3 AUROC、T2/T4 BA、T5 macro BA 显示。无意义条件列的是控制本身的表现，不是有意义减控制的增益；— 表示该条件没有评估此任务，不能理解为 0 分。

| 条件 | T1 AUROC | T2 BA | T3 AUROC | T4 BA | T5 macro BA |
| --- | --- | --- | --- | --- | --- |
| 固定 H1 真假提示 | 87.34 ± 1.10 | — | — | — | — |
| seen 表达 | 87.43 ± 0.98 | 77.92 ± 0.73 | 87.03 ± 1.04 | 72.52 ± 0.38 | 71.99 ± 1.26 |
| held-out 表达 | 85.23 ± 0.45 | 59.08 ± 2.31 | 85.35 ± 1.27 | 48.45 ± 5.08 | 43.98 ± 6.40 |
| 反事实配对问题 | — | 59.60 ± 2.29 | — | 48.55 ± 4.98 | — |
| 普通无意义候选 | 46.78 ± 1.74 | 49.67 ± 1.32 | 48.15 ± 1.14 | 49.50 ± 0.27 | 32.52 ± 0.71 |
| 普通无意义问题 | — | 50.40 ± 0.34 | — | 48.46 ± 0.26 | — |
| 肯定/否定四问句聚合 | — | 64.14 ± 1.10 | — | — | — |
| 组合描述 | 86.13 ± 1.45 | — | — | — | 61.43 ± 2.61 |
| 长度匹配无意义候选 | 50.72 ± 0.84 | 48.58 ± 0.43 | 56.46 ± 4.67 | 50.61 ± 1.64 | 29.67 ± 1.27 |
| 长度匹配无意义问题 | — | 50.31 ± 1.21 | — | 49.63 ± 0.20 | — |

## CDF 全条件主指标

以下仍按 T1/T3 AUROC、T2/T4 BA、T5 macro BA 显示。无意义条件列的是控制本身的表现，不是有意义减控制的增益；— 表示该条件没有评估此任务，不能理解为 0 分。

| 条件 | T1 AUROC | T2 BA | T3 AUROC | T4 BA | T5 macro BA |
| --- | --- | --- | --- | --- | --- |
| 固定 H1 真假提示 | 81.86 ± 0.38 | — | — | — | — |
| seen 表达 | 81.22 ± 0.72 | 66.09 ± 1.20 | 80.66 ± 0.72 | 68.36 ± 0.45 | 61.85 ± 1.89 |
| held-out 表达 | 80.34 ± 0.65 | 54.55 ± 2.73 | 79.54 ± 0.61 | 50.36 ± 2.06 | 41.31 ± 5.04 |
| 反事实配对问题 | — | 54.58 ± 2.71 | — | 50.46 ± 2.20 | — |
| 普通无意义候选 | 49.90 ± 0.79 | 49.93 ± 0.17 | 50.01 ± 0.80 | 50.10 ± 0.07 | 33.08 ± 1.18 |
| 普通无意义问题 | — | 49.79 ± 0.37 | — | 49.74 ± 0.19 | — |
| 肯定/否定四问句聚合 | — | 55.94 ± 2.44 | — | — | — |
| 组合描述 | 80.33 ± 1.56 | — | — | — | 45.07 ± 4.23 |
| 长度匹配无意义候选 | 49.55 ± 0.20 | 49.78 ± 0.37 | 53.93 ± 3.73 | 50.19 ± 1.12 | 31.55 ± 1.63 |
| 长度匹配无意义问题 | — | 49.77 ± 0.14 | — | 49.96 ± 0.05 | — |

## 显式否定逐问句

四问句聚合分数不能替代逐句诊断；以下均为映射回真假标签的 BA。

| 问句 | FF++ BA | CDF BA |
| --- | --- | --- |
| Is this image authentic? | 75.16 ± 1.72 | 59.15 ± 3.31 |
| Is this image not authentic? | 74.94 ± 4.19 | 66.71 ± 4.53 |
| Is this image manipulated? | 77.72 ± 2.74 | 63.81 ± 1.91 |
| Is this image not manipulated? | 28.73 ± 5.52 | 34.07 ± 2.33 |

表中问句为简写；实际冻结提示均包含 Is it true that this facial image ...，详见 configs/diagnostics.yaml。

## 组合与关系诊断

| 域 | 任务 | 组合 BA / macro BA | 原 held-out | 原条件减组合（百分点） | 通过 |
| --- | --- | --- | --- | --- | --- |
| ffpp | T1 | 70.35 | 77.94 | 7.58 | False |
| ffpp | T5 | 61.43 | 43.98 | -17.45 | True |
| cdf | T1 | 56.87 | 66.23 | 9.36 | False |
| cdf | T5 | 45.07 | 41.31 | -3.76 | False |

| 指标 | FF++ | CDF |
| --- | --- | --- |
| 极性一致性 | 21.24% | 10.14% |
| 真实家族排序 | 46.08% | 54.98% |
| authentic 肯定/否定互补率 | 79.70% | 82.18% |
| manipulated 肯定/否定互补率 | 30.50% | 25.45% |

换序测试没有逐任务拆分的正式指标：每域每种子 57,344 次候选比较；两域最大概率误差均为 1.1920928955e−7，硬预测一致率均为 1.0。不能填造成五个独立任务成绩。

## 普通文本控制增益（G6）

增益为有意义文本减控制，单位为百分点。T1/T3 用 AUROC，其余用 BA/macro BA。下界是 10,000 次配对 bootstrap 的第 2.5 百分位；通过要求增益至少 10 点且下界大于 0。

| 域/控制/任务 | 增益 | 下界 | 通过 |
| --- | --- | --- | --- |
| ffpp/meaningless_candidates/T1 | 38.45 | 33.58 | True |
| ffpp/meaningless_candidates/T2 | 9.41 | 7.91 | False |
| ffpp/meaningless_candidates/T3 | 37.20 | 33.78 | True |
| ffpp/meaningless_candidates/T4 | -1.05 | -1.57 | False |
| ffpp/meaningless_candidates/T5 | 11.45 | 8.92 | True |
| ffpp/meaningless_questions/T2 | 8.68 | 7.35 | False |
| ffpp/meaningless_questions/T4 | -0.01 | -0.27 | False |
| cdf/meaningless_candidates/T1 | 30.44 | 26.02 | True |
| cdf/meaningless_candidates/T2 | 4.63 | 3.42 | False |
| cdf/meaningless_candidates/T3 | 29.53 | 26.48 | True |
| cdf/meaningless_candidates/T4 | 0.26 | -0.03 | False |
| cdf/meaningless_candidates/T5 | 8.22 | 5.77 | False |
| cdf/meaningless_questions/T2 | 4.76 | 3.80 | False |
| cdf/meaningless_questions/T4 | 0.61 | 0.50 | False |

## 长度匹配文本控制增益

所有二分类任务用 BA，T5 用 macro BA；下界为第 5 百分位，即单侧 95%。与上一张表的 T1/T3 指标不同，不可直接比较大小。

| 域 | 控制 | 任务 | 增益（百分点） | 下界（百分点） | 通过 |
| --- | --- | --- | --- | --- | --- |
| ffpp | meaningless_matched_candidates | T1 | 27.30 | 24.02 | True |
| ffpp | meaningless_matched_candidates | T2 | 10.50 | 9.10 | True |
| ffpp | meaningless_matched_candidates | T3 | 19.56 | 16.83 | True |
| ffpp | meaningless_matched_candidates | T4 | -2.16 | -2.61 | False |
| ffpp | meaningless_matched_candidates | T5 | 14.31 | 12.21 | True |
| ffpp | meaningless_matched_questions | T2 | 8.77 | 7.62 | False |
| ffpp | meaningless_matched_questions | T4 | -1.18 | -1.41 | False |
| cdf | meaningless_matched_candidates | T1 | 16.54 | 13.48 | True |
| cdf | meaningless_matched_candidates | T2 | 4.77 | 3.77 | False |
| cdf | meaningless_matched_candidates | T3 | 7.33 | 5.54 | False |
| cdf | meaningless_matched_candidates | T4 | 0.16 | -0.08 | False |
| cdf | meaningless_matched_candidates | T5 | 9.75 | 7.89 | False |
| cdf | meaningless_matched_questions | T2 | 4.78 | 3.96 | False |
| cdf | meaningless_matched_questions | T4 | 0.39 | 0.30 | False |

## 比较与适用范围

此前报告中 E3 固定真假 AUROC 0.8734（FF++）/0.8186（CDF）对应 fixed_h1 提示；本表的 seen / held-out T1 使用不同冻结语言条件，两组数字不矛盾。H1 固定真假 AUROC 为 0.9985/0.9113。H1 没有可直接填入此表的 T2–T5 语言任务成绩。

两域均为 DF40 派生的物理视频划分，包含同样 16 种伪造方法；不是官方原始 FF++/CDF benchmark，也不能解释为未见伪造方法泛化。每种子 FF++ 覆盖 3,049 视频，CDF 覆盖 9,901 视频。视频—语言变体行数见 CSV，不把变体当成独立物理视频；三种子没有做预测集成。

## 详细产物

- t1_t5_all_recorded_metrics_by_seed.csv：全部 66 组已记录的指标行，含逐种子、逐方法、控制增益与换序等原始记录及来源哈希。
- t1_t5_all_conditions_summary.csv：各条件各任务总体指标的三种子均值、SD及三个原始值。
- t1_t5_all_conditions_ba_summary.csv：从逐视频预测独立重算的各条件 BA/macro BA，含各个种子值。
- t1_t5_negation_question_summary.csv：四个肯定/否定问句逐项 BA。
- t1_t5_full_results_manifest.json：所有输入及本次导出产物的 SHA256。

以上 CSV/manifest 均在 results/summary 下；最终科学判定仍以 gate_status.json 为准，未修改冻结门槛。
