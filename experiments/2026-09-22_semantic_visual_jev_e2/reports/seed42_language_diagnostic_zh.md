# seed42 FF++ 语言退化事后诊断

本诊断仅描述 **seed42、FF++、已冻结 checkpoint** 的 seen/heldout 视频结果。它不用于选模、修改阈值、选择提示语或替代 G2–G6；不推断因果，也不推广到其他种子或域。未启动 GPU，未修改科学源代码、配置或门槛。

按现有 `video×语言变体` 行等权重重算：T2 使用 canonical `prob_fake≥0.5` 对 `label`；T4 使用 `p_yes≥0.5` 对 `target_semantic==yes`；T5 使用 camera/swap/reenact 概率最大项，对三类召回率取宏平均。下表与原 metrics JSON 的六个主指标逐项一致（绝对误差 <1e−12）。

| 任务 | Seen | Heldout | 变化（百分点） | Seen/Heldout 计分行数 |
|---|---:|---:|---:|---:|
| T2 | 95.171% | 67.155% | -28.016 | 24026/24026 |
| T4 | 85.537% | 62.977% | -22.561 | 12196/12196 |
| T5 | 89.101% | 63.936% | -25.165 | 6098/3049 |

T1 AUROC 为 0.998451→0.998438，T3 为 0.998173→0.998372；当前退化主要见于这些语义分类任务的阈值判别表现。AUROC 与 balanced accuracy 衡量内容不同，不能据此断言所有判别任务都不受语言变化影响；视觉编码器权重本身在本轮是冻结的。

全部 heldout 二元问答变体如下，未按效果筛选。Formal 的候选为 `affirmative.` / `negative.`；Statement 为 `that is correct.` / `that is incorrect.`（前者 yes，后者 no）。Seen 的答案为 `yes.` / `no.` 和 `yes, it is.` / `no, it is not.`；完整 seen 与 heldout 原文均已写入 CSV。

| 任务/问题 ID | 冻结问题原文 | Formal BA | Statement BA |
|---|---|---:|---:|
| T2/t2_real_q1_heldout | Was this face captured without forensic modification? | 49.231% | 50.602% |
| T2/t2_real_q2_heldout | Is the depicted face genuine and untouched? | 51.079% | 91.165% |
| T2/t2_fake_q1_heldout | Does this face contain artificial manipulation? | 53.200% | 53.086% |
| T2/t2_fake_q2_heldout | Has the facial content been tampered with? | 96.611% | 92.442% |
| T4/t4_swap_q_heldout | Was the person's facial identity replaced with another? | 54.394% | 54.668% |
| T4/t4_reenact_q_heldout | Was this facial performance driven or reenacted? | 50.045% | 52.024% |

**T2：退化分布不均匀。** Seen 的八个变体 BA 为 93.767%–96.331%。Heldout 的 `fake_q1` 两套答案都约 53%，假类召回仅 6.17%–6.40%；`fake_q2` 仍为 92.44%–96.61%。`real_q1` 两套答案都约 50%，真实类召回为 0%/2.88%。`real_q2` 则对 Formal 的真实类召回仅 2.16%，对 Statement 为 82.61%。因此可描述为某些问题上同时恶化、另一个问题上不同答案表述存在较大表现差异；不能把整体下降归结为所有 heldout 文字同样失效，也不能单独归因于问题或候选。

**T4：四个 heldout 变体均下降。** Seen 四变体 BA 为 79.112%–87.478%；heldout 为 50.045%–54.668%。Reenact 问题几乎总答 no（yes 召回 1/1118 和 47/1118）；swap 问题偏答 yes（no 召回 152/1258 和 135/1258）。这是两种答案集均出现、且随问题变化的答案偏向。按问题分组的 BA 为 reenact **0.510345053**、swap **0.545309683**（各 6,098 行），都低于合并后的 **0.629767417**。这来自问题与目标类别分布的差异：reenact 问题有 no/yes=3,862/2,236 行，swap 问题有 no/yes=2,516/3,582 行；合并 BA 对所有 no 行和所有 yes 行各计算一次召回，**不是**先求每道问题 BA 再平均。该现象不表示汇总代码错误，也不应据此改变 gate 口径。

作为事后参照，仅按问题回答“swap 恒 yes、reenact 恒 no”，完全不使用模型概率，在同一 heldout T4 表上得到 BA **0.610597231**、accuracy **0.610364054**。其 no 召回为 3,862/6,378，yes 召回为 3,582/5,818；即使每道问题内恒定回答的 BA 都为 0.5，合并 BA 仍可超过 0.5。该参照仅揭示当前问题/标签比例下的汇总现象，不能证明模型采用该机制，也不替代预注册 G5 或任何门槛。

**T5：heldout 仅一套问题和候选，无法再区分不同 heldout 表述。** 问题为 `Which process most likely produced the face shown here?`；候选 camera=`a naturally recorded face with no digital facial edit.`，swap=`a face whose identity was replaced by a different person.`，reenact=`a face whose motion or expression was artificially driven.`。Camera、swap、reenact 召回分别为 130/140=92.857%、1769/1791=98.772%、2/1118=0.179%；1115 个 reenact 视频被判为 swap。Seen 两套候选的宏 BA 分别为 89.703%、88.499%。下降在类别层面集中于 reenact，但此处不能证明某句文本导致了该混淆。

完整 T5 heldout 混淆矩阵如下（行为真值、列为预测）：

| 真值 / 预测 | camera | swap | reenact |
|---|---:|---:|---:|
| camera | 130 | 9 | 1 |
| swap | 22 | 1769 | 0 |
| reenact | 1 | 1115 | 2 |

完整 CSV 共 **59 行**：模型总体 6 行、所有 question_id 14 行、所有候选集 11 行、所有 variant_id 27 行，另加上述仅按问题回答的事后参照 1 行；`predictor` 区分真实 E2 输出与事后参照；每行包含实际问题/候选原文、计分行数、唯一物理视频数、帧贡献、类别支持/召回、预测分布和混淆矩阵。不同分组层级重复描述同一批数据，不能将其行数相加当作样本量。混淆矩阵按 `class_order`，行为真值、列为预测。

这些行复用相同的 3,049 个物理视频，且有些语言变体只聚合其中一部分帧；T2 部分视频未覆盖全部变体。分组间样本或帧数并非完全相同；seen→heldout 同时改变问题和候选文字，不是单因素对照。本诊断不提供新的显著性检验或置信区间。正式跨种子证据仍以预定 G2–G6 及其原始计分规则为准。

[完整诊断 CSV](/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2/results/progress/seed42_ffpp_language_diagnostic.csv) · [冻结语言库](/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml) · [Seen 视频](/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation/seed_42/seen/ffpp_videos.csv) · [Heldout 视频](/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation/seed_42/heldout/ffpp_videos.csv)

生成时间（UTC）：2026-09-23T04:41:19.347028+00:00。语言库 SHA-256 `8faa86be0ae4c9ad74a95d4a0bb358c79a1271fefa3eb42d15f36d50a0705811` 与评估 provenance 一致；checkpoint SHA-256 `4a8a588b095436dd7dd99cff2b8900b10d4ef0dd9309b17c92571fd6fc3b60d5`。
