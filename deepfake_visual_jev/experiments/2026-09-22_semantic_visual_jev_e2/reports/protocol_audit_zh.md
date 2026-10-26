# E2 验证协议及结果解释审查

审查日期：2026-09-23；限制清单在正式 E2 测试结果出现前形成，文末补记正式完成与审计状态。

## 判定范围

冻结实现按 G1–G6 判定。任一点门槛不达标为 `not_supported`；点门槛全通过但必需置信下界不通过为 `insufficient_evidence`；全部通过为 `supported`。缺失或不合法证据导致 `adjudicated:false`，意味着无法完成判定。

## 已识别限制

1. 设计要求无意义候选长度匹配，但冻结 language_bank 的控制候选为固定四词短语；真实候选长度不同。这是控制设计偏差：G6 降幅不能完全归因于语义消失，也混合文本长度及表面分布变化。本轮不根据测试结果修改冻结词库。
2. 语义指标按视频×语言变体行计算；bootstrap 按 `method_id::video_id` 聚类，不把这些行当独立视频。seen/heldout 的变体数可能不同，部分帧被分入不同变体；宏分数是五个不同主指标的非加权平均，不是普通准确率。
3. 三个种子共享冻结视觉编码器，bootstrap 也条件于这三个固定种子，未估计从头训练视觉编码器或训练种子总体的不确定性。
4. `method_id::video_id` 防止不同方法的视频误合并，但未进一步建模同源视频跨方法相关性。
5. G3 每域审计确定性选出的至多4096个物理样本，穷举其2/3候选排列，并非穷举全域所有帧。
6. CDF真实类仅178/9901；全判fake即有98.2022% accuracy。H1三个种子的真实视频识别数为3、4、4；必须同时报告AUROC及balanced accuracy，不能以高accuracy声称两类均有效识别。
7. 本次按用户确认的范围仅评估FF++和CDF两个主域。DFD未在修复后的正式评估中运行，G3只核查两个主域；结论不覆盖DFD。
8. 结论只覆盖固定英语措辞库、二/三候选、既有十六种操作与当前数据；不能外推为开放语言推理、通用VLM或所有伪造类型成立。

9. E2 相比 H1 同时改变了决策网络（两层 decoder）、训练任务和文字分布；观察到的增益或退化不能唯一归因于“加入语义”。验证对象是当前 E2 组合方案，不是所有语义检测方法。

10. FF++／CDF 是 DF40 中以 FaceForensics++／Celeb-DF 为来源域、共用16种伪造方法的测试切分，并非官方原始 benchmark，也不是未见伪造方法评估。总体 AUROC 为视频级微平均；各方法数量不等（例如 FF++ mobileswap 有858/2909个伪造视频）。[独立数据域审计](../results/progress/dataset_domain_audit.json)复核了索引、三种子H1/E2预测和缓存记录。

## 完整性审计要求

- 根据用户后续收窄的汇报范围，独立核查三种子×FF++／CDF×七条件共42组；修复后的正式结果不包含DFD排列产物，G3只审计FF++／CDF。不能仅凭assess退出0推断42组全部完成，还须核对每组配套文件、来源指纹和原调用及恢复调用的真实退出码。
- 核对H1匹配种子的模型元数据、每组视频键/标签/帧数、G6 meaningful对照与heldout概率及覆盖、T2反事实原始视频概率与一致性结果。
- 保留帧级预测；补固定真假任务的帧/视频诊断、每种子结果和置信区间。
- `results/progress/baseline_independent_audit.csv`保存独立重算的六组H1基线与CSV SHA256。

## 验证证据

本轮全量CPU测试：315 passed, 33 warnings；完整日志在 `results/progress/final_pytest.log`。评估源文件与冻结配置在本次只读审查中未修改。

## 后续运行故障与版本隔离

首次正式评估的 seed42 调用在 FF++ 无意义候选条件发生真实 exit 1。重复物理帧在不同分布式批次下的连续诊断字段 `value` 相差 `5.960464477539063e-8`；它与已有 `drop` 字段表示同一概率差，但去重比较遗漏了 `value` 的现有 `1e-6` 浮点容差。修复提交 `103ab5c` 只把该字段纳入同一容差，未改变模型推理、概率、标签、文本、配对键或G1–G6阈值。回归测试先按真实差值失败、修复后通过；完整CPU测试更新为316 passed、33 warnings、exit0，日志在 `results/progress/post_dedup_fix_pytest.log`。独立审查未发现修复的严重问题。

修复使评估源码包指纹从 `a666788dd7a7ce1b884fbbd2c07a6b3fc256c8f360afd1d4df3b75a1ac1af389` 改为 `0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7`。旧 E2 结果及旧 G1 bootstrap 只作故障取证，不进入最终判定；三种子两个主域七条件全部在 `results/evaluation_primary_v2` 重新产生，并由正式判定器校验新指纹。H1比较器、训练检查点、冻结视觉缓存与预注册门槛保持原值。

修复后的首轮 v2 运行又暴露运行时等待上限：seed7 和 seed42 均在第12组 `meaningless_candidates/CDF` 的相同 NCCL Seq49 单值广播等待 rank0 后处理超过默认600秒，真实 launcher 退出码均为1。前11组各自保留完整产物；失败组没有正式 metrics。广播前该组推理和跨rank收集已结束，rank0在计算/写出大规模控制结果；没有发现模型数值异常或内存耗尽证据。由于缺少活体 Python 栈，不能将耗时精确归因到某一个聚合函数。

恢复调用使用[运行时包装器](../scripts/evaluation_runtime_timeout.py)在执行原评估脚本前仅将 PyTorch NCCL collective timeout 从600秒调整到3600秒。包装器不在覆盖样本、推理、聚合和指标的评估科学源码包内；后者的指纹保持 `0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7`。恢复监督器对20个缺失主域组做无覆盖检查，为每个调用记录包装器/监督器 SHA256、精确命令、四rank日志和真实退出码；原两个失败退出码不改写。

## 正式完成补记

截至 2026-09-23 18:30 KST，FF++／CDF 的 42/42 组与所需配套文件均齐备。恢复监督器[完成记录](../results/evaluation_logs/primary_v2_recovery_20260923T065528Z/completion.json)显示五次调用真实退出码全为0，缺失或空文件列表均为空。三份正式 provenance 的科学源码包指纹一致。冻结的 10,000 次配对 bootstrap 判定为 `not_supported`：G3 通过，G1/G2/G4/G5/G6 存在点门槛失败。[正式门槛状态](../results/summary/gate_status.json)与[中文汇报](e2_validation_summary_zh.md)给出数值。

[独立主域审计](../results/progress/audit_primary_artifacts.json)通过 108 项检查、0 错误，覆盖每组文件/计数、物理视频与帧覆盖、H1 配对、控制条件对应关系。附表[输入清单](../results/summary/report_tables_manifest.json)记录两主域的来源文件；GPU 作业结束后[四卡预检](../results/preflight.json)有效。恢复包装器加入后的[完整 CPU 回归](../results/progress/final_v2_pytest.log)为 318 passed、33 warnings、exit 0；此项比上文早期的 315/316 项记录更新。
