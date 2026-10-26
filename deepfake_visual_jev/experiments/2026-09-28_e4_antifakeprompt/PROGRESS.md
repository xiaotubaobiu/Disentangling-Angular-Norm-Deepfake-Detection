# E4 AntifakePrompt 第一阶段

目标：官方词表评分与共享标量评分，同骨干同数据微调对照。用户授权执行；禁止子智能体，GPU仅0–3，域仅FF++/CDF。

**当前状态：已按用户“不要做了”停止所有评估及自动续跑，goal已暂停。两组训练完成，正式评估15/72组完成；以下运行记录为历史进度，不表示仍在运行。暂停时结果汇报见`reports/paused_results_report_zh.md`。用户后续请求仅为汇报现有结果，未恢复实验。**

- [x] 固定上游提交，查明官方检测本来采用候选token NLL排序。
- [x] 写实验设计和实施计划（项目 docs/superpowers 下 2026-09-28-antifakeprompt）。
- [x] E2/E3 原路径保存并完成归档盘点，见 ARCHIVE_INDEX.md。
- [x] 权重下载、独立环境、官方原版真实前向验证。
- [x] 评分适配与关键测试、真实模型前向反向smoke；两组优化器真实更新通过。
- [x] 数据清单及协议冻结；双tokenizer长度匹配池预检通过。
- [ ] A1/A2 微调与A0/A1/A2 两域评估。
- [ ] 配对统计、T1–T5全条件报告与第一阶段结论。

已有部分正式E4检测结果，尚无完整阶段结论；下载或smoke不算科学实验结果。第一阶段单seed42、有限训练预算及语言诊断子集；所有范围在设计文档中预先说明，不与E3全量三种子表混用。

执行方式：本人按 executing-plans 与 TDD 完成，用户禁止子智能体优先于技能中的独立reviewer要求。主仓库现有拆分未提交，新增实验目录隔离，不重置旧文件。设计审批沿用用户已明确授权的goal，按上层自主执行要求继续，不重复请求许可。

## 2026-09-28 正式运行

- GPU0: A1 lm_prompt，systemd `visual-jev-e4-train-a1-20260928.service`。
- GPU1: A2 scalar_prompt，systemd `visual-jev-e4-train-a2-20260928.service`。
- GPU2/3: A0 FF++/CDF baseline，systemd `visual-jev-e4-eval-a0-ffpp-20260928.service` / `visual-jev-e4-eval-a0-cdf-20260928.service`。
- 后续本地进程调度：`visual-jev-e4-stage1-continuation-20260928.service`。等待训练成功退出及视觉cache完整，随后GPU0/1分别评A1/A2两域，最后审计全部72组并生成报告，不自动进入下一阶段。此为普通Python进程调度，不使用智能体。
- 20项关键测试通过；A0与A2所有12条件真实7B smoke完成，分开保存在results/smoke_*，不可当正式成绩。
- EVA原生视觉输出FP32，cache保留原精度。
- 原始换序审计同时改变batch分组，smoke出现概率误差约3.6e-3(A0)/4.8e-4(A2)，预测一致率100%；保留原始G3门槛与结果，另有固定batch审计工具作为补充，不能覆盖原始失败。
- 尚无最终E4结论。最后结果必须核验训练、评估与报告退出码和artifact_audit。

### 首个正式baseline（阶段尚未结束）

A0官方checkpoint / official_binary / FF++：3,049视频，AUROC=0.5023326622，BA=0.5158596474，real recall=0.4428571429，fake recall=0.5888621519。只说明公开checkpoint在当前DF40派生FF++划分上直接迁移弱，不能据此判定A1/A2或架构假设。来源 `results/evaluation/A0/official_binary/ffpp_videos.csv`。

### 两组微调完成；正式评估继续

- A1/A2均完成3轮×4096输入，无跳过optimizer更新；均按预定FF++验证CE选中epoch3。完成记录及checkpoint SHA已复核。
- 两组初始soft prompt完全相同；重建共同抽样序列，共12288训练输入，涉及6067个物理视频、7792张唯一帧。不是完整训练库遍历三次，见`results/summary/training_audit.json`。
- GPU0/1分别运行A1/A2，GPU2/3运行A0两域；仍由原有systemd pipeline接续，不使用子智能体。
- 首批FF++正式结果：A0 official AUROC50.23%，A1 official AUROC50.07%；A0 fixed_h1 AUROC46.23%、BA50%、所有视频判real。提示形式敏感，尚不能据此定位容量或结构原因。
- 配对统计新增对语言变体身份、帧计数和重复行的拒绝检查，24项测试通过；对所有smoke控制产物的统计格式预检通过。仅修改尚未运行的统计/报告代码，正式训练/评估来源文件保持原hash。
- 报告将提供三组两域全条件、T1–T5统一BA补表、关系/换序诊断、逐方法原始指标及来源hash；未完成72组前拒绝生成最终报告。阶段结束后停止，不继续压缩/RL。
- 已补充非最终快照`reports/current_detection_snapshot_zh.md`：A0官方CDF AUROC58.30%、BA56.41%；A2官方FF++ AUROC50.38%、BA50%，全部判假，普通准确率95.41%不代表有效检测。A1固定描述FF++ AUROC41.06%、BA50%，全部判真。完整T1–T5未齐，不提前定最终假设。
- 新完成A2固定描述FF++：AUROC54.36%、BA50%、全部判假。A2−A1 AUROC为+13.29个百分点，10000次物理视频bootstrap单侧95%下界+8.08个百分点；相对此弱基线的G1单域门槛通过，但不代表绝对检测能力或整体语义假设通过。证据`results/summary/interim_fixed_h1_ffpp_bootstrap.json`。目前7/72组审计通过，四个GPU评估进程持续运行。
- 后续8/72组审计通过：A0/FF++/seen首组五任务完成，T1 AUROC55.69%、T2 BA55.41%、T3 AUROC51.95%、T4 BA51.98%、T5 macro BA33.33%；T1/T3的BA均50%。652物理视频，不能与完整固定真假3049视频直接当同总体比较。已转入held-out；详见更新后的非最终快照。
- 旧实验归档复核：1119个文件存在、大小与mtime一致，459个配置/源码/报告SHA256一致；其余大型文件仅做存在/大小/mtime核查，没有宣称重新完成全部二进制hash校验。证据`results/summary/archive_preservation_audit.json`。
- 后续10/72组审计通过：A0/CDF/fixed_h1 AUROC48.78%、BA50%、全部判真；A1/FF++/seen五任务主指标为52.15%、54.57%、51.73%、53.27%、33.33%。A1软提示微调未在熟悉问法上形成一致改善，尚待A2同条件与held-out。最新主表见`results/summary/interim_semantic.csv`。
- 补充输入审计（不新增推断、不重评分）：Vicuna下fixed_h1真假候选分别6/10个答案token，官方yes/no为1/1；总计核对20组候选。求和语言似然可能受候选长度影响，但equal-length官方Yes/No同样较弱，因此长度不能作为唯一解释。保留冻结评分与全部结果，见`results/summary/candidate_length_audit.json`。
- 后续11/72组通过审计：A2/FF++/seen完成，T1 AUROC70.27%（BA50%）、T2 BA50.98%、T3 AUROC63.89%（BA51.16%）、T4 BA50.34%、T5 macro BA44.44%。三组同一652视频的seen表已齐并写入快照与interim_semantic.csv。描述型任务排序改善，极性/家族Yes-No仍弱；不能把T1 AUROC当答对率，不能把全量fixed_h1与语义子集的差异仅归因于措辞。全部held-out、CDF及控制仍按原pipeline运行。
- 后续12/72组通过审计：A0/FF++/heldout五任务主指标52.44%、52.84%、49.60%、52.30%、33.33%。新增`scripts/interim_report.py`可直接校验并刷新非最终快照及interim两张CSV，缺失组合显示—；无需重复手工拼表。`report.py --partial`现在始终只审计，即便72组齐全也不会意外启动最终bootstrap/benchmark，正式流程仍由原pipeline独占。针对该边界先复现失败再修复，当前25项测试通过。
- 后续15/72组通过审计：A0/FF++反事实极性一致率67.64%（652视频）、真实家族排序50.20%（512假视频）；A0/CDF/seen五任务主指标61.71%、52.52%、53.77%、52.23%、33.33%；A1/FF++/heldout为50.25%、52.41%、50.42%、48.83%、33.33%。已刷新快照与两张interim CSV，四个评估进程均存活推进，未改变冻结推断代码或训练预算。
