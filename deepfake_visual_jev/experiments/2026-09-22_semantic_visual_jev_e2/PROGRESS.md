# E2 实验进度

最近核对：2026-09-23 18:30 KST。当前状态按 v2 正式门槛状态、独立审计、原协调器真实退出记录和恢复协调器完成记录核对。

当前目标：验证 E2-Semantic Visual-Jev 的预注册指标。按用户最新范围，只评估 DF40 的 FF++ 来源域与 CDF 来源域；全新 `results/evaluation_primary_v2` 共三种子 × 两域 × 七条件 = 42 组。本次不运行 DFD，包括 DFD permutation。旧 `results/evaluation` 保留用于历史追溯，不复制、不混入 v2。

## 任务清单

| 阶段 | 状态 | 完成依据 / 下一步 |
|---|---|---|
| 1. 实现、预检与数值重复行修复 | 已完成 | [最终全量 CPU 回归](results/progress/final_v2_pytest.log) 318 passed / 33 warnings，退出码 0；原浮点修复提交 `103ab5c8fd7eedf5d2ba5c1a0f9fd24ad14c2107` |
| 2. 冻结视觉特征缓存 | 已完成 | FF++ 97,036 / CDF 311,984 / DFD 103,339 个唯一物理帧；v2 仅读取两个主域样本 |
| 3. 正式三种子训练 | 已完成，3/3 种子 | 全部训练 9 轮、最佳第 4 轮；seen 验证 CE 连续 5 轮未改善提前停止 |
| 4. v2 主域七条件评估 | 已完成，42/42 组 | 恢复监督器五次调用真实退出码均为 0，全部所需产物非空；原两次 exit 1 保留 |
| 5. v2 配对、指纹与完整性审计 | 已完成 | 独立主域审计 108 项通过、0 错误；三份科学源码指纹一致 |
| 6. Bootstrap、gates 与报告 | 已完成 | 冻结的 10,000 次配对 bootstrap：`not_supported`；G3 通过，其余 G1/G2/G4/G5/G6 失败；[正式汇报](reports/e2_validation_summary_zh.md) |

## 正式训练

| 种子 | 状态 | 完整轮次 | 最佳轮次 | 最佳 seen 验证 CE | 最佳轮次混合验证准确率 |
|---|---|---:|---:|---:|---:|
| 42 | 已完成 | 9 | 4 | 0.268212 | 91.90% |
| 7 | 已完成 | 9 | 4 | 0.266816 | 91.55% |
| 2048 | 已完成 | 9 | 4 | 0.251598 | 91.90% |

- seed 42：[逐轮数值](results/progress/seed_42_history.csv)、[训练完成核验](results/progress/seed_42_completion.json)；checkpoint SHA256 `4a8a588b095436dd7dd99cff2b8900b10d4ef0dd9309b17c92571fd6fc3b60d5`。
- seed 7：[逐轮数值](results/progress/seed_7_history.csv)、[训练完成核验](results/progress/seed_7_completion.json)；checkpoint SHA256 `f560189e99a0a2e90872db3d4f3a6dcac30dcab07c27e7b98ff1930104f949f6`。
- seed 2048：[逐轮数值](results/progress/seed_2048_history.csv)、[训练完成核验](results/progress/seed_2048_completion.json)；checkpoint SHA256 `3a11126be0dd7867a890845cd4d8f9b646f6f41e10a981d00fdfdaac0791f392`。

正式 checkpoint 均通过 `evaluate_semantic.validate_evaluation_checkpoint` 核验。原训练 launcher 会话无法追溯时，真实退出码仍记为 null，以最终 rank0 汇总、逐轮四 rank 诊断和实际 checkpoint 佐证完成，不补写退出 0。训练验证准确率混合 T1–T5，不能直接与 H1 二分类指标相减。

v2 正式 G1：FF++ 来源域 H1/E2 平均 AUROC 为 0.9985308321 / 0.9984874527；CDF 来源域为 0.9112673875 / 0.8221806782；E2−H1 单侧 95% 下界分别为 −0.0002742310 / −0.1026723143。CDF 未通过 −0.01 的非劣效门槛。虽然数值与旧 cohort 阶段结果一致，正式判定只读取 v2 同源产物和新指纹；旧[阶段 G1 证据](results/progress/g1_fixed_task_bootstrap.json)保留作历史记录。[正式 G1–G6 状态](results/summary/gate_status.json)。

## v2 主域评估矩阵

每格为已写出正式 metrics 的条件数 / 7。条件为 fixed_h1、seen、heldout、permutation、counterfactual、meaningless_candidates、meaningless_questions。全部所需配套文件已通过[独立审计](results/progress/audit_primary_artifacts.json)。

| 种子 | FF++ 来源域 | CDF 来源域 |
|---|---|---|
| 42 | 7/7 | 7/7 |
| 7 | 7/7 | 7/7 |
| 2048 | 7/7 | 7/7 |

seed42、seed7 的原 launcher 均在第12组 meaningless_candidates/CDF 的同一 NCCL broadcast 等待 rank0 输出处理超过默认600秒后真实退出1，失败日志保留。恢复运行仅把 PyTorch NCCL collective timeout 提至3600秒，不修改评估科学源码、数据、模型或阈值；包装器与监督器各有独立哈希和命令清单。正式结论读取 42/42 组及其完整审计，不把原失败退出码改写为0。

- 恢复监督器已结束，五次恢复调用全部真实退出 0，所需产物均非空；最多两个四卡 launcher 并发。
- 已观察的实际 launcher 退出码：seed 42: 1, seed 7: 1。
- 已观察的恢复 launcher 真实退出码：seed 7 两次、seed 42 两次、seed 2048 一次，均为0。[恢复完成记录](results/evaluation_logs/primary_v2_recovery_20260923T065528Z/completion.json)含每次调用及产物清单；原失败不改写为0。
- [恢复协调器状态](results/evaluation_logs/current_recovery_run.json)、[恢复逐调用日志](results/evaluation_logs/primary_v2_recovery_20260923T065528Z/)、[42 组及配套产物独立审计](results/progress/audit_primary_artifacts.json)。
- [当前协调器状态](results/evaluation_logs/current_run_v2.json)、[本次命令与逐 rank 日志](results/evaluation_logs/primary_v2_20260923T051934Z/)、[总控制台](results/evaluation_logs/primary_v2_supervisor_console.log)。每个 seed 保留精确 argv/环境、启动 PID、真实 `Popen.wait()` 退出码。

## 运行与来源证据

- GPU 0–3、每卡 batch128、每个 launcher 四 rank，最多两个同时运行。GPU 4–7 未用于本实验。
- 新 cohort 在 2026-09-23 05:19:34 UTC 启动 seed42 和 seed7；两个原调用均遇到默认超时，seed2048由06:55:28 UTC启动的恢复协调器首次派发。
- v2 evaluation source bundle SHA256：`0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7`。每个 `run_provenance.json` 绑定冻结 checkpoint、config、语言清单、缓存清单和该来源指纹；缓存来源协议包含三域 manifest，不表示运行了 DFD。
- GPU 作业结束后四卡[预检](results/preflight.json)有效；[正式附表清单](results/summary/report_tables_manifest.json)覆盖两主域、三种子，帧级诊断不进入门槛。
- 旧 seed42 因浮点舍入重复行比较失败，实际 exit 1；修复仅为已有 1e-6 容差字段补上 `value` 别名。旧 seed7 收到受控 SIGTERM 后实际 exit 1，所有旧 rank 已退出。[停止请求](results/evaluation_logs/parallel_20260923T043530Z/seed7_controlled_stop_request.json)、[真实完成记录](results/evaluation_logs/parallel_20260923T043530Z/semantic_controls_seed_7/completion.json)、[GPU 清空前置核验](results/evaluation_logs/parallel_20260923T043530Z/seed7_controlled_stop_completion.json)。
- 旧资源测量显示 seen/CDF 单 run 峰值进程树 PSS 46.01 GiB、单卡显存 5.78 GiB；双 run 观测单卡约 14.5 GiB，按授权维持最多两个。[资源证据](results/evaluation_logs/resource_profile_seed_42_seen_cdf/summary.json)。
- [协议审计与报告限制](reports/protocol_audit_zh.md)；[实验说明](README.md)。
