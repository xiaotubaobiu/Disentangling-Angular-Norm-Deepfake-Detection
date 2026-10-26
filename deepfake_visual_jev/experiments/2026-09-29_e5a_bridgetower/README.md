# E5-A: BridgeTower semantic qualification

研究问题：固定视觉和文本单模态编码器，仅训练 BridgeTower 跨模态层与原生 ITM head，是否能通过 Jev 式共享候选评分，同时满足 FF++ 真假判别、未见语言表达、极性和伪造家族门槛。

状态：**E5-A STATUS: FAIL**。Stage B 的 held-out T2 BA 为 82.38%，但极性一致率为 77.01%，低于 90% 门槛；候选换序一致率为 100%。原资格验证按协议在 Stage B 停止，结论与审计见 `reports/e5_a_final_report_zh.md` 和 `results/summary/`。用户随后要求补齐 FF++ T1–T5；该失败后的诊断性评估已完成，另见 `reports/e5_a_t1_t5_exploratory_zh.md` 和 `results/t1_t5_exploratory/`。未运行 CDF 或 E5-B/C/D。

权重：`BridgeTower/bridgetower-base-itm-mlm`，revision `6cfb318bb08588808368a9ab1fe26609bc148f1f`。使用 Hugging Face 官方 BridgeTower 模型类；[Microsoft 官方仓库](https://github.com/microsoft/BridgeTower)是架构来源。原始 checkpoint 文件哈希见 `results/protocol/model_loading_audit.json`。

协议：`configs/e5_a.yaml`；训练前物理划分审计：`results/protocol/physical_split_audit.json`。冻结数据来源为 E4 的 `results/protocol/`，只读取 train、val、ffpp_full 和语言清单；不使用 CDF。

正式顺序：

1. `CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python scripts/train.py`
2. 仅训练完成且校验最佳检查点后，在 GPU 0–3 各运行一次 `CUDA_VISIBLE_DEVICES=<编号> ../../.venv/bin/python scripts/evaluate_stage_b.py --shard-count 4 --shard-id <编号>`。每个物理视频固定分配至一个 shard。
3. 四个 shard 都完成后运行 `../../.venv/bin/python scripts/evaluate_stage_b.py --merge --shard-count 4`；合并时复核文件哈希、全量视频和帧覆盖。
4. 若 Stage B 失败：`../../.venv/bin/python scripts/report_stage_b_failure.py` 并结束 E5-A。
5. 仅若 `results/stage_b/stage_b_result.json` 为 `PASS`，才在 GPU 0–3 上按相同分片运行 `scripts/evaluate_stage_c.py --shard-count 4 --shard-id <编号>`，然后 `scripts/evaluate_stage_c.py --merge --shard-count 4` 和 `scripts/report_stage_c.py`。Stage C 程序会拒绝 Stage B 失败的检查点。

在 Stage B 已失败后，用户明确要求补测 T1–T5。使用 `--exploratory-after-stage-b-fail` 在 GPU 0–3 上运行同一评估器，输出隔离到 `results/t1_t5_exploratory/`；原 `results/summary/e5_a_gate_status.json` 仍为 FAIL。诊断报告由 `scripts/report_t1_t5_exploratory.py` 生成。

最初 FP16 尝试在第 1 轮第 576 批出现非有限梯度，未产生正式检查点，日志保留为 `results/training_fp16_failed.log`。随后发现当前 Transformers 5.5 的 Fast tokenizer 将该 checkpoint 的词表解析为逐字符输入；该次未完成的训练已整体隔离于 `results/invalid_fast_tokenizer_training/`。再次审计发现 T4 家族采样虽均衡，Yes/No 目标却不均衡；该次未完成训练隔离于 `results/invalid_t4_balance_training/`。上述运行均未执行测试集，不能作为 E5-A 成绩。正式训练使用官方 RoBERTa slow BPE tokenizer、T4 正反目标各半和 FP32，从原始 BridgeTower 权重重新开始；验证选模规则及门槛保持不变。
