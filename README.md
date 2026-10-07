# DF40 v1.1

31方法、三骨干、五个固定种子，研究聚合评分、源阈值迁移与稳定性。原始数据只读，GPU仅物理4–7，0–3/Jev不参与。科学协议未冻结；目前是准备、审计和部分管线实现，实验尚未完成。

## 目录与入口

- docs/superpowers/specs/：不变的协议原稿、设计、证据。
- docs/superpowers/plans/progress.md：续接实施账本。
- configs/v1.1.json：固定参数、公开权重哈希、未冻结门槛。
- src/deepfake_detector/、tests/：实现及CPU回归。
- artifacts：快捷方式，实际位于 /home/z/workspace/deepfake_detector_v11_20261007。
- reports/：来源提案、容量、内容证据及日志。

从任何目录执行，入口按sys.prefix使用项目环境：

```bash
python /home/z/project/deepfake_detector/run.py status
python /home/z/project/deepfake_detector/run.py prepare
python /home/z/project/deepfake_detector/run.py propose
python /home/z/project/deepfake_detector/run.py evidence
python /home/z/project/deepfake_detector/run.py audit
```

audit有独占锁，完整检查byte/RGB、解码和尺寸，SQLite支持续接。已有审计运行时先查status，不要重复启动。早期错误解析路径快照保留；现行证据只联结当前清单。训练前仍需核验源文件没有变化。

propose/evidence只做计数和关系证据，不生成训练角色清单。content_evidence_summary.json区别部分快照和全量终态。conditional_candidate_counts.csv是未批准的计数方案；EFS数量带未知来源条件，不能称为来源独立的划分。

## 环境与检查

.venv指向artifacts/environment_clean，不继承系统包，不修改其他项目环境。全部依赖锁在requirements.full.lock.txt，核心依赖在requirements.lock.txt；CUDA包源为官方cu124。公开初始化、依赖版本、增强默认值及哈希在artifacts。

```bash
cd /home/z/project/deepfake_detector
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -B -m pytest tests -q
```

CPU测试覆盖真实三骨干前向、8帧选择、共享真实权重、inclusive阈值并列、AdamW/RNG恢复、缓存损坏、无效bootstrap重复及模拟服务恢复。通过CPU测试不代表未执行的GPU训练通过。

## 尚未冻结

保留原发布/物理/已知源角色，所有experiment_role仍未分配。原TEST/CDF不回灌训练或校准。具体角色与副本规则须获用户科学确认后，才可冻结全15次运行共同清单。独立源校准未认证；fake校准0，TPR95不可用。

GPU生命周期代码核验容器与UUID，常规停止并恢复原来运行的onecat-vllm；不更改gateway或重启策略，目前只做模拟测试。GPU探测、试跑、正式队列和完整A–D报告仍须实现并通过门槛；当前CLI不会启动它们。
