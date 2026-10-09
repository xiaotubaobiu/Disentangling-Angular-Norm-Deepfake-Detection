# DF40 v1.1
31方法、3骨干、5种子。原始数据只读；仅物理GPU4–7；0–3/Jev与gateway不参与。本轮没有新方法更新或回放。

## 当前状态与证据
官方修订目录62份JSON与本机索引逐字节和语义一致，原作者图片压缩包的来源尚未认证。完整审计1514777路径、0解码失败。已授权按已知FF来源隔离并保留原发布角色和逐条变更；未知音频驱动/EFS来源保持未知，不依据存储桶虚构来源独立性。
报告看 reports/official_local_verification.md、reports/frozen_unit_counts.csv、reports/execution_receipt_20261007.md；实时状态看 artifacts/execution_status.json、queue_status.json 与 docs/superpowers/plans/progress.md。启动/CPU测试不代表实验完成。

## 目录
configs/ 固定协议与保存的用户授权；src/deepfake_detector/ 管线；tests/ 回归；reports/ 结果/证据。
artifacts -> /home/z/workspace/deepfake_detector_v11_20261007，SSD衍生物。旧快照与失败工件保留，不改写原JSON/图片。
.venv -> artifacts/environment_clean。requirements.full.lock.txt锁定完整环境，初始化权重与SHA在config和prepare_complete.json。

## 单一入口
从任何目录使用：
```bash
python /home/z/project/deepfake_detector/run.py status
python /home/z/project/deepfake_detector/run.py dry-run
python /home/z/project/deepfake_detector/run.py execute
```
execute独占并顺序完成CPU预检、GPU微批资格、独立1epoch完整管线pilot、15正式15epoch模型与全部特征、30匹配头/A–D分析/五种子汇总。资格测试和pilot不作为正式科学结果。通过pilot后最多四个正式run并行。
已存在作业先读status与PID/comm/cwd和锁，不重复启动execute。

手工分阶段复现：
```bash
cd /home/z/project/deepfake_detector
python run.py prepare
python run.py audit
python run.py freeze --approval configs/source_isolation_authorization.json
python run.py preflight
python run.py probe
python run.py pilot
python run.py run
python run.py report
```
已有已审计/已冻结目录不能重新freeze覆盖；新复现需独立artifact_root，保留相同索引/权重/输入哈希。原TEST/CDF不会进入train/cal。audit有锁与SQLite续接；现有完整审计无需重复。

## 数据与计分
训练以唯一RGB内容作为输入，保留每个方法关联贡献后合并sampling_mass，因此共享SG2/SG3内容不会重复算epoch长度，也不会丢弃方法抽样贡献。训练、FFtest、CDFtest关联清单均31方法；真实数据跨方法只保留一池。
21视频方法固定8不同RGB帧；EFS为独立图像，没有伪造视频AUC。完整关联先构建已知FF关系与RGB连接组，再选帧/去重，EFS存储组/CDF视频组只是披露的重采样代理。
real-only FPR5校准使用140原FF-val视频的固定1120帧；独立性仅指已知FF来源及精确内容范围，不认证未知驱动/生成祖先。fake-cal不具完整覆盖，TPR95不可用。阈值不借FFtest/CDF生成。

## 恢复与失败
每完成epoch原子保存模型、optimizer、AMP、Python/NumPy/CPU/CUDA RNG、history、实际样本序列哈希；中途失败从上个完整epoch重做，未承诺batch中点恢复。final.pt只用第15轮。
训练预算固定为每epoch抽取N_pool次、15epoch/final，而不是固定成功optimizer更新数。AMP遇非有限梯度时消费该批数据但跳过参数更新，不回放或补步；没有scheduler。所有模型的draw/attempt/success/AMP skip口径记录在reports/formal_training_budget_counts.csv和formal_training_budget_receipts.json。当前冻结代码未记录逐步AMP事件，无法还原各次skip的具体步号及scale数值；终态scaler和完整恢复状态仍保留，该日志缺口明确披露。

```bash
python run.py resume
```
任一输入/配置/初始化/代码/资格哈希变化拒绝冒充原run恢复。完整checkpoint与每块特征缓存都校验；失败不以旧结果补齐。
仅正常 docker stop --time 60 onecat-vllm / docker start onecat-vllm，不删除容器、不修改restart policy。SIGTERM先终止并join本次子进程再恢复服务。若owner崩溃留下原服务恢复意图，先确认该owner/worker均停止且目标GPU无占用，再：
```bash
python run.py restore
```
restore拒绝活owner或GPU碰撞；再resume。状态仅记录非秘密容器字段，不记录环境变量或凭据。

## 复核与结果
```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -B -m pytest tests -q
```
结果在artifacts/analysis/<backbone>_seed<seed>/与artifacts/report/：
逐样本分数/预测、阈值、逐方法/逐seed/五seed均值SD范围、四流程配对差、尺度/长度/匹配头/3×3转移、分开校准与测试bootstrap（含无效次数）、FS/FR/EFS汇总和稳定性/相对源阈值图。report须从保存分数重算主指标并验证所有交付后才写完成状态。
bootstrap缺少任一规定方法时整条宏平均重复无效，不缩减方法集合；区间只基于有效重复并披露无效次数。冻结分组支持的独立检查在reports/bootstrap_support_structure.json：RDDM图像池声明分组较少，FF/CDF图像存在缺方法重复；视频池的既定抽样方法支持完整。这仅检验声明group_id代理，不认证未知祖先独立性。

源图片状态预检比对全量byte/RGB审计时size/mtime；它是变化筛查，不等于新的逐byte证明。原数据与用户Obsidian v1.1协议保持不变。
