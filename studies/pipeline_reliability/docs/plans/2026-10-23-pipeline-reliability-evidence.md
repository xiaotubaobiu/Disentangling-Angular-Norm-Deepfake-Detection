# 新实验计划的本机证据表

日期：2026-10-23。用途：记录研究事实、建议与尚待执行的边界；本轮没有启动新实验。

## 1. 材料接收与完整阅读

附件目录：`/home/z/.codex/attachments/8ece4c13-1cbf-4344-92e7-dd68b0da3313/`。

| 文件 | 行数 | 字节 | 阅读范围 |
|---|---:|---:|---|
| README.md | 152 | 9,810 | 全文 |
| 验证报告_交付边界.md | 29 | 2,749 | 全文 |
| manuscript_en.md | 171 | 23,369 | 全文 |
| 论文初稿_中文.md | 175 | 18,858 | 全文 |
| 实验设计与执行协议_中文.md | 294 | 26,473 | 全文 |
| 合计 | 821 | 81,259 | 五份均完整阅读 |

原样副本与逐文件SHA256见`../reference_materials/2026-10-23/`。本次没有随附`pipeline_study`源码、45份YAML、manifest规范或42项测试工件。本次查到的现有代码是服务器上已完成31方法实验的`deepfake_detector`。

## 2. Research canon / claim-to-evidence

| 编号 | 主张或计划前提 | 类型 | 本机证据 | 处理 |
|---|---|---|---|---|
| E01 | FF++ c23完整视频已在本机 | 已核验 | 视频目录逐文件计数；下载verification.json | 1000真+4×1000假，尚未用于旧训练 |
| E02 | DFD完整视频已在本机 | 已核验数量 | actors/DeepFakeDetection视频目录 | 363真+3068假，未在本轮全帧解码 |
| E03 | FF官方720/140/140划分可用 | 已核验 | artifacts/data/source_splits/及视频路径核对 | H按来源对内再划TRAIN/DEV |
| E04 | 0–3可供新实验安排 | 时点事实 | nvidia-smi：V100-SXM2-32GB、核查时无进程 | 实际执行前再次检查，不能抢占 |
| E05 | 4–7在运行其他任务 | 时点事实 | VLLM计算进程/显存约28GB每卡 | 新队列只允许0–3 |
| E06 | 旧配置使用4–7 | 已核验 | configs/v1.1.json、README | 不能直接启动旧runner |
| E07 | 三种初始化权重存在 | 已核验路径/既有hash | configs/v1.1.json、artifacts/weights、CLIP cache | 新执行时重新校验hash |
| E08 | 当前接口不是open_clip | 已核验代码/环境 | models.py及当前Python包 | 复用OpenAI CLIP/timm/torchvision适配 |
| E09 | DINO确定性位置插值有特殊实现 | 已核验代码 | models.py:dino_position_embedding | 新runner保留并GPU资格验证 |
| E10 | 旧模型全参数训练31方法 | 已核验 | final_results、configs、training.py、验收 | 不改标签成新21方法模型 |
| E11 | 旧训练无独立DEV | 已核验 | freeze_complete、完整角色清单 | 必须从源TRAIN来源组划DEV |
| E12 | D纯视频TRAIN候选719真/14415假 | 已核验旧关联 | training_associations.parquet、frozen_unit_counts | 不是新冻结数；canonical代表行不能代替关联表 |
| E13 | SadTalker存在未知source_ids | 已核验旧关联 | 8视频关联、4个RGB代理组 | 新主清单核验或隔离 |
| E14 | CDF完整池177真/12876假 | 已核验旧冻结 | cdf_test.parquet、八格group_counts | 新角色及祖先处置可能改变数量 |
| E15 | 旧80/97+4132仅代理隔离 | 已核验实现边界 | calibration_table.py、protocol.json | 不宣称完整祖先独立 |
| E16 | DFD祖先可全部认证 | 尚无证据 | 文件名候选2799双侧/269单侧；未找到完整祖先表 | 来源范围如实披露 |
| E17 | 本机有SCRFD五点模型 | 已核验文件/hash | det_10g.onnx与scrfd_10g_bnkps.onnx同hash | detector环境尚未接入 |
| E18 | 新45正式模型已可直接启动 | 不成立 | 本次五附件仅文档；当前训练无新DEV/scheduler/H处理 | 新计划列实现与资格任务，全部NOT_STARTED |
| E19 | 八格需要八倍训练 | 不成立 | 既有8帧诊断同逐帧缓存240格 | D仅15模型，其他CPU计分 |
| E20 | 旧八格已显示所有骨干均值反转 | 不成立 | summary_by_backbone.csv | CDF mean-prob DINO/Eff均值反转，CLIP没有 |
| E21 | 归一化/二次预处理必定有害 | 待检验假设 | 新结果尚无 | 保留零/反向结果，不以预期方向验收 |
| E22 | 30epoch足够或DINO可被优化至某AUC | 待检验假设 | 旧15epoch曲线仅说明旧配方状况 | 只用源DEV执行有限预算开发 |
| E23 | CPU合成42测试代表本机GPU资格 | 不成立 | 附件明确是另一CPU容器；测试工件未收到 | 真实GPU资格仍待执行 |
| E24 | 旧实验可全部放进GitHub备份 | 需限定 | 本机权重/缓存/原片很大，数据有独立使用条件 | 保存代码、文档、现有小型结果；大工件仍在本机 |

## 3. 论证与交付契约

RQ1以共同部署输入比较训练历史，必须同时报告处理覆盖；RQ2以相同模型/原head区分公共刻度和样本排序；RQ3只改变计分及单位并量化相对结论；RQ4在固定测试池比较信息权限不同的阈值。四者共享审计规范，但不要求构成单一因果链。

本计划可以报告已核查的资源和旧证据，可以提出具体配方；不得填入未运行的新结果、承诺祖先独立、把旧CPU/GPU测试升格成新实现资格，或将性能方向作为完成条件。英文/中文稿原文只保存为参考，不改成完成稿。

## 4. 独立设计审查处理

材料审查指出：H共同成功测试队列选择偏差、H共同配方规则、成功/尝试更新歧义、开发成本未界定、完整CAL池隔离、碰撞bootstrap闭包、比较族未固定。

本机计划分别补入：完整B部署主队列+共同矩阵、两历史联合源DEV选配方、固定attempt预算/逐步日志/H统一FP32回退、开发30上限42、一次排除全CAL池关联fake、来源与碰撞连通闭包+draw_id、固定主比较及pointwise区间。更新预算采用了与审查建议不同但明确可执行的固定attempt方案，避免声称成功数天然相等；这属于设计选择，不冒充审查员原建议。

最终builder审查记录在`2026-10-23-pipeline-reliability-review.md`。计划没有要求运行结果必须支持有利叙事。
