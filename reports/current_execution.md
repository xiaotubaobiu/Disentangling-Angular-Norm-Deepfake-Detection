
2026-10-08T00:09:25.341532+00:00：CLIP ViT-B/16 seed42完整四角色缓存独立核验通过，679977行/512维，全部hash/FP32有限/非零向量/冻结清单顺序/最终head一致，canonical verified_run=true。原worker3990787于2026-10-08T00:02:56.152649+00:00退出0；原supervisor3984143在GPU7自动启动CLIP seed123 PID255008，PPID/cwd/run.py及两锁核对一致。完整run3/15、训练模型4/15；DINO seed7缓存继续，DINO42/Efficient42/CLIP123训练继续。预算/代码/配置未变，无新队列/调度，无最终目标性能。证据reports/completed_run_cache_receipts.json。


2026-10-07T23:58:21.686885+00:00：CLIP ViT-B/16 seed7完整15epoch+四角色缓存独立核验通过，共679977行/512维，块hash/FP32有限值/非零norm/全清单顺序/最终两类head一致；canonical verified_run=true。旧worker3990784退出0，完成事件2026-10-07T23:48:58.939269+00:00；原supervisor3984143在GPU4自动启动EfficientNet seed42 PID245444，PPID/cwd/run.py及两锁正常。完整cache run2/15，训练模型4/15；无目标性能结果。仅记录既有轮转，未重新启动作业或新增调度。证据reports/completed_run_cache_receipts.json；原第一完成模型为EfficientNet/seed7，AMP4次skip实际聚合终态与逐步日志缺口见efficientnet_b0_seed7_AMP_skip_audit.json，无scheduler、数据已消费不回放、不补步，原N_pool×15epoch预算未变。


2026-10-07T22:30:27.568271+00:00：首批四个15epoch/final模型均已独立核验：CLIP seed7 success29305/skip5、DINO seed7 success29307/skip3、Efficient seed7 success29306/skip4、CLIP seed42 success29306/skip4；各draw3750000/attempt29310。全部final/state一致、模型及optimizer有限、AMP/Python/NumPy/CPU/CUDA RNG齐全，15轮采样序列精确重建。训练预算合规，但逐步AMP日志缺口仍明确；不补步、不重训、无新配置。训练模型4/15，完整cache run1/15，分析待全15完整run。原队列3984143继续，DINO seed42已首轮，GPU0–3与gateway未干预。证据reports/completed_model_state_receipts.json、formal_training_budget_counts.csv/receipts.json。


2026-10-07T22:20:16.217732+00:00：22:16平台连接通知后的实际SSH成功；原supervisor3984143/四个当前worker身份、PPID、cwd/run.py和execution/queue两锁均正常。训练完成3/15、完整cache run1/15；CLIP seed7/42已15epoch，DINO seed7为14epoch、seed42为1epoch。缓存持续写入；无实验失败、实际访问/审批阻碍，未重启/重复启动/新建调度。逐模型预算与AMP聚合skip记录继续更新，缺少逐步AMP事件保持明确披露。


2026-10-07T21:58:48.035979+00:00：CLIP ViT-B/16 seed7完成15epoch/final模型核验；draw3750000、attempt29310、success29305、AMP skip5。完整模型/optimizer有限，final与state一致，完整AMP/Python/NumPy/CPU/CUDA RNG及15轮实际序列精确重建通过。训练loss=0.028471196019172668，不作目标性能结果。逐次skip步号/scale未被冻结代码记录，按相同口径明确缺口；未补步/重训。模型训练完成2/15，完整cache run仍1/15；CLIP缓存提取继续。证据reports/completed_model_state_receipts.json与formal_training_budget_counts.csv。


2026-10-07T21:22:35.415156+00:00：首个完整run efficientnet_b0_seed7（15epoch + train250000/FFtest59935/CDFtest368922/cal1120原始缓存，共679977行）独立hash/FP32有限值/1280维/全清单顺序/最终两类head核验通过；canonical verified_run=true。原worker3990786已退出0，完成事件 2026-10-07T21:18:18.404736+00:00；原supervisor3984143随后在GPU6自动启动DINOv2 seed42 PID144861，事件 2026-10-07T21:18:18.406325+00:00；新worker PPID/cwd/run.py和两锁验证一致。完整run为1/15，无最终科学结果。原LLM容器仍exited、restart unless-stopped，等待全GPU队列结束恢复。证据reports/first_complete_cache_receipt.json与first_complete_run_rotation_receipt.json。AMP逐步日志缺口继续披露，不补步。


2026-10-07T21:00:35.985449+00:00 AMP预算核验：efficientnet_b0/seed7，draw=3750000、attempt=29310、success=29306、AMP skip=4。v1.1原文第75行固定N_pool抽样/epoch，第67行15epoch/final，第69行无学习率调度；不是固定成功更新数。无补步/回放/超参更改。49个BN层相对初始化计数均增加29310，跳过optimizer的批仍经过前向；scheduler不存在，LR保持1e-4。AMP初始scale128、最终16384，回退因子0.5。但冻结代码仅保存epoch日志和终态scaler，四次skip具体步号及逐事件scale数值不可追溯；明确作为日志缺口，不称AMP逐步审计完全通过。所有15模型按draw/attempt/success/skip同口径累积记录，见reports/formal_training_budget_counts.csv与formal_training_budget_receipts.json；首模型详细证据efficientnet_b0_seed7_AMP_skip_audit.json。


2026-10-07T20:10:19.114091+00:00：首个完整15轮模型 efficientnet_b0_seed7 已于 2026-10-07T19:53:24.568631+00:00 核验；3,750,000槽位全部序列逐epoch重建匹配，final/state模型一致，optimizer及模型有限，AMP与全RNG齐全。实际optimizer成功29306/尝试29310，AMP跳过4次。最终训练loss=0.03724643358850479，不代表目标域性能。首模型四角色原始特征仍提取，未计完整run；原supervisor3984143/worker3990786继续，未重新启动。证据 reports/first_complete_model_receipt.json；完整state另存artifacts/continuations/20261007_exclusive_resume/first_complete_model。代码/配置/冻结数据保持原身份。

2026-10-07T18:24:14.006966+00:00：实际SSH成功，原supervisor3984143和四worker身份/锁正常；已完成epoch clip_vit_b16_seed7=9/15, dinov2_vit_b14_seed7=9/15, efficientnet_b0_seed7=12/15, clip_vit_b16_seed42=9/15，无失败或访问阻碍，继续原固定队列。

2026-10-07T17:53:20.081323+00:00：工具上下文刷新后的实际SSH和原进程/锁核对成功；已完成epoch clip_vit_b16_seed7=9/15, dinov2_vit_b14_seed7=8/15, efficientnet_b0_seed7=11/15, clip_vit_b16_seed42=8/15，无失败或实际访问阻碍，首个完整模型及缓存仍等待中。

2026-10-07T14:21:48.908706+00:00：平台消息临时错误后实际SSH复核成功，supervisor3984143及原四worker身份、cwd/run.py和锁正常；已完成epoch clip_vit_b16_seed7=3/15, dinov2_vit_b14_seed7=3/15, efficientnet_b0_seed7=4/15, clip_vit_b16_seed42=3/15；日志正常，无实际访问阻碍，继续原队列。

2026-10-07T13:11:31.922229+00:00：连接通知后的实际SSH复核成功；原supervisor3984143/四worker身份与cwd/run.py一致，锁继续持有。已完成epoch clip_vit_b16_seed7=2/15, dinov2_vit_b14_seed7=1/15, efficientnet_b0_seed7=2/15, clip_vit_b16_seed42=2/15；无新的失败或访问阻碍，继续等待既定队列。

远端实际检查：2026-10-07T12:48:06.486649+00:00。当前等待原正式队列第2轮训练；phase=15_formal_runs_and_features，完整run为0/15。
监督进程3984143在运行，execution.lock与queue.lock均held；未重启或重复启动，无周期调度。

|物理GPU|作业|已完成epoch|最近state.pt UTC|
|---:|---|---:|---|
|4|clip_vit_b16_seed7|1/15|2026-10-07T12:28:58.098388+00:00|
|5|dinov2_vit_b14_seed7|1/15|2026-10-07T12:31:30.821453+00:00|
|6|efficientnet_b0_seed7|1/15|2026-10-07T12:20:40.770900+00:00|
|7|clip_vit_b16_seed42|1/15|2026-10-07T12:29:41.882106+00:00|

此次检查无失败run/stage，SSH与账本可读，无新的访问或审批阻碍。LLM原容器exited（137，OOMKilled=false），restart仍unless-stopped，等待既有GPU队列完成后恢复；GPU0–3未干预。
详细证据：reports/latest_actual_progress.json；四份首轮checkpoint完整状态核对见reports/first_epoch_checkpoint_receipts.json。

此前核对记录保留如下：

最新核对：2026-10-07T12:38:19.984856+00:00。监督进程3984143及首批四个worker继续运行，均完成第1轮并进入第2轮，完整15轮run仍为0/15。

|run|完整epoch|训练loss|本轮分钟|optimizer更新|
|---|---:|---:|---:|---:|
|clip_vit_b16_seed7|1|0.174416|40.38|1954|
|dinov2_vit_b14_seed7|1|0.525858|42.94|1954|
|efficientnet_b0_seed7|1|0.267736|32.16|1954|
|clip_vit_b16_seed42|1|0.173343|41.10|1954|

四份实际checkpoint均验证identity/history、有限模型及optimizer张量、完整Python/NumPy/CPU/CUDA RNG，首轮AMP无跳过更新；另存于artifacts/continuations/20261007_exclusive_resume/first_completed_epoch。训练loss不是测试性能。

LLM停用补充：原Docker记录exit_code=137、OOMKilled=false；docker stop --time60已完成，未断言优雅退出。容器原ID、restart policy不变；GPU队列结束后现有监督流程负责恢复和health核对。


此前核对记录保留如下：

最新进度：2026-10-07T12:29:50.852213+00:00：正式训练首个完整 epoch 已完成：EfficientNet-B0 / seed7，第1轮，训练 loss 0.2677361860332489，250000个槽位，耗时1929.6947157382965秒。已另存48.6MB实际 checkpoint；213组 optimizer 状态均完成1954次更新，AMP scale=128、growth_tracker=1954，本轮无跳过更新。Python/NumPy/CPU/CUDA RNG 齐全，模型及 optimizer 张量有限，identity 与 history 已核对。四个 worker 继续运行，完整正式 run 为0/15，无最终测试性能。代码、配置和冻结输入未变。

此前核对快照保留如下：

# 2026-10-07 当前执行回执

核对时间：2026-10-07T12:07:53.431147+00:00。监督进程3984143和首批四个worker仍运行，15次正式训练尚未完成。没有重复启动队列。资格测试及pilot属于准备验证，不作为科学实验结果。

## 服务停用

原onecat-vllm容器ID：0de7633bc4eb9b0d12b41dd2d05e3bb42961310e62bcb1b78f0f2ed0f54aad5c。
停用前快照时间为2026-10-07T11:47:18.617785+00:00；实际Docker退出时间为2026-10-07T11:48:19.050875106Z。两者不得混同。
执行正常命令：docker stop --time 60 onecat-vllm。当前同一容器exited，restart policy仍unless-stopped；gateway未改动，GPU0–3/Jev未干预。
GPU队列结束后由现有监督流程docker start同一容器并核对本地health HTTP200；目前尚未恢复，不在worker运行期间手工恢复。

## 首批运行映射

|物理GPU|骨干|seed|PID|
|---:|---|---:|---:|
|4|clip_vit_b16|7|3990784|
|5|dinov2_vit_b14|7|3990785|
|6|efficientnet_b0|7|3990786|
|7|clip_vit_b16|42|3990787|

统一15epoch/final-only，有效batch128、实际microbatch128、AdamW WD5e-4，学习率CLIP1e-5、DINO5e-5、Efficient1e-4。正式预算3骨干×5seed=15次完整训练与30个匹配头，种子[7,42,123,2026,3407]。另有已完成的独立256输入/1epoch CLIP资格pilot，不计入正式结果。

## 冻结输入口径

|角色|唯一输入行|real|fake|
|---|---:|---:|---:|
|train|250,000|5,752|244,248|
|calibration|1,120|1,120|0|
|ff_test|59,935|1,120|58,815|
|cdf_test|368,922|1,416|367,506|

每个epoch从train清单有放回抽取N_pool=250000次；这是整次run的epoch槽位数，不是每方法或每GPU的独立额外预算，也不保证每个唯一输入恰好被看一次。真实总质量1/2、31个fake方法各1/62；视频内先等权抽视频再抽固定8帧，EFS按独立RGB图抽样。共享RGB输入在方法/视频关联权重计算后合并质量；265527条训练关联对应250000条唯一内容输入。canonical代表method_id只有30种，而method_memberships与质量保留完整31种，未删除SG3方法。

校准140个原FF-val真实视频、1120帧，fake-cal0，所以TPR95不可用。隔离保证限定已知FF来源和精确RGB内容，未知EFS生成祖先/音频驱动保持未知。CDF测试178个声明real视频中1个不足8个不同RGB帧，实际保留177个/1416帧；这不是解码失败，原审计解码失败0。所有原角色、逐条处理与失败证据留存，不把EFS存储桶解释为自然视频。

## 数值实现与科学范围

严格确定性下实测CUDA不支持DINO bicubic-antialias插值反向：相同FP32插值迁至CPU，原37×37位置参数及可微GPU复制保留。CPU参考值和原网格梯度回归、实际CUDA训练资格通过；CPU/CUDA舍入及速度差异仍需如实披露。AMP float16 initial_scale128，TF32 off，各骨干micro128。此次不是更改结构、epoch、学习率、种子或引入目标域调参。

官方修订62份JSON与本机输入索引逐字节/语义一致；本机跨角色像素证据与独立真实文件核对留存，原图片压缩包出处未认证，不将本机像素重复归咎官方。原Obsidian v1.1协议未改写。

代码SHA256：7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5。
冻结标记SHA256：c3044b9219a8ef18dd1c25f7f75de0d38dd93642bbe4cc0e19381218bf820b5d。
正式队列配置SHA256：bbc329454aad0a1974db6384984a173a32904a93d02c4632d784f78b79613f9f。

## 回执与实时证据

reports/llm_stop_receipt.json、reports/execution_configuration_receipt.json保存机器可读回执。
artifacts/execution_status.json、artifacts/queue_status.json是实时状态；artifacts/runs/<run_id>/history.json记录完整epoch后才更新。
报告和分析必须等待15份最终权重/特征、30头及完整A–D/五seed校验；当前无正式最终性能结果。
