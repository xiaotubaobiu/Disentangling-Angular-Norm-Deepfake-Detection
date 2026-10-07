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
