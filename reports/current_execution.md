
## 2026-10-09T01:04:42.435201+00:00 — bootstrap声明分组支持的独立检查

仅读取已冻结清单的group_id/标签/方法，不使用分数、不拟合/选择任何模型或阈值，按既定seed和2000次共同分组重采样独立核验方法支持。RDDM在FF图像池仅4个声明分组、CDF图像池仅2个声明分组；seed[7,42,123,2026,3407]的FF方法宏平均无效重复分别[39,39,39,34,37]，CDF分别[274,264,247,288,284]。所有视频21方法的2000次共同抽样均有完整支持。原始分组/未知祖先的独立性没有因此认证。

冻结规则已规定缺任一方法就使整条macro replicate无效，不能忽略该方法后缩小平均；无效次数与有效重复上的条件区间需原样披露。详细输入hash、分组数量与每seed支持在reports/bootstrap_support_structure.json，终态验收将与保存的bootstrap数组/摘要逐项比对。未改变数据/分组/次数/科学代码或队列。当前12/15完整run已核验，最后seed3407三个原worker继续，正式结果尚未生成。原设计SHA不变。


## 2026-10-09T01:01:06.712170+00:00 — 前四seed×三骨干12个完整run全部独立核验

DINOv2 B14 seed2026的train250000、FFtest59935、CDFtest368922、calibration1120，共679977条768维FP32 raw cache通过独立CPU核验：块hash/有限值、冻结清单全列逻辑值与顺序、final epoch15分类头、canonical verified_run均通过。校准parquet逻辑一致，未声称序列化字节一致。原PID729256在2026-10-09 00:47:56.534755 UTC exit0，已退出且GPU7无后继本轮worker。

至此seed[7,42,123,2026]×3backbone的12个训练终态与12组完整四角色缓存均独立核验。仅seed3407三模型原worker继续，原supervisor PID3984143与执行/队列锁保持；原LLM按授权持续停止，待全部GPU任务后由原supervisor恢复并核验健康，再做CPU正式分析。未新建队列/observer/计划，未调整模型或加速设置；科学源代码/配置/冻结数据及原Obsidian设计SHA不变。15模型/30匹配头/A–D汇总/服务终态尚未全部完成，当前不作为实验最终交付。


## 2026-10-08T22:57:55.085372+00:00 — 四个seed全部12个训练终态通过，完整run11个

DINOv2 B14 seed2026完成15epoch/final并独立CPU核验：3750000 draws、29310 attempts、29303 successful optimizer updates、7 AMP skips、final scale16384；完整model/optimizer有限、full-state与final一致、Python/NumPy/CPU/CUDA RNG齐全、15份epoch抽样序列精确重建。逐step AMP日志缺口照实披露，冻结draw预算不补步、不回放。至此seed[7,42,123,2026]×3backbone全部12个训练终态通过。

EfficientNet-B0 seed2026四角色缓存独立核验：679977条1280维FP32原始特征，块hash/有限值、冻结清单全列值与顺序、final分类头与canonical verified_run通过。原PID772379在2026-10-08 22:45:28.078063 UTC exit0；原supervisor PID3984143在GPU5自动接续EfficientNet seed3407 PID1105684，PPID/cwd/script/锁正确。累计12/15训练终态、11/15完整run；DINO2026继续抽取特征，三个seed3407模型已由原队列启动，未重复提交或变更科学设置。30匹配头、完整正式分析及服务实际恢复仍待完成，尚未交付科学结果。原Obsidian设计SHA不变。


## 2026-10-08T22:32:40.215743+00:00 — 第10个完整正式run独立核验

CLIP ViT-B16 seed2026 完整训练及4-role raw cache通过独立CPU核验：train250000、FFtest59935、CDFtest368922、calibration1120，共679977条512维FP32原始特征。每块hash/维度/有限值、冻结输入全列值与顺序、final epoch15分类头、canonical verified_run均通过。校准清单是逻辑全值/顺序一致，不声称parquet序列化字节一致。

原worker在2026-10-08 22:20:55.576895 UTC exit0且PID657113已退出；原supervisor PID3984143自动在GPU6接续DINOv2 seed3407 PID1091670，PPID/cwd/script/锁验证正确。累计11/15训练终态、10/15完整run与缓存均独立核验；其余原队列继续，无重复作业，无科学设置变更。30匹配头、完整正式分析、最后服务实际恢复尚未完成。原Obsidian设计SHA不变。


## 2026-10-08T21:23:42.851337+00:00 — 第11个正式训练终态独立核验

EfficientNet-B0 seed2026 完成15epoch/final：3750000 draws、29310 attempts、29306 successful optimizer updates、4 AMP skips、final AMP scale 16384。完整model/optimizer有限、final与full-state一致，Python/NumPy/CPU/CUDA RNG齐全；15份epoch抽样序列逐一精确重建，49个BN counter相对初始化均前进29310。冻结预算是draw/epoch，逐step AMP事件日志缺口继续明确披露，不补步、不回放、不改科学设置。

当前独立核验训练终态11/15；原队列完整训练及4-role缓存9/15，已独立核验9/15。原supervisor PID3984143持续执行；GPU0–3/Jev与gateway未干预。模型特征抽取、剩余训练、30匹配头/正式分析/服务实际恢复仍待完成，尚未交付科学结果。原设计SHA不变。

## 2026-10-08T20:27:31.410738+00:00：第10个最终训练状态独立核验

- CLIP ViT-B16 seed2026最终epoch15，3750000 draws / 29310 attempts / 29305 successful optimizer updates / 5 AMP skips，final AMP scale16384。15轮抽样数组SHA与history相符，并按冻结sampling_mass/seed精确重建；完整model/optimizer浮点张量有限、final.pt与完整state模型张量精确一致，Python/NumPy/CPU/CUDA RNG完整。CPU验证CUDA_initialized=false。
- 固定15轮/每轮N_pool抽样合同满足，无补步/回放/超参数变化；逐step AMP跳步及scale事件日志缺口持续明确披露，不能从既有记录恢复具体跳步位置。
- 累计最终训练状态完成并独立核验10/15组，完整四角色模型/cache run独立核验9/15组。原CLIP2026 worker657113继续提取cache，EfficientNet2026/DINO2026/CLIP3407原worker继续训练；20:25 UTC只读SSH监督成功，无访问或已记录执行/队列错误。
- 回执completed_model_state_receipts.json、formal_training_budget_receipts.json、formal_training_budget_counts.csv已更新；剩余5组训练/6组完整cache、全部30匹配头/完整分析报告及LLM实际恢复/HTTP健康确认尚未完成。source/config/frozen data/科学预算及用户模型/加速设置未改变，源码SHA保持7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5。

## 2026-10-08T19:00:18.871330+00:00：前三个种子的9个完整run独立核验

- EfficientNet-B0 seed123完整四角色cache共679977行、1280维FP32；train250000/ff_test59935/cdf_test368922/calibration1120。全部分块SHA、有限值/非零向量、连续行及完整清单列值/dtype/顺序对齐、两类head与final.pt精确一致均通过，canonical verified_run=true。校准parquet为逻辑等价，不声称与输入byte相同。
- 原worker622111于2026-10-08T18:41:16.625344+00:00 COMPLETE/returncode0，已退出；原supervisor3984143于2026-10-08T18:41:16.626964+00:00自动轮转GPU4至CLIP3407/PID960076。新worker PPID/cwd/run.py/worker阶段匹配，execution.lock/queue.lock均持有；CPU验证CUDA_initialized=false，未启动重复队列。
- seed7/42/123各三个骨干共9个最终训练状态和9个完整模型/cache run全部独立核验。余下seed2026/3407共6组继续原队列，当前CLIP3407(GPU4)、EfficientNet2026(GPU5)、CLIP2026(GPU6)、DINO2026(GPU7)运行，DINO3407/EfficientNet3407待调度。无访问或已记录执行/队列错误。
- 最终30匹配头、全模型尺度/长度控制/配对bootstrap/源FPR5冻结阈值报告及原LLM实际恢复/HTTP健康确认未完成，不声称实验完成。completed_run_cache_receipts.json已扩展至9组；源码SHA7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5及原设计SHA61e583963a4d661ccf83d02798600c83eb4aca2238d2984c5422bab1553ecc6b保持不变，source/config/frozen data/科学预算/用户模型与加速设置未改变。

## 2026-10-08T17:18:57.399214+00:00：第9个最终训练状态独立核验

- EfficientNet-B0 seed123最终epoch15：3750000 draws / 29310 attempts / 29305 successful optimizer updates / 5 AMP skips；final AMP scale32768。15轮抽样数组SHA与history对应并按冻结sampling_mass/seed逐轮精确重建；model/optimizer浮点张量有限、final.pt与完整state模型张量精确一致，Python/NumPy/CPU/CUDA RNG完整。49个BatchNorm层相对同一初始化forward计数均增加29310。CPU验证CUDA_initialized=false。
- 固定15轮/每轮N_pool抽样预算满足；不补步/回放/改超参数。逐step AMP跳步位置/scale事件日志缺口持续明确披露，无法从既有记录恢复具体位置，不声称完整逐step审计。
- 累计最终训练状态完成并独立核验9/15组，完整四角色模型/cache run完成并独立核验8/15组。原EfficientNet123 worker622111继续提取cache，CLIP2026/DINO2026/EfficientNet2026继续原队列训练。17:16 UTC只读SSH检查成功，无访问或已记录执行/队列错误。
- 回执completed_model_state_receipts.json、formal_training_budget_receipts.json、formal_training_budget_counts.csv更新；剩余6组训练/7组完整cache及全部30匹配头、完整分析报告、LLM实际恢复/HTTP健康确认未完成。source/config/frozen data/科学预算和用户模型/加速设置未改变；源码SHA仍7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5。

## 2026-10-08T13:27:26.264409+00:00：第8个完整训练/cache run独立核验

- DINOv2-B14 seed123完整四角色cache共679977行、768维FP32；train250000/ff_test59935/cdf_test368922/calibration1120。全部分块SHA、有限值/非零向量、连续行与完整输入清单列值/dtype/顺序对齐、两类head与final.pt精确一致均通过，canonical verified_run=true。校准parquet逻辑等价，不声称与输入byte相同。
- 原DINO123 worker273524于2026-10-08T13:18:40.008107+00:00 COMPLETE/returncode0，已退出；原supervisor3984143于2026-10-08T13:18:40.009325+00:00自动轮转GPU5至EfficientNet2026/PID772379。新worker PPID/cwd/run.py/worker阶段匹配；execution.lock/queue.lock均持有，CPU核验CUDA_initialized=false，无重复队列。
- 累计最终训练状态及完整四角色cache run均完成并独立核验8/15组。当前原队列继续训练EfficientNet123(GPU4/PID622111)、EfficientNet2026(GPU5/PID772379)、CLIP2026(GPU6/PID657113)、DINO2026(GPU7/PID729256)；余下三个seed3407 run待原队列调度。
- 尚未完成：剩余7组训练/cache、全部30匹配头、完整尺度/长度控制/配对bootstrap/源FPR5冻结阈值分析与报告、原LLM实际恢复及HTTP健康确认。不是实验完成；无访问或已记录执行/队列错误。
- completed_run_cache_receipts.json扩展至8组；原设计SHA61e583963a4d661ccf83d02798600c83eb4aca2238d2984c5422bab1553ecc6b及训练源码SHA7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5保持不变。source/config/frozen data/科学预算及用户模型/加速设置未调整。

## 2026-10-08T12:14:13.476678+00:00：第7个完整模型/cache run独立核验

- CLIP ViT-B16 seed123完整四角色cache共679977行、512维FP32；train250000/ff_test59935/cdf_test368922/calibration1120。全部分块SHA256、有限值/非零向量、连续行对齐、完整清单列值/dtype/顺序与冻结输入一致、两类head与final.pt精确一致均通过，canonical verified_run=true。校准parquet序列化byte不同但完整逻辑值一致，不声称byte相同。
- 原CLIP123 worker255008于2026-10-08T12:06:32.912518+00:00 COMPLETE/returncode0，已退出；原supervisor3984143于2026-10-08T12:06:32.914021+00:00自动轮转GPU7至DINO2026/PID729256。新worker PPID/cwd/run.py/worker阶段匹配，execution.lock和queue.lock均持有；CPU验证CUDA_initialized=false；无重复队列。
- 最终训练状态完成并独立核验8/15组，完整四角色模型/cache run完成并独立核验7/15组。DINO123继续提取缓存；EfficientNet123、CLIP2026、DINO2026继续原队列训练。剩余训练/cache、全部30匹配头/全模型分析报告、LLM实际恢复及HTTP健康确认待完成。
- 回执completed_run_cache_receipts.json扩展至7组，原设计SHA保持61e583963a4d661ccf83d02798600c83eb4aca2238d2984c5422bab1553ecc6b，训练源码SHA保持7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5；source/config/frozen data/科学预算及用户模型/加速设置未改。

## 2026-10-08T11:24:57.430273+00:00：第8个最终训练状态独立核验

- DINOv2-B14 seed123最终epoch15，3750000 draws / 29310 attempts / 29307 successful optimizer updates / 3 AMP skips；final AMP scale131072。15轮抽样文件hash与history对应，按相同冻结sampling_mass和seed逐轮精确重建；完整model/optimizer浮点张量有限、final.pt与完整state模型张量精确一致、Python/NumPy/CPU/CUDA RNG完整，CPU验证CUDA_initialized=false。
- 固定15轮/每轮N_pool抽样合同满足，未补步/回放/改超参数；逐step AMP跳步与scale事件未记录的缺口持续明确披露。累计最终训练状态完成并独立核验8/15组，完整四角色模型/cache run独立核验6/15组。
- 原supervisor3984143及锁正常；DINO123原worker273524继续提取缓存，CLIP123缓存继续推进，EfficientNet123与CLIP2026继续原队列训练。11:23 UTC只读SSH核验成功，无访问或已记录执行/队列错误；完整报告、30匹配头及LLM恢复/健康确认仍待完成。
- 回执completed_model_state_receipts.json、formal_training_budget_receipts.json、formal_training_budget_counts.csv更新；训练代码SHA仍为7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5。未改source/config/frozen data/科学预算或用户模型/加速设置，未启动重复作业或调度器。

## 2026-10-08T10:21:56.926042+00:00：第7个最终训练状态与第6个完整run独立核验

- CLIP ViT-B16 seed123：final epoch15，3750000 draws / 29310 attempts / 29305 successful optimizer updates / 5 AMP skips，final AMP scale16384。完整模型/优化器浮点状态有限，final.pt与完整state模型张量精确一致，Python/NumPy/CPU/CUDA RNG保留；15轮抽样数组hash与history一致并精确重建。固定抽样预算满足，不补步或回放；逐step AMP跳步位置及scale事件日志缺口继续披露。
- DINOv2-B14 seed42：train250000/ff_test59935/cdf_test368922/calibration1120，共679977行768维FP32完整cache；分块SHA、有限值/非零向量、完整清单值/dtype/顺序对齐、两类head与最终权重精确一致均通过，canonical verified_run=true。校准parquet为逻辑等价而非与输入byte相同。
- 原DINO42 worker144861于2026-10-08T10:10:39.780178+00:00 COMPLETE/returncode0，已退出；原supervisor3984143于2026-10-08T10:10:39.781614+00:00自动轮转GPU6至CLIP2026/PID657113。新worker PPID/cwd/script/stage匹配；execution/queue锁均持有。CPU核验CUDA_initialized=false，未启动重复队列。
- 15组最终训练状态完成并独立核验7组，四角色完整模型/cache run完成并独立核验6组。CLIP123提取缓存；DINO123、EfficientNet123、CLIP2026按既定队列继续训练；全部30匹配头、全模型分析报告及LLM实际恢复/HTTP健康确认仍待完成，不声称实验结束。
- 回执completed_model_state_receipts.json、formal_training_budget_receipts.json、formal_training_budget_counts.csv、completed_run_cache_receipts.json均已扩展。训练源码SHA仍为7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5；source/config/frozen data/科学预算及用户模型/加速设置未调整。

## 2026-10-08T09:33:06.950812+00:00：第5个完整模型/cache run独立核验

- EfficientNet-B0 seed42四角色cache共679977行、1280维；train250000/ff_test59935/cdf_test368922/calibration1120。分块SHA256、FP32/有限值/非零向量、连续行对齐、完整输入清单列值与顺序、两类head与final.pt精确一致全部通过；canonical verified_run=true。校准parquet序列化SHA与输入不同，完整逻辑值/顺序/dtype相同，不声称两文件byte相同。
- 原worker245444于2026-10-08T09:17:31.966558+00:00 COMPLETE，returncode0，已退出；原supervisor3984143于2026-10-08T09:17:31.968018+00:00自动轮转GPU4至EfficientNet123/PID622111。新worker PPID/cwd/run.py/worker阶段均核验匹配，execution.lock与queue.lock均持有；未启动第二个队列。
- 15组中训练完成并独立核验6组，四角色完整run完成并独立核验5组。DINO42仍缓存提取，CLIP123/DINO123仍训练，其余待队列推进；最终30匹配头/全部分析报告/LLM恢复与健康检查仍未完成。
- CPU独立核验 CUDA_initialized=false；训练源码SHA保持7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5。回执reports/completed_run_cache_receipts.json已扩展至5组；未调整source/config/data/科学预算/模型或加速设置。

## 2026-10-08T08:08:24.139089+00:00：第二批两个最终检查点独立核验

- 原 supervisor PID 3984143、execution/queue 两把锁和四个原 worker 身份核验正常。15 组中训练完成并独立核验 6 组，完整四角色缓存 run 完成并独立核验 4 组；不是最终实验完成。
- DINOv2-B14 seed42：final epoch15；3750000 draws / 29310 attempts / 29305 successful optimizer updates / 5 AMP skips；final AMP scale32768。
- EfficientNet-B0 seed42：final epoch15；3750000 draws / 29310 attempts / 29304 successful optimizer updates / 6 AMP skips；final AMP scale16384；49 个 BatchNorm 层 forward counter 相对相同初始化均增加29310。
- 两个完整 state 的模型张量均与 final.pt 精确相同，所有模型/优化器浮点张量有限，Python/NumPy/CPU/CUDA RNG 均保留；各自15轮抽样数组 hash 对应 history，并精确重建。
- 仅 CPU 独立核验，CUDA_initialized=false。固定15轮、每轮N_pool抽样预算满足；逐step AMP overflow/scaler事件未记录，具体跳步位置及逐次scale数值仍不可从既有记录恢复，回执逐条保留null及日志缺口说明。不补步、不回放、不改超参数。
- DINO42和EfficientNet42原worker继续提取缓存；CLIP123/DINO123仍训练。既有四个完整cache各679977行已独立核验；新的完整cache和最终分析均待完成。
- 回执：reports/completed_model_state_receipts.json、formal_training_budget_receipts.json、formal_training_budget_counts.csv；本地只读监督回执 latest_readonly_progress_receipt.json UTC08:05:52；SSD当时剩689266290688 bytes。未启动重复作业/调度器，未调整模型或加速配置。


2026-10-08T00:41:23.953929+00:00：首批四个完整15epoch+四角色rawcache run全部独立核验。最后DINOv2 seed7为679977行/768维，完整块hash/FP32有限/非零向量/冻结输入顺序/最终head一致，canonical verified_run=true；原worker3990785于2026-10-08T00:31:06.016812+00:00退出0，原supervisor3984143在GPU5自动启动DINO seed123 PID273524，两锁/PPID/cwd/run.py正常。当前完整run4/15；GPU4 Efficient42 PID245444、GPU5 DINO123 PID273524、GPU6 DINO42 PID144861、GPU7 CLIP123 PID255008继续原队列。所有源设置/代码/预算未变，未新建队列或调度；15run全部缓存、30头分析及原LLM恢复与health核对仍待后续。原第一完成模型EfficientNet/seed7的AMP4次skip实际聚合/不可追溯逐事件缺口、无scheduler与已消费数据不回放说明均已保留，未补步。


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
