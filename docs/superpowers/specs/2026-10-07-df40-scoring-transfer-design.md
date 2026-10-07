# DF40 计分机制与跨域阈值稳定性：实施设计

日期：2026-10-07（Asia/Seoul）。阶段：用户已确认会话中的设计方向；本书面设计等待审阅。本文不是实验结果，也不是已经执行的实施计划。

## 1. 意图、范围与依据

在新项目 `/home/z/project/deepfake_detector` 中执行 Obsidian v1.1：解释特征长度处理、帧级非线性、视频聚合的交互作用，评价独立源校准后的跨域冻结阈值，以及五种子的稳定性。用户明确两个 GitHub 项目只作代码参考，不额外增加它们原方法的对照训练。

科学协议以同目录的 [v1.1 原稿快照](2026-10-07-df40-v1.1-source.md) 为准。原文件为 `/home/z/obsidian/paper/deepfake-detector/实验重设计_3backbone_5seed_v1.1.md`，SHA256 为 `61e583963a4d661ccf83d02798600c83eb4aca2238d2984c5422bab1553ecc6b`。快照逐字节保存，不改写 Obsidian 原稿。本文落实接口、审计、运行和故障处理，不扩大研究范围；若本文与科学协议冲突，先停止实施并解决冲突，不静默修改协议。

| 项目 | 固定约束 |
| --- | --- |
| 方法集合 | 31：9 FS + 12 FR + 10 EFS；e4s 保留，e4e 与全部 FE 排除 |
| 编码器 | CLIP ViT-B/16、DINOv2 ViT-B/14、EfficientNet-B0 |
| 训练种子 | `[7, 42, 123, 2026, 3407]` |
| 端到端训练 | 15 个 raw 模型，31 方法联合训练；每个模型 15 epoch，使用最终轮权重 |
| 匹配头 | 每个冻结编码器拟合 raw/direction 两个线性头，共 30 个 |
| 视频评价 | 21 FS/FR；一个域中一个方法的一个视频为一个单位，每视频固定 8 张不同有效帧 |
| 图像评价 | 全部 31 方法；FS/FR 所选帧与 EFS 独立图片分开注明单位 |
| 校准 | FF++ 官方 validation；主阈值 FPR5，TPR95 仅在完整独立伪造校准覆盖下执行 |
| 目标域 | CDF 只作评价，不训练、选模、拟合变换、选阈值或选流程 |
| GPU | 仅物理 4–7；0–3 的其他实验继续运行 |

固定本地方法 ID：

| 类别 | 本地方法 ID |
| --- | --- |
| FS | `fsgan, faceswap, simswap, inswap, blendface, uniface, mobileswap, e4s, facedancer` |
| FR | `fomm, facevid2vid, wav2lip, MRAA, one_shot_free, pirender, tpsm, lia, danet, sadtalker, mcnet, hyperreenact` |
| EFS | `VQGAN, StyleGAN2, StyleGAN3, StyleGANXL, sd2.1, ddim, rddm, pixart, DiT, SiT` |

规范名与本地 ID 分列；`ddim` 是本地条目名，论文方法表的 DDPM 命名必须留有来源说明，不能无证据声称 DDPM 与 DDIM 实现完全相同。`RDDM` 文件夹与 `rddm` 索引的大小写差异显式映射。

## 2. 上游参考与已复现问题

| 上游 | 固定参考提交 | 使用边界 |
| --- | --- | --- |
| [DF40](https://github.com/YZY-stack/DF40) | `ce808160e0df31544b41c7b60b942849ba88bca8` | 方法登记、索引结构、数据和公开初始化参考 |
| [Angular–Norm](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection) | `5cc0105477509566295ea6cd3c277cdcf4ede82e` | 编码器、特征提取、线性头与固定增强策略参考 |

上游源码与许可证保留来源记录。可复用的模型封装在新包内适配；上游 checkout 若需要保留，存于独立 `references/`，不以修改上游默认训练脚本来定义本轮协议。

只读探针已复现 Angular–Norm 的两处错误：

1. `evaluation/sample_eval.py:77–89` 仅按 `video_id` 平均。同 ID 的两个方法分数 0.9/0.2 被合成 0.55；两个 fake 视频与一个 real 输入，返回两个而非三个评价单位。其样本行还缺少方法字段。
2. `data/builders.py:46–57` 仅按 `pair_id` 抽帧。两个方法各 32 帧、要求每视频 8 帧时，共取 8 帧（每方法 4 帧），而非 16 帧。

上游的 16 方法列表、586/100 验证切片、EMA、warmup、cosine、early stopping、最佳 validation 检查点、旧种子和 8 卡 launcher 不适用于 v1.1。DF40 上游的父目录名聚合、评价异常时回退帧 AUC、抽帧和 test 选模逻辑也不继承。新管线的关键约束须由测试锁住。

## 3. 数据层：官方角色恢复与审计

数据根为 `/Dataset/deepfake_detection/DF40_all`，只读使用；不移动、重命名或改写原数据和原 JSON。数据衍生物写入新项目 `artifacts/`，代码/配置/设计纳入 Git，特征、权重、逐帧分数与图像不纳入 Git。

### 3.1 角色依据

官方 FF++ split 来源为：

- `https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json`
- `https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json`
- `https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json`

实施时保存官方文件、获取版本和内容哈希；它们分别包含 720、140、140 个视频 ID。目录与索引的原角色保留在 `original_role`，恢复后的角色另存为 `resolved_role`。本地真实帧目录现有 999 个视频组：train 719（缺 `281`）、val 140、test 140。真实 validation 有 4,477 张图片，全部 140 个组的帧数至少为 8；这只是文件及数量核查，尚未完成解码和内容审计。

以下索引角色不能直接用于训练：

| 原索引条目 | 按可解析官方来源 ID 得到的组成 |
| --- | --- |
| SimSwap fake train 988 组 | train 713、val 135、test 140 |
| InSwapper fake train 883 组 | train 664、val 101、test 118 |
| UniFace fake train 372 组 | train 318、val 54 |
| MobileSwap fake test 858 组 | train 718、test 140 |

所有可解析数字双 ID 组在只读核查中均匹配官方配对，两个 ID 属同一角色。按官方来源恢复角色属于修复本地映射，不是随机重划；记录每项角色更正及其依据，官方 val/test 来源不能留在训练池。不同位置的相同语义视频不是额外样本；选择本地可用副本并保留替代路径与内容哈希，内容不一致时报告冲突并暂停受影响数据的使用。

对目标/来源/驱动信息，只使用可证实的关系：Wav2Lip、SadTalker 的测试驱动后缀也出现在训练路径中，但字符串重复尚不能证明相同驱动内容。实施审计须核对可得来源映射；若确认同一驱动来源跨角色，则按 v1.1 隔离受影响样本并记录，不改动官方划分。若无法解除其对主比较的影响，暂停受影响方法和完整 31 方法正式运行，不悄悄缩小主集合。SadTalker 的 8 个目标 ID 不可解析训练条目先隔离，不猜测 ID。

### 3.2 EFS 的单位和路径

EFS 分别登记生成域、生成器、可得生成批次和存储组；已知的存储分组不自动升级为真实来源视频关系。其独立图像不作为自然视频平均，不能用于 same-frame 位移。若需要根据官方 ID 修复 EFS 的存储角色，须记录官方索引/组织映射依据，不能仅凭类似视频的文件夹名推断生成因果来源。

9 个 EFS 的 CDF 索引路径含 `Celeb-real` / `YouTube-real`，实际目录名为 `Fake_from_Celeb-real` / `Fake_from_Youtube-real`；通过显式逐方法映射处理，不能简单全局替换，也不能将映射未处理报告成图像缺失。发现的 EFS validation 存储组不证明视频配对或完整伪造校准覆盖。

### 3.3 清单和隔离规则

每条输入记录至少包含：`sample_id, domain, original_role, resolved_role, label, method_id, method_type, evaluation_unit, video_id, source_ids, driving_source_id, storage_group_id, generator_id, generation_batch_id, image_path, original_image_path, frame_number, content_sha256, include, exclusion_reason`。未知来源字段显式为空，不虚构配对。仅批次、驱动或来源 provenance 未知，不自动排除已有可靠官方角色的输入；无法确定官方角色、存在已证实跨角色冲突或不满足输入有效性规则的条目才隔离。未知元信息带来的解释边界写入审计报告。清单版本及哈希一旦冻结，15 个模型及所有流程共用。

- 真实图片跨方法去重，真实视频的规范方法为 `real`。
- 对路径与内容重复进行核查。跨角色相同内容或已知来源冲突必须解释/隔离，不能依赖模型分数筛除。
- FS/FR 在数值帧序中确定性均匀选择 8 张不同有效帧。不足 8 帧的组不进入主视频表，不补重复帧。
- 主训练与图像/帧评价使用同一冻结 8 帧池；不足 8 帧的 FS/FR 组不进入本轮主输入池，保留完整失败统计，不另行引入可变帧数补充主比较。
- EFS 使用清单中的独立有效图片，按内容处理重复并保留存储组织元信息。
- 图像必须可读取/解码；所有排除与计数在训练前完成并对所有 backbone/seed 一致。
- 仅当 31 方法的三角色覆盖和主池审计通过时启动正式训练；缺少方法须报告阻断，不能以 30 或更少方法冒充原计划。

FF++ real validation 由官方 140 ID 恢复。FPR5 仅需独立真实校准数据；当前独立 FS/FR fake validation 不能覆盖全部 21 方法，因此 TPR95 标记 `unavailable_missing_official_fake_validation_coverage`。只有在正式运行前找到完整、独立并通过审计的官方覆盖，才允许按原协议启用相应评价层的 TPR95；不得使用 FF++ test、CDF 或临时 train 切片补齐。

## 4. 新项目架构与接口

使用独立 Python 包 `src/deepfake_detector/`，避免旧平铺包 `data/models/engine` 的跨项目导入冲突。CLI 以项目根和配置文件为锚点，不依赖调用目录。模块按以下边界划分：

| 模块 | 职责 | 输入 → 输出 |
| --- | --- | --- |
| `data/registry` | 固定方法与路径别名 | 注册配置 → 方法登记表 |
| `data/manifest` | 官方角色恢复、去重、审计、8 帧固定池 | 原 JSON / 官方 ID / 文件 → 冻结清单与审计报告 |
| `data/sampling` | 类、方法、视频/图片的分层抽样 | 训练清单与 RNG → 样本槽位序列 |
| `models` | 三编码器及单线性头 | 解析配置/初始化权重 → raw 特征与分类 logit |
| `training` | 固定训练、状态保存和恢复 | 配置/清单 → epoch 15 权重和完整日志 |
| `features` | 分块特征与元信息缓存 | 最终权重/冻结输入 → h、r、头参数和清单对应关系 |
| `scoring` | 四流程、尺度和常长度干预 | 同一缓存 → 视频/图像分数 |
| `calibration` | 独立源校准、冻结阈值、预测 | 源校准分数 → 阈值；阈值+测试分数 → 预测 |
| `analysis` | AUC、宏指标、pair 转移、匹配头、重采样 | 冻结分数/预测/来源组 → 分析表 |
| `reporting` | 汇总每 seed 与逐方法证据 | 表/日志 → 主表、图、报告 |
| `orchestration` | 四卡队列、断点、服务恢复 | 运行清单/GPU 状态 → 作业账本和恢复记录 |

视频身份键为 `(domain, method_id, video_id)`；结果表再包含 `checkpoint_id, seed, backbone, flow, role`。所有聚合、抽帧、join、bootstrap 与缓存均继承这些边界，不仅修一个显示层函数。一个视频内标签必须一致。真实视频键在 `method_id=real` 下建立；重复使用同一真实池做逐方法比较不增加独立真实样本数。

EFS 图片有独立 `sample_id`；图像层仅有 raw/direction 两个实质不同的排序条件。其存储组只在声明的聚类假设下用于重采样，不输出伪造视频 AUC。

配置解析、方法登记、清单、初始化和运行状态各有版本与哈希。恢复时比对配置、清单和初始化来源，任一变化均拒绝作为原 run 续训，须创建新 run。失败/未完成单元不能以旧结果填补。

## 5. 模型、训练与随机性

模型提取分类头前的未额外 L2 归一化特征。CLIP 避免直接使用已经单位化的便利接口；DINOv2/ EfficientNet 原有内部归一化层保留。每个 raw 模型使用单线性二分类头和交叉熵。

| 设置 | 固定值 |
| --- | --- |
| 分辨率 | 224 × 224 |
| 优化器 | AdamW，weight decay=5e-4 |
| 学习率 | CLIP 1e-5；DINOv2 5e-5；EfficientNet 1e-4 |
| 有效 batch | 128（不能解释为每 GPU 128 的多卡总 batch） |
| 训练 | 15 epoch，无 EMA、early stopping、warmup 或学习率调度 |
| 检查点 | 最终 epoch；不按 calibration/test/CDF 指标选取 |

固定增强顺序复用参考提交：HorizontalFlip(p=0.5)；Rotate(±10°, p=0.5)；GaussianBlur(kernel 3–7, p=0.5)；OneOf(RandomBrightnessContrast(limit 0.1), FancyPCA, HueSaturationValue, 外层 p=0.5)；ImageCompression(quality 40–100, p=0.5)；Normalize。实现前解析所用库版本的内部默认概率/范围并完整落入配置，语义保持参考策略，不引入新增强。评价仅作尺寸处理与标准化。CLIP 使用其公开 mean/std，其余使用 ImageNet mean/std，Normalize 后不重复除以 255。

抽样概率 real/fake 各 1/2；fake 中 31 方法等权；FS/FR 方法内等权视频后抽固定池帧，EFS 方法内等权独立图；real 同样按视频后帧抽样。每 epoch 的槽位总数固定为 `N_pool`：唯一真实和 FS/FR 主池图像数，加唯一 EFS 图像数。最后不足 128 槽位的小批也执行，按实际槽位归一化；不额外补齐或丢弃 epoch 尾部。抽样允许有放回，实际序列和运行随机状态可复核。

优先一张物理卡承载一个 run，最多四个独立 run 并行。正式运行前用无目标指标的内存探针固定每骨干 microbatch 和梯度累积设置，满足有效 batch 128；同骨干五 seed 使用相同设置，记录 EfficientNet BatchNorm 的实际 microbatch。若 128 microbatch 可行就不人为降低。AMP/TF32/cuDNN/确定性和 DataLoader worker RNG 设置在环境预检时冻结，CLIP/DINO/ EfficientNet 不为某个 seed 单独改变数值策略。

保存 Python/NumPy/PyTorch CPU/CUDA RNG、优化器、AMP scaler、epoch、抽样进度和配置哈希，以支持同配置恢复；测试确认恢复语义，未支持精确 batch 中点恢复时从明确标记的 epoch 边界恢复，不能声称逐位中途复现。

## 6. 计分、机制与匹配头

每个最终编码器只为相应冻结输入提取一次 raw 特征，按 FP32 分块缓存，统计在 FP64 中执行。二分类头采用 signed 参数 `w=W_fake−W_real, b=b_fake−b_real`。

令 `r=||h||₂`、`u=h/max(r,1e-12)`，同一头产生 `z_R=wᵀh+b`、`z_D=wᵀu+b`：

| 视频流程 | 分数 |
| --- | --- |
| RL | mean(z_R) |
| RP | mean(sigmoid(z_R)) |
| DL | mean(z_D) |
| DP | mean(sigmoid(z_D)) |

四流程共用相同 8 帧和最终模型，禁止将概率平均替换成 sigmoid(mean-logit)。缓存对齐需校验帧/视频键、标签、有限值和覆盖；不能 inner-join 后默默丢弃未对齐样本。

实验 A–D 的主指标和公式继承 v1.1：逐方法/宏平均与 pooled AUC 分开；视频/图像分表；固定尺度网格 `[0.25,0.5,1,2,4]`；源训练平衡权重下估计长度中位数 c，比较 h、h/c、u、c·u；b 不被错误地同步缩放；正仿射映射同步变换阈值的负对照；精确饱和、数值 ties 与非线性池化分开解释。

30 个匹配头均在各自编码器完全冻结、eval 模式的官方训练特征上拟合。raw/direction 头均零初始化，L2=1e-4、不惩罚偏置、迭代上限 2000，记录收敛与固定权重；不更新骨干运行统计，不使用目标标签选正则或挑头。

pair 转移采用正确/并列/错误的 3×3 表，ties 按半分；重建的 DeltaAUC 与直接值在 1e-10 内一致。实现采用可核对的加权/分块算法，避免对大图像池构造无界 fake×real 矩阵。图像 real 和 FS/FR fake 按来源视频等权；31 方法宏平均中的方法权重固定，pair 转移和 bootstrap 复用同一权重定义。

## 7. 源校准、评价与统计

每个 backbone/seed/flow/评价单位独立源校准。FPR5：取使源真实校准 FPR ≤0.05 的最小可行阈值；判 fake 当且仅当 `score >= threshold`。同分整体处理，保存源真实样本数、阈值规则与实际源 FPR。候选阈值固定为每个不同的 FP64 源真实分数及其朝正无穷的 `nextafter` 值，取满足约束的最小候选，不由测试分数生成候选；140 个等权真实视频最多允许 7 个 FP，分数并列必须整体排除或整体纳入。图像层按视频等权的帧权重计算 FPR，不能直接复用视频的整数 FP 预算。真实帧在图像层保持视频内总权重相同，视频层按视频等权。

冻结阈值原样用于 FF++ test 和 CDF test；视频阈值不挪用于 EFS 图片。报告源/目标 AUROC、FPR、逐方法 TPR、宏 TPR、宏 BA、逐类/类平衡 Brier。四流程概率视图定义固定：RL/DL 的 Brier 使用 sigmoid(video logit)，RP/DP 使用概率均值；在对应 logit/prob 分数空间执行阈值，不能用 Brier 概率重选阈值。

每个 backbone 单独展示全部 5 seed、均值/SD/范围，流程差异同 seed 配对。固定模型后源校准来源组重采样 1000 次，测试来源组配对 bootstrap 2000 次，raw/direction/聚合流程共享抽样权重；校准抽样与目标测试抽样不确定性分开。EFS 使用经登记的可得生成批次或存储组，并明确其假设，不把所有图片无条件视为独立来源。AUC 使用 ties-aware 计数；因抽样只有一类产生的无效重复登记，不能静默少报迭代数。

TPR95 当前只记录不可用原因，不输出冒充完整方法覆盖的结果。任何目标 oracle、尺度网格和机制结果只能作诊断，不反馈选择训练模型、主流程或源阈值。

## 8. 工件、路径与阶段产物

| 位置 | 内容 |
| --- | --- |
| `configs/` | 解析后的实验协议、31 方法、路径映射、模型/运行参数 |
| `references/` | 上游参考版本与许可证；与本轮可执行模块分开 |
| `artifacts/data/` | 官方 split 原件/哈希、方法登记、冻结输入与校准清单、审计/隔离账本 |
| `artifacts/runs/<run_id>/` | 15 模型各自状态、epoch 日志、最终权重、环境、RNG |
| `artifacts/features/` | 按模型/角色分块 h、r、头参数与元信息 |
| `artifacts/scores/` | 四视频流程和两图像条件、机制干预分数、冻结阈值和预测 |
| `artifacts/analysis/` | 30 匹配头、逐 seed/方法指标、pair 表和 bootstrap 摘要 |
| `reports/` | 可追溯主表、图、异常/失败说明与最终报告 |

所有阶段输出采用临时文件加原子完成标记；完成标记绑定输入哈希。每阶段有独立 CLI，允许在匹配输入下断点恢复。数据与特征清单可由逐样本工件重算主表，历史 16 方法的结果不能混入 31 方法主表。

## 9. GPU 与 LLM 生命周期

只读核查时，物理 GPU 4–7 被 Docker 容器 `onecat-vllm` 占用；容器为 TP=4，restart policy=`unless-stopped`。GPU 0–3 已有其他深伪实验。网关只转发请求，不负责重启该模型容器。

1. CPU 数据审计、源码回归检查、模型初始化来源与环境准备通过之前保持 LLM 服务运行；不为了等待设计/安装提前释放 GPU。
2. 正式进入 GPU 探针前，重新核实物理 GPU index/UUID 与当前 compute PID，避免引用过期 PID。保存该容器 ID、运行状态、GPU UUID、恢复命令和服务状态，不保存或显示 API key/完整环境。
3. 若容器原本运行，执行 `docker stop --time 60 onecat-vllm`；确认目标容器已停止且物理 4–7 没有其他进程，再运行 GPU 探针/队列。不得用宽泛 pkill，不删除/重建容器，不修改 API、模式或启动参数。
4. 每个 run 用明确的物理 GPU/UUID 绑定；容器内 0–3 是原 LLM 的设备编号，不能误当成主机 0–3。训练/推理进程只使用已授权物理 4–7。
5. 队列保存 run 状态及续训入口；发生数据审计、协议或回归错误停止正式队列，记录错误，不继续产出主结果。OOM 先结束探针并按固定数值/有效 batch 协议调整，正式 run 设置发生变化时不得悄悄拼接原实验。
6. 完成所有必要 GPU 训练/特征提取后，或者发生终止性错误且项目自己的 GPU 进程退出后，恢复原本运行的容器：`docker start onecat-vllm`。模型冷启动可能需要等待；确认容器状态与不泄露凭据的健康状态。恢复失败记录并通知用户，不重复占用未释放的显存。
7. 原本已停止的容器不自动启动；原本运行状态和恢复意图写入持久作业账本，runner 异常重启也能读到。CPU 缓存分析可在 LLM 恢复后继续。

不改变 0–3 卡作业、其他 Docker 服务或其他项目环境。用户已授权为实验暂时暂停该 LLM；真正的停止/启动属于实施阶段，本书面设计阶段不执行。

## 10. 必要验证与完成条件

| 验证 | 成功条件 |
| --- | --- |
| 跨方法同 ID | 两个 fake 视频和一个 real 返回三个独立视频；分数不跨方法平均 |
| 抽帧边界 | 两方法同 ID 各取 8 帧，共 16；数值帧序、少帧和重复帧规则可复核 |
| 官方角色 | 数字双 ID 误标样本恢复正确角色；源 val/test 不进入 train；无法可靠确定官方角色的条目隔离，已证实跨角色来源冲突按审计门处理 |
| 真实去重 | 跨方法相同真实输入只入池一次；逐方法比较保留共享真实集合 |
| EFS 单位 | 独立图片不误聚合成视频，不凭存储 ID 构造真实帧配对 |
| 特征一致性 | 缓存特征与模型直接输出的范数及 signed raw/direction logit 一致 |
| 计分负对照 | logit 均值与概率均值的定义正确；单图严格单调排序的数值 ties 单独报告 |
| 阈值与来源隔离 | 仅官方源校准拟合；同分及 >= 规则一致；仿射同步阈值预测不变 |
| 匹配头 | 骨干和运行统计冻结；两头共享容量、权重、预算及初始化 |
| 配对统计 | 同 checkpoint、同输入、同 resample；pair 转移重建 AUC；共享 real 不当独立重复 |
| 训练及恢复 | 15 epoch final，无 val 选模；有效 batch/epoch 槽位符合配置；恢复检查输入哈希 |
| GPU 生命周期 | 仅停止指定容器并使用 4–7；GPU 队列结束后恢复原服务，0–3 未受影响 |

完成要求：31 方法输入审计没有未解决阻断，15 次训练与30匹配头的状态完整披露，所有预定可用分析与失败有账本，逐样本输出能重算主表，无 CDF 驱动的选择。允许结果支持 raw、direction、异质性或无稳定最优流程，不预设论文结论。

## 11. 设计审阅后进入的实施阶段

本轮会话已批准上述设计方向，尚未批准本文或书面实施计划。下一阶段为 `writing-plans`：把数据审计/清单、回归测试、模型/训练、特征/计分、统计与 GPU 管理拆成有明确检查点的执行任务，先完成源码与数据检查，再进入 GPU 探针和正式队列。

本文件和 v1.1 快照会提交到新项目的 `codex/df40-redesign` 分支。书面设计审阅通过后才能编写实施计划；实施计划审阅并选定执行方式后开始实现、暂停服务和实验。当前没有训练结果，也没有改变原数据、Obsidian 原稿、上游仓库或 GPU 服务状态。
