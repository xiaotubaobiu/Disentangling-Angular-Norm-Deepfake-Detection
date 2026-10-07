# DF40 计分机制与跨域阈值稳定性：实施设计

日期：2026-10-07（Asia/Seoul）。修订：按用户指出的数据理解问题重新核查机器上的DF40；本版数据章替代初版，等待审阅。科学范围已确认，具体数据清单/校准策略尚未批准。本文不是实验结果，也不是已经执行的实施计划。

## 1. 意图、范围与依据

在新项目 `/home/z/project/deepfake_detector` 中执行 Obsidian v1.1：解释特征长度处理、帧级非线性、视频聚合的交互作用，评价独立源校准后的跨域冻结阈值，以及五种子的稳定性。用户明确两个 GitHub 项目只作代码参考，不额外增加它们原方法的对照训练。

科学协议以同目录的 [v1.1 原稿快照](2026-10-07-df40-v1.1-source.md) 为准。原文件为 `/home/z/obsidian/paper/deepfake-detector/实验重设计_3backbone_5seed_v1.1.md`，SHA256 为 `61e583963a4d661ccf83d02798600c83eb4aca2238d2984c5422bab1553ecc6b`。快照逐字节保存，不改写 Obsidian 原稿。本文落实接口、审计、运行和故障处理，不扩大研究范围。原稿中的数据假设以本机证据复核，事实与假设冲突处明确记录；不能仅因原稿写了某个角色便声称磁盘已有对应独立数据。涉及划分或校准策略变化先完成审阅，不静默修改协议。

| 项目 | 固定约束 |
| --- | --- |
| 方法集合 | 31：9 FS + 12 FR + 10 EFS；e4s 保留，e4e 与全部 FE 排除 |
| 编码器 | CLIP ViT-B/16、DINOv2 ViT-B/14、EfficientNet-B0 |
| 训练种子 | `[7, 42, 123, 2026, 3407]` |
| 端到端训练 | 15 个 raw 模型，31 方法联合训练；每个模型 15 epoch，使用最终轮权重 |
| 匹配头 | 每个冻结编码器拟合 raw/direction 两个线性头，共 30 个 |
| 视频评价 | 21 FS/FR；一个域中一个方法的一个视频为一个单位，每视频固定 8 张不同有效帧 |
| 图像评价 | 全部 31 方法；FS/FR 所选帧与 EFS 独立图片分开注明单位 |
| 校准 | 需要独立FF源校准池；原FF val真实视频是候选，隔离策略尚待确认；FPR5为主、TPR95须完整fake覆盖 |
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

## 3. 数据层：本机事实、发布角色与实验划分

本节替代初版“官方角色恢复”的表述。机器核查记录和内容副本实例见 [本机 DF40 数据核查](2026-10-07-df40-local-data-audit.md)。数量、角色声明、源ID映射和文件哈希是不同证据层级。

数据根为 `/Dataset/deepfake_detection/DF40_all`，只读使用；不移动、重命名或改写原数据和原JSON。数据衍生物写入新项目 `artifacts/`。机器上还有 `/Dataset/deepfake_detection/Celeb-DF-v2-raw` 的完整原版CDF视频（590 Celeb-real、300 YouTube-real、5,639原版fake）；它与本轮DF40各方法CDF域的伪造数据分开，不自动纳入主实验。

### 3.1 实际发布角色与原始来源角色

| 层级 | 本机实际情况 |
| --- | --- |
| 31个方法的FF JSON | 只有train/test，无声明val角色 |
| 31个方法的CDF JSON | train为空，test非空 |
| FSAll/FRAll/EFSAll | 各9/12/10方法索引并集，真实池随方法重复 |
| DF40_all.json | 训练聚合，test为空，还含本轮排除方法 |
| FF真实预处理帧 | 999组；原FF来源train719、val140、test140，缺281 |
| CDF真实预处理帧 | Celeb-real588+YouTube-real300；方法评价索引共享178组 |

原始FF++ split来源仍为 `https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{train,val,test}.json`，实施时保存精确版本和内容哈希。它们用于标注可确认的原始视频来源；不能直接被称为DF40发布角色的官方修正器。分列 `released_role`（JSON声明）、`physical_role`（本地位置）、`upstream_source_role`（可验证来源）、`experiment_role`（本轮明确选定的用途）。记录来源不能静默改变样本的实验角色。

本地SimSwap、InSwapper、UniFace的训练侧含部分原FF val/test来源，MobileSwap测试侧含部分原FF train来源。这些分布也存在于实际目录，不是仅JSON路径标错。UniFace没有同名train/test视频重叠，不能只因54个val来源就称索引错误。

SimSwap/ InSwapper/ MobileSwap分别有140/118/718个完整同名视频组跨train/test。每方法8组×4帧的有界检查均32/32图片SHA256相同，证明部分实际副本重叠；不据此声称所有帧都已全量校验。内容副本修复与是否使用原FF来源划分定义新实验角色是两项不同操作，在实施计划中分别明确，不能仅改video_id来掩盖共享内容。

### 3.2 EFS单位、内容重复与路径

EFS是生成图片。外层数字/视频形似目录是存储桶，不默认表示自然视频、真实帧配对、生成条件来源或独立批次。FF索引中每方法fake train为564桶/18,007路径，test为116桶/3,711路径；检查的物理训练目录为676桶，而非索引宣称的564桶。多出的112桶不能被直接称为独立官方fake validation。

对StyleGAN2、DiT、sd2.1、VQGAN、RDDM做有界SHA256检查：跨train存储桶20/20图片对相同，JSON train/test跨角色20/20相同；每方法还有723个共同basename，尚未全部哈希。相同seed/文件名跨FF/CDF的16对却全部内容不同。依据内容进行重复核查，不能以文件名替代全量判断；存储桶级重新分组不能解决同一图片跨角色的问题。

逐方法、发布角色和实际存在性映射路径。EFS fake train的旧JSON也写 `DF40/<method>/ff/...`，但本地对应训练图片位于 `DF40_train/<method>/...`；不能把所有 `DF40/` 一律映射到 `DF40_test/`。sd2.1 CDF使用 `Celeb-real/YouTube-real`，检查的其他四方法使用 `Fake_from_Celeb-real/Fake_from_Youtube-real`；不能全局替换。RDDM实际有大量空CDF桶，索引444桶/14,027图不能被提升为888个有效来源。图像计分与存储组织/来源信息分列，EFS不输出伪造视频AUC或same-frame位移。

### 3.3 元信息、真实去重与有效帧

清单至少包含 `sample_id, domain, released_role, physical_role, upstream_source_role, experiment_role, label, method_id, method_type, evaluation_unit, video_id, source_ids, driving_source_id, storage_group_id, generator_id, generation_batch_id, image_path, original_image_path, frame_number, content_sha256, include, exclusion_reason`。角色变化需另附政策版本、变更原因和证据；所有模型共用批准后冻结的清单。

- 真实图片/视频跨方法去重，真实方法ID规范为 `real`；共享真实池用于逐方法比较不增加独立样本数。
- 视频身份仍是 `(domain, method_id, video_id)`。FS/FR的不同方法同名视频不混合抽帧或评价；跨角色相同视频/图片副本另作内容审计。
- 主视频层数值排序后固定8张不同有效帧；不足8张不补重复帧，不进入主视频表。图像层覆盖依原稿预定规则单独登记，不能把视频层不足8帧规则直接扩为训练和单图全部排除，也不能用于EFS稀疏桶。
- 所有输入做存在/解码核查，内容重复与跨角色证据记录；未知批次和驱动来源保留未知，不自动排除已有可靠发布角色的样本。
- Wav2Lip/SadTalker完整train/test目录名不重叠；后缀重复尚不能证明相同音频/驱动，更不能当同一伪造视频。SadTalker异常及temp目录存在首帧副本证据，但尚无全帧证据；先登记候选重复，不因ID不能解析全部删除。
- 主集合必须覆盖FF训练、FF测试、CDF测试三个输入池，并保持31方法。这里不要求每个FF JSON有validation；任何副本政策不得静默把某方法训练池清空后仍称31方法训练。

### 3.4 源校准条件与进入实施的边界

原FF来源val的140个真实组在磁盘存在，共4,477张图片，全部至少8帧；它们是候选数据，尚不能直接声明独立源校准池已经恢复。必须结合检测器实际训练清单、已确认来源关系和内容隔离确定独立性；EFS桶名不能替代此证据。部分FS和EFS在物理上有原FF val形似来源/桶，并不等于DF40提供了独立fake validation。

科学目标仍为独立源FPR5阈值；在校准策略及其与训练的隔离定义获得批准之前，不执行冻结阈值实验。TPR95仍要求完整独立fake校准覆盖，目前不能声明具备。如果不能取得独立池，按v1.1保留排序实验与校准不可用记录；不擅自随机重划，不拿FF test/CDF作校准。

实施前必须明确副本归属/去重政策和源校准策略；这属于修正数据理解后仍需完成的实验定义。本版撤回初版按来源ID自动迁移/隔离的规定，不执行任何新划分、样本删除或原始文件变更。

## 4. 新项目架构与接口

使用独立 Python 包 `src/deepfake_detector/`，避免旧平铺包 `data/models/engine` 的跨项目导入冲突。CLI 以项目根和配置文件为锚点，不依赖调用目录。模块按以下边界划分：

| 模块 | 职责 | 输入 → 输出 |
| --- | --- | --- |
| `data/registry` | 固定方法与路径别名 | 注册配置 → 方法登记表 |
| `data/manifest` | 发布/来源角色分列、内容与路径审计、批准后的角色策略和8帧固定池 | 原JSON / 来源注释 / 文件 / 已批准策略 → 冻结清单与审计报告 |
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

30 个匹配头均在各自编码器完全冻结、eval 模式的已批准源训练清单特征上拟合。raw/direction 头均零初始化，L2=1e-4、不惩罚偏置、迭代上限 2000，记录收敛与固定权重；不更新骨干运行统计，不使用目标标签选正则或挑头。

pair 转移采用正确/并列/错误的 3×3 表，ties 按半分；重建的 DeltaAUC 与直接值在 1e-10 内一致。实现采用可核对的加权/分块算法，避免对大图像池构造无界 fake×real 矩阵。图像 real 和 FS/FR fake 按来源视频等权；31 方法宏平均中的方法权重固定，pair 转移和 bootstrap 复用同一权重定义。

## 7. 源校准、评价与统计

每个 backbone/seed/flow/评价单位独立源校准。FPR5：取使源真实校准 FPR ≤0.05 的最小可行阈值；判 fake 当且仅当 `score >= threshold`。同分整体处理，保存源真实样本数、阈值规则与实际源 FPR。候选阈值固定为每个不同的 FP64 源真实分数及其朝正无穷的 `nextafter` 值，取满足约束的最小候选，不由测试分数生成候选；若最终批准池包含140个等权真实视频，最多允许7个FP；其他数量按同一0.05约束计算，分数并列必须整体排除或整体纳入。图像层按视频等权的帧权重计算 FPR，不能直接复用视频的整数 FP 预算。真实帧在图像层保持视频内总权重相同，视频层按视频等权。

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
| 数据角色 | 发布/物理/原始来源/本轮实验角色分列；只按批准的数据策略改变实验角色；内容副本检查与来源分组分开，不能把EFS桶名当来源 |
| 真实去重 | 跨方法相同真实输入只入池一次；逐方法比较保留共享真实集合 |
| EFS 单位 | 独立图片不误聚合成视频，不凭存储 ID 构造真实帧配对 |
| 特征一致性 | 缓存特征与模型直接输出的范数及 signed raw/direction logit 一致 |
| 计分负对照 | logit 均值与概率均值的定义正确；单图严格单调排序的数值 ties 单独报告 |
| 阈值与来源隔离 | 仅批准并验证隔离的源校准池拟合；无合格校准池时不执行阈值实验；同分及 >= 规则一致；仿射同步阈值预测不变 |
| 匹配头 | 骨干和运行统计冻结；两头共享容量、权重、预算及初始化 |
| 配对统计 | 同 checkpoint、同输入、同 resample；pair 转移重建 AUC；共享 real 不当独立重复 |
| 训练及恢复 | 15 epoch final，无 val 选模；有效 batch/epoch 槽位符合配置；恢复检查输入哈希 |
| GPU 生命周期 | 仅停止指定容器并使用 4–7；GPU 队列结束后恢复原服务，0–3 未受影响 |

完成要求：31 方法输入审计没有未解决阻断，15 次训练与30匹配头的状态完整披露，所有预定可用分析与失败有账本，逐样本输出能重算主表，无 CDF 驱动的选择。允许结果支持 raw、direction、异质性或无稳定最优流程，不预设论文结论。

## 11. 设计审阅后进入的实施阶段

本轮会话已批准上述设计方向，尚未批准本文或书面实施计划。下一阶段为 `writing-plans`：把数据审计/清单、回归测试、模型/训练、特征/计分、统计与 GPU 管理拆成有明确检查点的执行任务，先完成源码与数据检查，再进入 GPU 探针和正式队列。

本文件和 v1.1 快照会提交到新项目的 `codex/df40-redesign` 分支。书面设计审阅通过后才能编写实施计划；实施计划审阅并选定执行方式后开始实现、暂停服务和实验。当前没有训练结果，也没有改变原数据、Obsidian 原稿、上游仓库或 GPU 服务状态。
