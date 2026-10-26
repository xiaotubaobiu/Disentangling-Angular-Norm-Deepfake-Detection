# 本机 DF40 数据核查：发布角色、来源角色与内容重复

核查日期：2026-10-07（Asia/Seoul）。依据是机器上的目录、JSON 和有界图片哈希核查。未修改原始数据，未启动训练或暂停 GPU 服务；不是全量图片完整性/解码审计。

## 1. 修正上一版的理解

上一版把 DF40 提供的 train/test 角色与原始 FF++ 来源视频 ID 的划分合并为一个含义，并称来源 ID 重新分配为“恢复官方 DF40 角色”。这个说法证据不足。现在必须分列：

| 字段 | 来自哪里 | 能说明什么 |
| --- | --- | --- |
| `released_role` | DF40 的方法 JSON train/test | 提供者声明的数据用途 |
| `physical_role` | 实际 DF40_train / DF40_test 存放位置 | 本地副本所在位置，不自动判定角色正确性 |
| `upstream_source_role` | 可验证的 FF++ 原始来源 ID 与原始 split | 原始素材的来源角色，不自动覆盖 released_role |
| 内容重复证据 | SHA256 / 图片内容检查 | 能确认图片副本，不靠同名 ID 或 seed 猜测 |
| `experiment_role` | 明确记录、审阅后的本轮划分策略 | 本轮真正使用的 train/calibration/test 角色 |

来源角色不同、内容重复、未知来源和需要新的校准划分是四种问题。不能全部叫 JSON 错误，也不能以数字目录名直接得出图片生成谱系。

## 2. 实际目录和角色

| 本地位置 | 实际含义与核查结果 |
| --- | --- |
| `/Dataset/deepfake_detection/DF40_all/DF40_train` | 31 个方法的训练侧物理数据；FS/FR 是帧目录，EFS 是生成图片存储桶 |
| `/Dataset/deepfake_detection/DF40_all/DF40_test` | 40 个方法；主集合方法通常分 ff/cdf，其他方法有独立 real/fake 目录 |
| `/Dataset/deepfake_detection/DF40_all/dataset_json_update` | 31 个 FF 方法 JSON 都只有 train/test；CDF 的 train 为空，test 非空；无声明的 val 角色 |
| `FSAll_ff/FRAll_ff/EFSAll_ff` | 恰好是各 9/12/10 方法索引的并集；真实数据分别重复 9/12/10 次 |
| `DF40_all.json` | 训练聚合索引，test 为空，且包括本轮排除方法；不能充当本轮全角色清单 |
| DF40_all 下的 FF++ 真实帧目录 | 999 视频组；按原始 FF++ split 可标注 train719/val140/test140，缺 train ID281 |
| DF40_all 下的 CDF 真实帧目录 | Celeb-real588 + YouTube-real300；方法 JSON 评价真实池是178组 |
| `/Dataset/deepfake_detection/Celeb-DF-v2-raw` | 本机还有完整原版视频：Celeb-real590、YouTube-real300、Celeb-synthesis5639；本地测试名单518项=178 real+340 fake |

原版 CDF 的 5,639 个伪造视频与 DF40 各方法生成的 CDF 域伪造数据不是同一集合。机器上有原版全量视频，不表示本轮检测器应读取原版伪造数据或用 CDF 训练。

提供的 FF fake JSON 数量（未内容去重/解码；FS/FR 为视频组，EFS 为存储组）：

| 类型 | JSON train | JSON test |
| --- | --- | --- |
| FS 9方法 | 6,512组 / 206,662帧 | 1,931组 / 61,478帧 |
| FR 12方法 | 8,481组 / 269,606帧 | 1,678组 / 53,308帧 |
| EFS 10方法 | 5,640存储组 / 180,070图片路径 | 1,160存储组 / 37,110图片路径 |

各方法共享真实池：FF train719组/22,993帧、FF test140组/4,479帧。跨方法聚合不能把同一真实集合算成新增样本。

## 3. FS/FR：物理副本与来源差异分开

| 方法 | 物理 train / FF test 组数 | 完整同名组重叠 | 有界哈希核查 |
| --- | --- | --- | --- |
| SimSwap | 988 / 140 | 140 | 8组×4帧，32/32图片对相同 |
| InSwapper | 883 / 118 | 118 | 8组×4帧，32/32相同 |
| MobileSwap | 718 / 858 | 718 | 8组×4帧，32/32相同 |
| UniFace | 372 / 140 | 0 | 没有同名组可按此方式比较 |

前三方法抽查的24组中，每组完整帧文件名集合也相同。此处96/96是抽样结果，不是对所有重叠组的全帧哈希结论。

从可解析数字来源观察，SimSwap train 中含原 FF train713/val135/test140，InSwapper含664/101/118，UniFace含318/54/0，MobileSwap test含718/0/140。这些分布也存在于物理数据。UniFace 的54个 val来源不构成“官方DF40索引错误”的直接证据；但它会影响选择原FF val作为本轮来源独立校准集的假设。

Wav2Lip/SadTalker train/test完整视频目录名重叠均为0。数字后缀如50001不唯一，不是已确认驱动身份。完整后缀有70种跨角色复用，但没有驱动音频映射。两方法各抽16个同后缀不同目标帧对均为0/16相同图片；这不能证明音频无复用，也不能把它们当相同伪造视频或强制排除。

SadTalker 的8个异常目录是4个 `000##…` 及对应4个 `temp_000##…`；各对有32个共同帧名，首帧哈希相同。其余帧未全量核对，不能只因目标ID不能解析把8组全删。

## 4. EFS：存储桶与真正图片重复

检查 StyleGAN2、DiT、sd2.1、VQGAN、RDDM：

| 对象 | 本地发现 | 解释边界 |
| --- | --- | --- |
| FF fake JSON train | 每方法564桶 / 18,007路径 | 数字桶不是已确认的来源视频 |
| FF fake JSON test | 每方法116桶 / 3,711路径 | 不能假定与train独立 |
| 物理 train | 每方法676桶，比JSON多112桶 | 映射到原FF val形似ID不证明官方fake validation |
| 物理 FF test | 通常140桶，sd2.1有999桶 | 不能拿目录数当图像有效样本数 |
| train内跨桶同名图片 | 五方法共20/20抽样对字节相同 | 存储桶之间也有图片复制 |
| JSON train↔test同名图片 | 五方法共20/20抽样对字节相同 | 有真实跨角色内容重叠 |
| JSON train/test共有basename | 每方法723个 | 未逐个哈希，不能称全部723张都重复 |
| FF↔CDF同basename | 抽查16/16内容不同 | 跨域相同seed/样本名不能当副本 |

例如 sd2.1 外层 `ff/446` 中的图片名可以是 `805_343.png`；CDF `id6_0004` 桶中可以同时有 `Celeb-real_id27_0003_163.png` 和 `YouTube-real_00231_276.png`。文件名的其他ID是否表示生成条件仍需生成元信息；外桶名更不能被默认当作同帧来源。

EFS fake train的旧JSON仍使用 `DF40/<method>/ff/...` 前缀，本地却有 `DF40_train/<method>/...` 副本；路径解析须同时核对方法、发布角色和实际存在性，不能全局把 `DF40/` 替换为 `DF40_test/`。

sd2.1 CDF真实物理组织是 `Celeb-real/YouTube-real`；抽查的其他四方法为 `Fake_from_Celeb-real/Fake_from_Youtube-real`。需逐方法解析，不能统一替换。RDDM的CDF索引444桶/14,027图，来自Celeb路径；物理存在大量空桶，不能用888个目录宣称888个有效样本来源。

## 5. 主代理直接复核的副本证据

以下路径对已由主代理重新计算SHA256，均逐字节相同：

| 方法 | 训练侧相对路径 | 测试侧相对路径 | SHA256 |
| --- | --- | --- | --- |
| SimSwap | `DF40_train/simswap/frames/000_003/000.png` | `DF40_test/simswap/ff/frames/000_003/000.png` | `591a8208d5b211479e69b684bb0244b339aadc73511612204fb9b338e9540046` |
| InSwapper | `DF40_train/inswap/frames/003_000/000.png` | `DF40_test/inswap/ff/frames/003_000/000.png` | `0e01a46f03cb00773a5154a60d0517815f21fa109155fc127c85f24000ae68d3` |
| MobileSwap | `DF40_train/mobileswap/frames/001_870/000.png` | `DF40_test/mobileswap/ff/frames/001_870/000.png` | `5bf01b27d179fa75fef2b0af90a76f3e5b21117ec63edebbbd878381c7fcd726` |
| StyleGAN2 | `DF40_train/StyleGAN2/765/seed0016.png` | `DF40_test/StyleGAN2/ff/003/seed0016.png` | `0db5a1c94b6d811b934aba4e39c8bf5318ce527e49a21a9676f8e5b8c64a8e06` |
| DiT | `DF40_train/DiT/840/10028.png` | `DF40_test/DiT/ff/000/10028.png` | `2c8cda7633303e0b1f403aa6b8202df63e120d698d3814124a72a07c0b792fef` |
| SD-2.1 | `DF40_train/sd2.1/840/006_199.png` | `DF40_test/sd2.1/ff/000/006_199.png` | `e4c05510297e216bbd8670ee57bf4af5a83072a703b9b6374e68fb0046472354` |

相对路径的根均为 `/Dataset/deepfake_detection/DF40_all`。

## 6. 对设计的确定修正与尚需定义的协议

确定修正：三个角色字段分离；EFS只按独立图片计分；真实跨方法去重；直接核对文件副本；未知驱动/批次保持未知；不把桶名当自然来源；不声称DF40发布了完整validation；原FF140 real val只是候选校准数据。

需在实施前明确：本轮如何保留/分配跨train/test内容副本，是否采取原FF来源分组定义新的实验角色，以及独立源校准池如何与检测器训练输入隔离。不能一边声明保留DF40提供划分、一边用另一套来源ID静默迁移其角色。修复副本与重定义来源划分分别记录；没有按模型成绩挑数据，尚未执行任何迁移、删除或新划分。

31 方法范围、15次训练和30匹配头的科学预算仍保持，但具体数据清单未获批准，不据此启动正式训练。主集合的“三个输入池”明确指 FF train、FF test、CDF test，不是要求每个FF JSON有train/val/test。
