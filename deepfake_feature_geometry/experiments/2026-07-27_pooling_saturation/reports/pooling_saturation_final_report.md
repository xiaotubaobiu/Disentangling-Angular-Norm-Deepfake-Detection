# Pooling / Saturation 机制研究 — 最终报告
- 日期:2026-07-27
- 分支:`study/pooling-saturation`
- 数据:FF++(test)+ CDF(DFD 排除);raw-trained checkpoint × {clip,dinov2,efficientnet} × 5 seeds × 16 methods
- 所有校准参数(T / robust-scale / Platt)仅在 **FF++ val** 拟合;CDF γ_best 仅作 oracle 诊断。

## 0. 结论摘要

**判定:A 为主因 + D(方法依赖):direction>raw 的视频级 mean_prob 现象本质是 sigmoid 饱和 + 概率聚合伪影(校准后归零),但其符号/幅度强依赖伪造方法(FSGAN 与 SadTalker 相反)**

- **池化伪影(A 信号)**:在 7/32 个 method×domain 上 `Delta_prob>0 且 Delta_logit≤0`(占比 22%)。
- 温度校准:`Delta_prob>0` 的 combo 经温度后 `Delta_temp` 均值 = +0.0005(≈0 或转负 ⇒ 强支持 A)。
- 饱和率差(fake @0.10):raw − direction 均值 = +0.9449(raw 更饱和 ⇒ 支持 A)。
- 视频内方差下降 std(raw)−std(dir) 均值 = +1.3568(direction 降低方差 ⇒ 支持 C)。
- **真实方向作用(B 信号)**:10/32 个 combo 在 `mean_logit`(无饱和)下 direction 仍 `>0`。
- 温度后仍有 8/32 个 combo direction `>0`。
- **方法依赖(D 信号)**:I_pool FSGAN − SadTalker = +0.0418(符号相反、幅度大 ⇒ 强 D)。FSGAN I_pool=+0.0167,SadTalker I_pool=-0.0252。

**下一步方法建议:** (1) 视频级评估必须温度校准(FF++ val 拟合 T)后再聚合,或直接用 mean_logit/median 等无饱和聚合;(2) 因符号方法依赖,不得将 mean_prob 优势解读为方向特征更好;(3) 暂停径向—角向复杂训练;若需自适应,用样本级径向可靠性门控,不得依赖测试时 method label。

## 1. 实验1 聚合器对照(prob vs logit 池化)

`delta_auc = AUC_direction − AUC_raw`(均值 over seed/backbone)。`I_pool = Delta_prob − Delta_logit`。

|                         |   Delta_prob |   Delta_logit |   I_pool |   Delta_temp |   Delta_scale |
|:------------------------|-------------:|--------------:|---------:|-------------:|--------------:|
| ('MRAA', 'cdf')         |      -0.006  |       -0.0133 |   0.0073 |      -0.0086 |       -0.0131 |
| ('MRAA', 'ffpp')        |      -0.0001 |       -0.0001 |   0      |      -0      |       -0.0001 |
| ('blendface', 'cdf')    |       0.0012 |        0.0032 |  -0.0019 |       0.0024 |        0.0032 |
| ('blendface', 'ffpp')   |       0.0004 |        0.0002 |   0.0002 |       0.0001 |        0.0002 |
| ('e4s', 'cdf')          |      -0.001  |       -0.0026 |   0.0016 |      -0.0018 |       -0.0026 |
| ('e4s', 'ffpp')         |       0.0008 |       -0.0001 |   0.0009 |      -0      |       -0.0001 |
| ('facedancer', 'cdf')   |       0.0094 |        0.0022 |   0.0072 |       0.0022 |        0.0022 |
| ('facedancer', 'ffpp')  |       0.0004 |        0      |   0.0004 |      -0.0001 |        0.0001 |
| ('faceswap', 'cdf')     |      -0.0206 |       -0.0131 |  -0.0075 |      -0.0098 |       -0.013  |
| ('faceswap', 'ffpp')    |       0      |       -0.0008 |   0.0008 |      -0.0005 |       -0.0007 |
| ('facevid2vid', 'cdf')  |      -0.0032 |       -0.0031 |  -0.0001 |      -0.0025 |       -0.0031 |
| ('facevid2vid', 'ffpp') |      -0      |       -0      |   0      |       0      |       -0      |
| ('fomm', 'cdf')         |      -0.0052 |       -0.0051 |  -0.0001 |      -0.0042 |       -0.0051 |
| ('fomm', 'ffpp')        |      -0.0001 |       -0      |  -0      |      -0      |       -0      |
| ('fsgan', 'cdf')        |       0.0292 |       -0.0026 |   0.0318 |      -0      |       -0.0026 |
| ('fsgan', 'ffpp')       |       0.0016 |        0.0001 |   0.0015 |       0.0001 |        0.0001 |
| ('inswap', 'cdf')       |      -0.0006 |       -0.0025 |   0.0019 |      -0.0018 |       -0.0026 |
| ('inswap', 'ffpp')      |       0      |       -0      |   0.0001 |      -0.0001 |       -0      |
| ('lia', 'cdf')          |      -0.0007 |       -0.0116 |   0.0109 |      -0.0057 |       -0.0114 |
| ('lia', 'ffpp')         |       0.0007 |       -0.0002 |   0.0009 |      -0.0002 |       -0.0002 |
| ('mobileswap', 'cdf')   |      -0.0055 |       -0.0028 |  -0.0027 |      -0.0025 |       -0.0028 |
| ('mobileswap', 'ffpp')  |       0.0004 |        0      |   0.0004 |      -0      |        0      |
| ('pirender', 'cdf')     |      -0.0049 |       -0.0072 |   0.0023 |      -0.0057 |       -0.0072 |
| ('pirender', 'ffpp')    |      -0.0001 |       -0      |  -0      |      -0      |       -0      |
| ('sadtalker', 'cdf')    |      -0.0478 |        0.0035 |  -0.0514 |       0.0002 |        0.0034 |
| ('sadtalker', 'ffpp')   |       0.001  |       -0      |   0.001  |      -0      |       -0.0001 |
| ('simswap', 'cdf')      |      -0.0037 |       -0.0005 |  -0.0032 |      -0.0001 |       -0.0005 |
| ('simswap', 'ffpp')     |       0.0002 |        0.0002 |   0      |      -0      |        0.0002 |
| ('tpsm', 'cdf')         |      -0.0087 |       -0.0034 |  -0.0053 |      -0.0024 |       -0.0034 |
| ('tpsm', 'ffpp')        |       0.0006 |       -0      |   0.0006 |      -0.0001 |       -0      |
| ('wav2lip', 'cdf')      |       0.0029 |        0.0045 |  -0.0015 |       0.0033 |        0.0044 |
| ('wav2lip', 'ffpp')     |       0.0021 |        0.0004 |   0.0017 |       0.0004 |        0.0004 |

- **关键判读**:若 `Delta_prob>0` 但 `Delta_logit≤0`,则 direction 在视频级的「领先」来自概率聚合而非特征本身(指向 A)。

## 2. 实验2 饱和统计

fake 帧饱和率 @0.10(raw vs direction,均值):

|                         |   sat_dir |   sat_raw |   sat_excess_raw |
|:------------------------|----------:|----------:|-----------------:|
| ('MRAA', 'cdf')         |         0 |    0.9718 |           0.9718 |
| ('MRAA', 'ffpp')        |         0 |    0.987  |           0.987  |
| ('blendface', 'cdf')    |         0 |    0.98   |           0.98   |
| ('blendface', 'ffpp')   |         0 |    0.9499 |           0.9499 |
| ('e4s', 'cdf')          |         0 |    0.9857 |           0.9857 |
| ('e4s', 'ffpp')         |         0 |    0.9544 |           0.9544 |
| ('facedancer', 'cdf')   |         0 |    0.9572 |           0.9572 |
| ('facedancer', 'ffpp')  |         0 |    0.8761 |           0.8761 |
| ('faceswap', 'cdf')     |         0 |    0.9715 |           0.9715 |
| ('faceswap', 'ffpp')    |         0 |    0.8333 |           0.8333 |
| ('facevid2vid', 'cdf')  |         0 |    0.9992 |           0.9992 |
| ('facevid2vid', 'ffpp') |         0 |    0.9989 |           0.9989 |
| ('fomm', 'cdf')         |         0 |    0.9989 |           0.9989 |
| ('fomm', 'ffpp')        |         0 |    0.9977 |           0.9977 |
| ('fsgan', 'cdf')        |         0 |    0.9741 |           0.9741 |
| ('fsgan', 'ffpp')       |         0 |    0.8988 |           0.8988 |
| ('inswap', 'cdf')       |         0 |    0.8676 |           0.8676 |
| ('inswap', 'ffpp')      |         0 |    0.889  |           0.889  |
| ('lia', 'cdf')          |         0 |    0.8918 |           0.8918 |
| ('lia', 'ffpp')         |         0 |    0.9391 |           0.9391 |
| ('mobileswap', 'cdf')   |         0 |    0.9708 |           0.9708 |
| ('mobileswap', 'ffpp')  |         0 |    0.9805 |           0.9805 |
| ('pirender', 'cdf')     |         0 |    0.9957 |           0.9957 |
| ('pirender', 'ffpp')    |         0 |    0.9972 |           0.9972 |
| ('sadtalker', 'cdf')    |         0 |    0.8972 |           0.8972 |
| ('sadtalker', 'ffpp')   |         0 |    0.8737 |           0.8737 |
| ('simswap', 'cdf')      |         0 |    0.9419 |           0.9419 |
| ('simswap', 'ffpp')     |         0 |    0.954  |           0.954  |
| ('tpsm', 'cdf')         |         0 |    0.9532 |           0.9532 |
| ('tpsm', 'ffpp')        |         0 |    0.9784 |           0.9784 |
| ('wav2lip', 'cdf')      |         0 |    0.8932 |           0.8932 |
| ('wav2lip', 'ffpp')     |         0 |    0.8787 |           0.8787 |

- `sat_excess_raw>0` ⇒ raw 比 direction 更饱和(支持 A:raw 受 sigmoid 压缩更严重)。

## 3. 实验3 Jensen gap

`J = mean σ(s) − σ(mean s)`,per-video `|J|` 均值(raw vs direction):

|                         |   Jabs_dir |   Jabs_raw |
|:------------------------|-----------:|-----------:|
| ('MRAA', 'cdf')         |     0.0001 |     0.0149 |
| ('MRAA', 'ffpp')        |     0.0001 |     0.0177 |
| ('blendface', 'cdf')    |     0.0001 |     0.0132 |
| ('blendface', 'ffpp')   |     0.0001 |     0.0222 |
| ('e4s', 'cdf')          |     0.0001 |     0.0194 |
| ('e4s', 'ffpp')         |     0.0001 |     0.0234 |
| ('facedancer', 'cdf')   |     0.0001 |     0.0173 |
| ('facedancer', 'ffpp')  |     0.0001 |     0.0291 |
| ('faceswap', 'cdf')     |     0.0001 |     0.0136 |
| ('faceswap', 'ffpp')    |     0.0001 |     0.0353 |
| ('facevid2vid', 'cdf')  |     0.0001 |     0.0101 |
| ('facevid2vid', 'ffpp') |     0.0001 |     0.0167 |
| ('fomm', 'cdf')         |     0.0001 |     0.0101 |
| ('fomm', 'ffpp')        |     0.0001 |     0.0168 |
| ('fsgan', 'cdf')        |     0.0001 |     0.0201 |
| ('fsgan', 'ffpp')       |     0.0001 |     0.0306 |
| ('inswap', 'cdf')       |     0.0001 |     0.0282 |
| ('inswap', 'ffpp')      |     0.0001 |     0.0273 |
| ('lia', 'cdf')          |     0.0001 |     0.0257 |
| ('lia', 'ffpp')         |     0.0001 |     0.0257 |
| ('mobileswap', 'cdf')   |     0.0001 |     0.0149 |
| ('mobileswap', 'ffpp')  |     0.0001 |     0.0098 |
| ('pirender', 'cdf')     |     0.0001 |     0.011  |
| ('pirender', 'ffpp')    |     0.0001 |     0.0169 |
| ('sadtalker', 'cdf')    |     0.0001 |     0.0144 |
| ('sadtalker', 'ffpp')   |     0.0001 |     0.0209 |
| ('simswap', 'cdf')      |     0.0001 |     0.018  |
| ('simswap', 'ffpp')     |     0.0001 |     0.0211 |
| ('tpsm', 'cdf')         |     0.0001 |     0.0157 |
| ('tpsm', 'ffpp')        |     0.0001 |     0.0206 |
| ('wav2lip', 'cdf')      |     0.0001 |     0.0246 |
| ('wav2lip', 'ffpp')     |     0.0001 |     0.0321 |

- direction 是否系统性降低 |J| ⇒ 饱和造成的概率聚合偏移是否被方向归一化削弱。

## 4. 实验4 温度校准(FF++ val 拟合)

温度 T 与 ECE(校准前后):

| backbone     |   T_raw |   T_dir |   val_ece_before_raw |   val_ece_after_raw |   val_ece_before_dir |   val_ece_after_dir |
|:-------------|--------:|--------:|---------------------:|--------------------:|---------------------:|--------------------:|
| clip         |   1.224 |   0.106 |                0.027 |               0.027 |                0.353 |               0.03  |
| dinov2       |   1.507 |   0.032 |                0.022 |               0.019 |                0.368 |               0.036 |
| efficientnet |   1.732 |   0.139 |                0.061 |               0.056 |                0.316 |               0.059 |

温度后 `Delta_temp`(mean_prob)均值 per method:

|                         |   Delta_temp |
|:------------------------|-------------:|
| ('MRAA', 'cdf')         |      -0.0086 |
| ('MRAA', 'ffpp')        |      -0      |
| ('blendface', 'cdf')    |       0.0024 |
| ('blendface', 'ffpp')   |       0.0001 |
| ('e4s', 'cdf')          |      -0.0018 |
| ('e4s', 'ffpp')         |      -0      |
| ('facedancer', 'cdf')   |       0.0022 |
| ('facedancer', 'ffpp')  |      -0.0001 |
| ('faceswap', 'cdf')     |      -0.0098 |
| ('faceswap', 'ffpp')    |      -0.0005 |
| ('facevid2vid', 'cdf')  |      -0.0025 |
| ('facevid2vid', 'ffpp') |       0      |
| ('fomm', 'cdf')         |      -0.0042 |
| ('fomm', 'ffpp')        |      -0      |
| ('fsgan', 'cdf')        |      -0      |
| ('fsgan', 'ffpp')       |       0.0001 |
| ('inswap', 'cdf')       |      -0.0018 |
| ('inswap', 'ffpp')      |      -0.0001 |
| ('lia', 'cdf')          |      -0.0057 |
| ('lia', 'ffpp')         |      -0.0002 |
| ('mobileswap', 'cdf')   |      -0.0025 |
| ('mobileswap', 'ffpp')  |      -0      |
| ('pirender', 'cdf')     |      -0.0057 |
| ('pirender', 'ffpp')    |      -0      |
| ('sadtalker', 'cdf')    |       0.0002 |
| ('sadtalker', 'ffpp')   |      -0      |
| ('simswap', 'cdf')      |      -0.0001 |
| ('simswap', 'ffpp')     |      -0      |
| ('tpsm', 'cdf')         |      -0.0024 |
| ('tpsm', 'ffpp')        |      -0.0001 |
| ('wav2lip', 'cdf')      |       0.0033 |
| ('wav2lip', 'ffpp')     |       0.0004 |

- **关键判读**:`Delta_prob>0` 经温度后是否 → `Delta_temp≈0` 或转负?若是 ⇒ 强支持 A(池化伪影被温度消除)。

## 5. 实验5 尺度匹配(val 拟合 robust scale)

robust 标准化后 `Delta_scale`(mean_prob):

|                         |   Delta_scale |
|:------------------------|--------------:|
| ('MRAA', 'cdf')         |       -0.0131 |
| ('MRAA', 'ffpp')        |       -0.0001 |
| ('blendface', 'cdf')    |        0.0032 |
| ('blendface', 'ffpp')   |        0.0002 |
| ('e4s', 'cdf')          |       -0.0026 |
| ('e4s', 'ffpp')         |       -0.0001 |
| ('facedancer', 'cdf')   |        0.0022 |
| ('facedancer', 'ffpp')  |        0.0001 |
| ('faceswap', 'cdf')     |       -0.013  |
| ('faceswap', 'ffpp')    |       -0.0007 |
| ('facevid2vid', 'cdf')  |       -0.0031 |
| ('facevid2vid', 'ffpp') |       -0      |
| ('fomm', 'cdf')         |       -0.0051 |
| ('fomm', 'ffpp')        |       -0      |
| ('fsgan', 'cdf')        |       -0.0026 |
| ('fsgan', 'ffpp')       |        0.0001 |
| ('inswap', 'cdf')       |       -0.0026 |
| ('inswap', 'ffpp')      |       -0      |
| ('lia', 'cdf')          |       -0.0114 |
| ('lia', 'ffpp')         |       -0.0002 |
| ('mobileswap', 'cdf')   |       -0.0028 |
| ('mobileswap', 'ffpp')  |        0      |
| ('pirender', 'cdf')     |       -0.0072 |
| ('pirender', 'ffpp')    |       -0      |
| ('sadtalker', 'cdf')    |        0.0034 |
| ('sadtalker', 'ffpp')   |       -0.0001 |
| ('simswap', 'cdf')      |       -0.0005 |
| ('simswap', 'ffpp')     |        0.0002 |
| ('tpsm', 'cdf')         |       -0.0034 |
| ('tpsm', 'ffpp')        |       -0      |
| ('wav2lip', 'cdf')      |        0.0044 |
| ('wav2lip', 'ffpp')     |        0.0004 |

- 尺度匹配是否同样抹平方向优势?与温度结论是否一致。

## 6. 实验6 各聚合器 rank-flip (H−B)

mean H−B by method × aggregator:

| method      |   mean_prob |   mean_logit |   median_logit |   trimmed_logit |   temperature_mean_prob |   standardized_mean_prob |
|:------------|------------:|-------------:|---------------:|----------------:|------------------------:|-------------------------:|
| MRAA        |       -0    |        -0.01 |          -0.01 |           -0.01 |                   -0    |                    -0.01 |
| blendface   |        0    |         0    |           0    |            0    |                    0    |                     0    |
| e4s         |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| facedancer  |        0    |         0    |           0    |            0    |                    0    |                     0    |
| faceswap    |       -0.01 |        -0.01 |          -0.01 |           -0.01 |                   -0.01 |                    -0.01 |
| facevid2vid |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| fomm        |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| fsgan       |        0.02 |        -0    |           0    |           -0    |                    0    |                    -0    |
| inswap      |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| lia         |        0    |        -0.01 |          -0    |           -0.01 |                   -0    |                    -0.01 |
| mobileswap  |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| pirender    |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| sadtalker   |       -0.02 |         0    |          -0    |            0    |                    0    |                     0    |
| simswap     |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| tpsm        |       -0    |        -0    |          -0    |           -0    |                   -0    |                    -0    |
| wav2lip     |        0    |         0    |           0    |            0    |                    0    |                     0    |

- **identity 自检**:max|identity_error| = 1.08e-16(应≈0)。
- **关键判读**:FSGAN 是否只在 mean_prob 下 H≫B,mean_logit 下 H≈B 或 B>H?SadTalker 是否多数聚合器 B>H?

## 7. 实验7 视频内时间一致性

within-video std(raw vs direction)与离群比例下降:

|                         |   std_dir |   std_raw |   var_reduction |   out_red |
|:------------------------|----------:|----------:|----------------:|----------:|
| ('MRAA', 'cdf')         |    0.0598 |    1.4834 |          1.4236 |   -0.0024 |
| ('MRAA', 'ffpp')        |    0.0577 |    1.3222 |          1.2645 |   -0.0036 |
| ('blendface', 'cdf')    |    0.0553 |    1.3579 |          1.3025 |   -0.005  |
| ('blendface', 'ffpp')   |    0.0645 |    1.4437 |          1.3792 |   -0.0049 |
| ('e4s', 'cdf')          |    0.0597 |    1.4135 |          1.3538 |   -0.0046 |
| ('e4s', 'ffpp')         |    0.066  |    1.4332 |          1.3672 |   -0.0048 |
| ('facedancer', 'cdf')   |    0.0619 |    1.4773 |          1.4154 |   -0.0046 |
| ('facedancer', 'ffpp')  |    0.0751 |    1.5893 |          1.5142 |   -0.005  |
| ('faceswap', 'cdf')     |    0.0517 |    1.2708 |          1.2192 |   -0.0028 |
| ('faceswap', 'ffpp')    |    0.0819 |    1.6621 |          1.5801 |   -0.0054 |
| ('facevid2vid', 'cdf')  |    0.0386 |    1.0611 |          1.0225 |   -0.003  |
| ('facevid2vid', 'ffpp') |    0.0518 |    1.2089 |          1.1571 |   -0.0038 |
| ('fomm', 'cdf')         |    0.0437 |    1.2148 |          1.1711 |   -0.002  |
| ('fomm', 'ffpp')        |    0.0524 |    1.2347 |          1.1823 |   -0.0036 |
| ('fsgan', 'cdf')        |    0.0637 |    1.6083 |          1.5447 |   -0.0067 |
| ('fsgan', 'ffpp')       |    0.0775 |    1.6405 |          1.5629 |   -0.0045 |
| ('inswap', 'cdf')       |    0.0775 |    1.5947 |          1.5171 |   -0.0039 |
| ('inswap', 'ffpp')      |    0.075  |    1.5398 |          1.4648 |   -0.0047 |
| ('lia', 'cdf')          |    0.0733 |    1.6375 |          1.5642 |   -0.0016 |
| ('lia', 'ffpp')         |    0.0727 |    1.5764 |          1.5038 |   -0.0057 |
| ('mobileswap', 'cdf')   |    0.0583 |    1.4631 |          1.4047 |   -0.0049 |
| ('mobileswap', 'ffpp')  |    0.0514 |    1.3609 |          1.3095 |   -0.0051 |
| ('pirender', 'cdf')     |    0.0493 |    1.3324 |          1.2832 |   -0.0032 |
| ('pirender', 'ffpp')    |    0.0553 |    1.3077 |          1.2524 |   -0.0036 |
| ('sadtalker', 'cdf')    |    0.0456 |    0.9904 |          0.9448 |   -0.0019 |
| ('sadtalker', 'ffpp')   |    0.0589 |    1.2116 |          1.1528 |   -0.0032 |
| ('simswap', 'cdf')      |    0.064  |    1.4407 |          1.3767 |   -0.0043 |
| ('simswap', 'ffpp')     |    0.0652 |    1.4208 |          1.3556 |   -0.0049 |
| ('tpsm', 'cdf')         |    0.0542 |    1.3089 |          1.2547 |   -0.0021 |
| ('tpsm', 'ffpp')        |    0.0658 |    1.5278 |          1.462  |   -0.0049 |
| ('wav2lip', 'cdf')      |    0.072  |    1.5618 |          1.4898 |   -0.0038 |
| ('wav2lip', 'ffpp')     |    0.082  |    1.7046 |          1.6226 |   -0.006  |

- `var_reduction>0` ⇒ direction 降低视频内方差(支持 C);若该改善与 mean_logit AUC 提升相关而饱和/Jensen 不能解释,则判 C。

## 8. γ 扫描(真实 b,CDF=oracle)

CDF(oracle)上每个 method × aggregator 的最优 γ(0=direction,1=raw):

| method      |   mean_logit |   mean_prob |
|:------------|-------------:|------------:|
| MRAA        |          0.8 |         0   |
| blendface   |          0.1 |         0.9 |
| e4s         |          0.2 |         0.2 |
| facedancer  |          1   |         0.6 |
| faceswap    |          1   |         0.5 |
| facevid2vid |          0.4 |         0.2 |
| fomm        |          0.4 |         0   |
| fsgan       |          0.9 |         0   |
| inswap      |          0.9 |         0.8 |
| lia         |          0.6 |         0   |
| mobileswap  |          0.3 |         0.7 |
| pirender    |          0   |         0   |
| sadtalker   |          1   |         1   |
| simswap     |          0   |         0.9 |
| tpsm        |          1   |         0.3 |
| wav2lip     |          1   |         0.7 |

- **关键判读**:FSGAN 与 SadTalker 最优 γ 是否相反?最优 γ 是否随聚合器剧变?γ 跨聚合器稳定且偏向 0 ⇒ 支持 B。

## 9. 机制判定(A/B/C/D)

| 机制 | 证据条件 | 本数据 |
|---|---|---|
| A 饱和+概率聚合 | Delta_prob>0但Delta_logit≤0;校准后≈0;raw更饱和 | 池化伪影占比22%;温度后Δ=0.00045283578099725494;饱和差=0.9448917461429829 |
| B 真实径向—角向 | mean_logit 与校准后 direction 仍稳定>0 | logit 下 10/32 为正;温度后 8/32 为正 |
| C 时间一致性 | direction 显著降方差/离群 | 方差下降均值=1.3568447481394523 |
| D 混合方法依赖 | FSGAN/SadTalker 相反最优γ或相反H-B | I_pool FSGAN−SadTalker=0.04184385969603002 |

## 10. 下一步方法建议

(1) 视频级评估必须温度校准(FF++ val 拟合 T)后再聚合,或直接用 mean_logit/median 等无饱和聚合;(2) 因符号方法依赖,不得将 mean_prob 优势解读为方向特征更好;(3) 暂停径向—角向复杂训练;若需自适应,用样本级径向可靠性门控,不得依赖测试时 method label。

## 11. 偏差记录

- FF++ val 用项目 `build_val_loader` 切片(FF++ train videos [586:686]),非官方 140 清单;leak-free,与训练 val 一致(已与用户确认)。
- DFD 全程排除。
- CDF 上 γ_best / T 评估仅作诊断,从未用于挑选聚合器或正式方法。
- bootstrap:video-cluster paired ≥2000;BH 多重比较(在跨方法汇总时应用);帧不作独立样本。
