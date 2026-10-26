# T1-H：冻结预处理覆盖统计 / Frozen preprocessing coverage

此表来自修复后的正式元数据入口，已重放完整内容图并逐行核对13张冻结表；未重新检测、解码或审计PNG/mask。它报告输入覆盖，不是模型性能或H GPU资格。原失败记录、独立代码复审和324项CPU回归均保留。

This table comes from the repaired production metadata entry point, with complete content-graph replay and exact validation of thirteen frozen tables. It does not re-detect faces, decode images or repeat PNG/mask audits. It reports input coverage, not detector performance or H GPU qualification. Original failure evidence, independent review and the 324-test CPU regression are retained.

| Dataset / Role | 候选视频 / Candidates | B处理合格 / B eligible | B/AB共同合格 / Common eligible |
| --- | ---: | ---: | ---: |
| DFD / TARGET_TEST | 3431 | 3320 | 3090 |
| FFPP / DEV | 540 | 540 | 535 |
| FFPP / SOURCE_CAL | 140 | 140 | 140 |
| FFPP / SOURCE_TEST | 700 | 700 | 696 |
| FFPP / SOURCE_UNUSED | 560 | 557 | 557 |
| FFPP / TRAIN | 3060 | 3060 | 3053 |

| DFD类别 / Class | 候选 / Candidates | B合格 / B eligible | 共同合格 / Common eligible |
| --- | ---: | ---: | ---: |
| DeepFakeDetection | 3068 | 3027 | 2831 |
| real | 363 | 293 | 259 |

TRAIN/DEV实际使用共同合格视频，不能把B单版本的处理成功数当作训练暴露数。SOURCE_UNUSED的560个视频全部行政排除，即使其中557个处理合格；它们只参与完整关系/RGB图，不参与训练、调参、校准或测试。DFD的all-B部署队列为3320/3431（96.7648%），共同队列为3090/3431（90.0612%）；共同队列保留3090/3320（93.0723%）的all-B视频。按类别的覆盖差异须保留，不能从覆盖率推断检测准确率，不能把两队列效应相减。

TRAIN and DEV use common-eligible videos; B-only processing counts are not training exposure counts. All 560 SOURCE_UNUSED videos are administratively excluded even though 557 are processing-eligible. They contribute only relationship/RGB graph edges. DFD all-B coverage is 3320/3431 (96.7648%); common coverage is 3090/3431 (90.0612%), retaining 3090/3320 (93.0723%) of all-B videos. Preserve class-specific coverage, without inferring detection accuracy or subtracting effects across cohorts.

全部8431候选的母帧槽位为67448，实际选出67400；6个视频因不足8个不同母帧留下48个未选槽位。A/B版本各114个视频不合格（6个母帧资格失败、108个含无合格人脸），AB版本360个不合格（6个母帧资格失败、354个含无合格人脸）。这些全候选计数包括SOURCE_UNUSED；帧失败、视频失败和不同原因不得混作相同分母。

Across all 8431 candidates, 67400 of 67448 prescribed mother slots are selected; six videos lacking eight distinct mothers account for 48 unselected slots. A and B each have 114 ineligible videos (six mother-selection failures and 108 with no-qualified-face failures); AB has 360 (six and 354). These all-candidate totals include SOURCE_UNUSED. Frame failures, failed videos and failure categories have different denominators.

原始检测与A后重检测分别有67400和67248个应有检测槽位，记录有效检测67248和66745次；因已记录失败而缺失152和503次，无失败证据的未知缺失及无效记录均为0。全候选统计中的多脸帧比例为10922/67248（16.2414%）与146/66745（0.2187%），以有效已记录检测为分母；这不是所有人脸被操纵身份的认证。padding按版本、合格/不合格/可用范围分别保存在完整JSON中，不能把frame-weighted均值当成自然视频等权均值。

Original and A-redetection stages have 67400 and 67248 expected detection slots, with 67248 and 66745 valid recorded detections. The 152 and 503 missing records follow documented failures; unknown missing records and invalid records are zero. All-candidate multiface-frame rates are 10922/67248 (16.2414%) and 146/66745 (0.2187%), conditional on valid recorded detections. This does not certify manipulated-face identity. Version- and eligibility-specific padding statistics remain in the full JSON; frame-weighted means are not natural-video-equal-weight means.
