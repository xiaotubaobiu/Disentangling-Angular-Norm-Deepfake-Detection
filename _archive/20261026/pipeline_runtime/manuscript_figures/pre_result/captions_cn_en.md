# Pre-result figure captions / 结果前图件图注

**H preprocessing coverage.** Bars show B and joint B/AB processing eligibility among all frozen candidate natural videos, by role and DFD class. Labels are numerator/denominator counts. All SOURCE_UNUSED candidates are administratively excluded from training, calibration and testing even when processing-eligible. TRAIN/DEV exposure uses the common cohort. This is input coverage, without detector accuracy or H GPU-qualification claims. Overlapping failure categories are not added.

**H预处理覆盖率。** 按冻结角色及DFD类别报告B和B/AB共同处理合格率，柱内为自然视频计数的分子/分母。SOURCE_UNUSED即使处理合格仍全部行政排除，不参与训练、校准或测试；TRAIN/DEV实际使用共同队列。图示不是检测准确率或H的GPU资格。不同失败原因不相加。

**D source development curves.** Every registered development run is shown: two seeds per fixed learning-rate candidate. Points are independently accepted source DEV method-macro AUC; shared displayed limits retain every observed point. Each epoch has 799 attempted updates and 102272 sampled inputs. Successes and AMP skips are retained separately in plot data. Source DEV participates in the frozen recipe selection, and does not estimate formal TEST performance or training-seed confidence intervals.

**D源域开发曲线。** 完整显示各登记学习率候选的两个开发种子；每点为已独立验收的源DEV方法宏AUC，三面板使用共同纵轴范围且保留全部观测点。每轮799次更新尝试和102272次输入抽样，成功更新及AMP跳步另存。源DEV参与既定配方选择，不用于推断正式TEST性能或训练种子置信区间。
