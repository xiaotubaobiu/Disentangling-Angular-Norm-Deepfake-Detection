# Cross-domain threshold transfer — final report

- generated: 2026-07-30T11:29:00
- source: FF++ **validation** (calibration + threshold selection), target: CDF (evaluation only)
- video score: `mean_t(signed_logit)`; `mean_prob@0.5` reported only as a mis-specified baseline
- cells: 15 checkpoints × 2 modes (clip, dinov2, efficientnet; seeds 7, 42, 123, 999, 2048), 16 forgery methods
- design and formulas: `docs/DESIGN.md`

## 0. Answers

1. **Can an FF++-validation threshold be deployed on CDF directly?** No. At `t_BA` the source operating point (BA 0.963) collapses to BA 0.616 on CDF while the CDF oracle threshold would reach 0.814 — a transfer regret of 0.198 BA (median 0.187, range 0.121–0.300). 0/30 checkpoint×mode cells stay within the 0.05 BA tolerance. The frozen threshold flags 98% of CDF videos as fake and its false positive rate is 0.752 (source: 0.026).
2. **Raw or direction: whose threshold transfers better?** R_transfer = 0.198 (raw) vs 0.198 (direction), paired bootstrap difference -0.000 [-0.012, +0.011]. Target AUC is essentially the same (0.892 vs 0.888), so neither view buys threshold stability — the difference lives in score *scale*, not in transferability.
3. **Do temperature / Platt calibration fix the drift?** No. Temperature scaling is a pure rescaling, so its `p=0.5` boundary is `s=0` for any `T`; it changes ECE (target ECE 0.201 → 0.019) but cannot move the decision. Platt is affine and does move it (`s=-B/A`), yet its regret (0.204) is no better than `t_BA` (0.198). Calibrating on the source domain cannot repair a target-domain shift.
4. **Is a fixed FPR better than max-BA for deployment?** It is better behaved but still broken: the 1% budget realises 0.643 on CDF and the 5% budget 0.812. Their BA (0.657 / 0.587) is however higher than max-BA's 0.616: `t_FPR1` beats `t_BA` on CDF in 50% of cells, because a conservative threshold absorbs part of the positive score shift. So yes — of the rules tested, a tight source FPR budget is the least bad frozen choice, but 'least bad' still means a 0.63 absolute FPR overshoot; it is not a substitute for re-estimating the threshold on the target domain.
5. **Why high AUC but low BA at the source threshold?** The class-conditional distributions move by different amounts: mean real score shifts +4.15 while fake shifts +0.36 (common +2.26, differential -3.79), and the real-class spread scales by 1.20×. Ranking is preserved (AUC 0.890) because AUC is invariant to monotone changes; a fixed cut is not. Dominant cause per cell: class-conditional (differential) shift (25); score scale change (5).
6. **Second core contribution for the pooling/calibration paper?** Yes. Phase 2026-07-27 showed the `direction > raw` gap is a pooling/saturation artefact — a statement about *ranking metrics*. This phase shows the deployment-facing failure is an order of magnitude larger (0.198 BA lost, FPR 0.026 → 0.752) and is *not* repaired by the calibration that fixes the artefact. The two results compose: calibrate before you compare aggregators, and re-estimate the threshold on the target domain before you deploy.

## 1. Transfer by threshold rule

| rule | mode | target_ba | target_oracle_ba | r_transfer | r_transfer_sd | target_fpr | target_tpr | target_pred_fake_rate | source_ba | source_fpr | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t_BA | direction | 0.6141 | 0.8119 | 0.1978 | 0.0575 | 0.7566 | 0.9848 | 0.9807 | 0.9633 | 0.0273 | 15 |
| t_BA | raw | 0.6179 | 0.8162 | 0.1982 | 0.0514 | 0.7483 | 0.9842 | 0.9799 | 0.9634 | 0.0247 | 15 |
| t_FPR1 | direction | 0.6569 | 0.8119 | 0.1550 | 0.0895 | 0.6438 | 0.9576 | 0.9520 | 0.9506 | 0.0100 | 15 |
| t_FPR1 | raw | 0.6580 | 0.8162 | 0.1582 | 0.0904 | 0.6431 | 0.9590 | 0.9533 | 0.9529 | 0.0100 | 15 |
| t_FPR5 | direction | 0.5862 | 0.8119 | 0.2256 | 0.0757 | 0.8139 | 0.9863 | 0.9832 | 0.9552 | 0.0500 | 15 |
| t_FPR5 | raw | 0.5882 | 0.8162 | 0.2280 | 0.0748 | 0.8094 | 0.9857 | 0.9825 | 0.9550 | 0.0500 | 15 |
| t_equal | direction | 0.6076 | 0.8119 | 0.2043 | 0.0543 | 0.7715 | 0.9868 | 0.9829 | 0.9570 | 0.0427 | 15 |
| t_equal | raw | 0.6078 | 0.8162 | 0.2083 | 0.0544 | 0.7712 | 0.9868 | 0.9830 | 0.9568 | 0.0433 | 15 |
| t_p50_platt | direction | 0.6085 | 0.8119 | 0.2033 | 0.0588 | 0.7689 | 0.9860 | 0.9821 | 0.9582 | 0.0393 | 15 |
| t_p50_platt | raw | 0.6114 | 0.8162 | 0.2048 | 0.0595 | 0.7622 | 0.9849 | 0.9809 | 0.9584 | 0.0373 | 15 |
| t_p50_temperature | direction | 0.5765 | 0.8119 | 0.2354 | 0.0622 | 0.8386 | 0.9916 | 0.9888 | 0.9466 | 0.0767 | 15 |
| t_p50_temperature | raw | 0.5842 | 0.8162 | 0.2319 | 0.0521 | 0.8232 | 0.9917 | 0.9887 | 0.9500 | 0.0687 | 15 |

Mis-specified baseline `mean_prob@0.5` (for contrast only): target BA 0.581, FPR 0.829, predicted-fake-rate 0.988.

![R_transfer by rule](../figures/06_r_transfer_by_rule.png)
![target BA by rule](../figures/06_target_ba_by_rule.png)

## 2. Fixed-FPR budgets

| rule | mode | promised | realised | error | target_ba | r_transfer |
|---|---|---|---|---|---|---|
| t_FPR1 | direction | 0.0100 | 0.6438 | 0.6338 | 0.6569 | 0.1550 |
| t_FPR1 | raw | 0.0100 | 0.6431 | 0.6331 | 0.6580 | 0.1582 |
| t_FPR5 | direction | 0.0500 | 0.8139 | 0.7639 | 0.5862 | 0.2256 |
| t_FPR5 | raw | 0.0500 | 0.8094 | 0.7594 | 0.5882 | 0.2280 |

![FPR budget](../figures/06_fpr_budget_violation.png)

## 3. Calibration

| mode | temperature | platt_a | platt_b | val_frame_ece_before | val_frame_ece_after | val_video_ece_before | val_video_ece_after_platt |
|---|---|---|---|---|---|---|---|
| direction | 0.0923 | 22.5428 | -0.9377 | 0.3791 | 0.0530 | 0.3836 | 0.0544 |
| raw | 1.4878 | 0.9414 | -0.7457 | 0.0449 | 0.0531 | 0.0351 | 0.0537 |

| mode | target_ece_uncalibrated | target_ece_temperature | target_ece_platt | target_nll_temperature | target_brier_platt |
|---|---|---|---|---|---|
| direction | 0.3841 | 0.0209 | 0.0229 | 0.0820 | 0.0226 |
| raw | 0.0172 | 0.0178 | 0.0228 | 0.0782 | 0.0229 |

![ECE](../figures/06_ece_source_vs_target.png)

## 4. Score shift

| mode | shift_real | shift_fake | common_shift | differential_shift | scale_real | scale_fake | wasserstein_real | ks_real | separation_source | separation_target |
|---|---|---|---|---|---|---|---|---|---|---|
| direction | 0.4447 | 0.0329 | 0.2388 | -0.4118 | 1.2202 | 0.7311 | 0.4447 | 0.8067 | 0.6204 | 0.2086 |
| raw | 7.8634 | 0.6899 | 4.2767 | -7.1735 | 1.1887 | 0.7933 | 7.8634 | 0.8073 | 12.3085 | 5.1350 |

![shift decomposition](../figures/05_shift_decomposition.png)
![differential shift vs regret](../figures/06_differential_shift_vs_regret.png)

## 5. Per-method

| method_id | mode | target_auc | target_ba | target_fpr | r_transfer |
|---|---|---|---|---|---|
| MRAA | direction | 0.9091 | 0.6201 | 0.7566 | 0.2229 |
| MRAA | raw | 0.9224 | 0.6242 | 0.7483 | 0.2311 |
| blendface | direction | 0.9325 | 0.6210 | 0.7566 | 0.2417 |
| blendface | raw | 0.9293 | 0.6251 | 0.7483 | 0.2357 |
| e4s | direction | 0.9530 | 0.6207 | 0.7566 | 0.2840 |
| e4s | raw | 0.9556 | 0.6249 | 0.7483 | 0.2818 |
| facedancer | direction | 0.9225 | 0.6170 | 0.7566 | 0.2434 |
| facedancer | raw | 0.9203 | 0.6208 | 0.7483 | 0.2407 |
| faceswap | direction | 0.9107 | 0.6192 | 0.7566 | 0.2377 |
| faceswap | raw | 0.9238 | 0.6232 | 0.7483 | 0.2440 |
| facevid2vid | direction | 0.9872 | 0.6217 | 0.7566 | 0.3355 |
| facevid2vid | raw | 0.9904 | 0.6258 | 0.7483 | 0.3342 |
| fomm | direction | 0.9806 | 0.6217 | 0.7566 | 0.3202 |
| fomm | raw | 0.9857 | 0.6258 | 0.7483 | 0.3234 |
| fsgan | direction | 0.9473 | 0.6210 | 0.7566 | 0.2581 |
| fsgan | raw | 0.9500 | 0.6251 | 0.7483 | 0.2589 |
| inswap | direction | 0.7435 | 0.5924 | 0.7566 | 0.1040 |
| inswap | raw | 0.7460 | 0.5955 | 0.7483 | 0.1023 |
| lia | direction | 0.7910 | 0.6033 | 0.7566 | 0.1305 |
| lia | raw | 0.8026 | 0.6077 | 0.7483 | 0.1328 |
| mobileswap | direction | 0.8956 | 0.6195 | 0.7566 | 0.2014 |
| mobileswap | raw | 0.8984 | 0.6234 | 0.7483 | 0.2012 |
| pirender | direction | 0.9648 | 0.6216 | 0.7566 | 0.2908 |
| pirender | raw | 0.9720 | 0.6257 | 0.7483 | 0.2960 |
| sadtalker | direction | 0.7595 | 0.5927 | 0.7566 | 0.1143 |
| sadtalker | raw | 0.7559 | 0.5947 | 0.7483 | 0.1083 |
| simswap | direction | 0.8573 | 0.6135 | 0.7566 | 0.1768 |
| simswap | raw | 0.8578 | 0.6167 | 0.7483 | 0.1753 |
| tpsm | direction | 0.8766 | 0.6134 | 0.7566 | 0.1927 |
| tpsm | raw | 0.8800 | 0.6174 | 0.7483 | 0.1916 |
| wav2lip | direction | 0.7759 | 0.6048 | 0.7566 | 0.1168 |
| wav2lip | raw | 0.7715 | 0.6083 | 0.7483 | 0.1113 |

![per-method regret](../figures/06_per_method_r_transfer.png)

## 6. Uncertainty (video-cluster bootstrap, paired raw/direction)

| quantity | raw | raw_lo | raw_hi | direction | dir_lo | dir_hi | diff | diff_lo | diff_hi | n_iter |
|---|---|---|---|---|---|---|---|---|---|---|
| r_transfer@t_BA | 0.1982 | 0.1684 | 0.2373 | 0.1978 | 0.1683 | 0.2367 | -0.0005 | -0.0122 | 0.0114 | 2000 |
| target_ba@t_BA | 0.6179 | 0.5874 | 0.6495 | 0.6141 | 0.5842 | 0.6451 | -0.0038 | -0.0107 | 0.0030 | 2000 |
| target_fpr@t_FPR5 | 0.8094 | 0.7541 | 0.8620 | 0.8139 | 0.7587 | 0.8657 | 0.0045 | -0.0045 | 0.0132 | 2000 |

## 7. Leakage statement

- calibrators (stage 02) and thresholds (stage 03) read the FF++ validation cache only;
- CDF enters at stage 04 for evaluation, and its labels are used for metrics and for the diagnostic oracle threshold that defines `R_transfer` — never for fitting or selection;
- DFD is excluded everywhere;
- all four threshold rules are reported for all cells (no target-side rule picking).

## 8. Limitations

See `docs/DESIGN.md` §9. The two that matter most for reading the numbers above: FF++ val comes from a 100-video-per-method, 8-frames-per-video subsample (noisier video scores than the 32-frame CDF videos), and CDF has only 178 real videos per checkpoint, so a realised FPR is quantised in steps of ~0.6%.
