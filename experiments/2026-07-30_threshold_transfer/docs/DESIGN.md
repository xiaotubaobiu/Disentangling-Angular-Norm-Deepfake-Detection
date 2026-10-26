# DESIGN — cross-domain threshold transfer (2026-07-30)

Detailed protocol, formulas and statistical choices. The README holds only the
operational summary.

## 1. Question

Earlier phases established that **ranking** transfers reasonably well from FF++ to
CDF (video AUC stays high) while the `direction > raw` advantage under
`mean_prob` pooling is largely a saturation artefact (phase 2026-07-27). Ranking
quality is not what a deployed detector uses: it uses a **fixed decision
threshold**. This phase asks whether a threshold chosen honestly on the source
domain still delivers its promised operating point on the target domain, and if
not, why.

Formally, with a frozen decision rule `d(x) = 1[s(x) >= t]`:

1. how much balanced accuracy is lost by using `t` selected on FF++ val instead of
   the CDF-optimal `t*` (which requires target labels and is therefore not
   deployable)?
2. does a source-domain FPR budget (1%, 5%) hold on the target domain?
3. can temperature or Platt calibration repair the loss?
4. is the loss caused by a rigid score translation, a scale change, a
   class-conditional (differential) shift, or by calibration error?

## 2. Scores

Per frame, the classifier's two logits give

```
signed_logit  s = fake_logit - real_logit
p_frame       = softmax(logits)[fake] = sigmoid(s)          (exact identity)
```

`raw` uses the features as trained; `direction` L2-normalises the feature before
the classifier. Both are already materialised per frame by phases 2026-07-21
(target) and 2026-07-27 (source), so this phase runs **no inference** unless a
source cache is missing.

Video score (primary):

```
S_video = mean_t s_t            (aggregator "mean_logit")
```

Video score (baseline, deliberately wrong):

```
S_video^prob = mean_t sigmoid(s_t)   at threshold 0.5
```

`mean_prob` saturates: frames with `|s|` large contribute ~0 or ~1 regardless of
magnitude, so the video score compresses towards 0/1 and its 0.5 threshold is
meaningless across domains. It is reported only to quantify how misleading that
baseline is.

Videos are keyed by `(method_id, video_id)`. In FF++/CDF the same `video_id` is
reused across forgery methods, so `video_id` alone merges different methods'
fakes of one identity — the bug behind the original `direction > raw` claim
(phase 2026-05-05, `reports/final_results_notes.md`). Real frames are replicated
per method in the frame parquet and are de-duplicated with
`study.io.dedup_reals` before aggregation.

**Raw / direction parity.** Both modes come from one wide frame table
(`s_raw`, `s_dir` merged on `image_path`), so they cover identical frames, identical
videos, identical aggregation, identical bootstrap resamples and identical metric
code. `study_threshold.io.assert_modes_share_videos` fails loudly otherwise.

## 3. Calibration (source domain only)

Fitted on FF++ validation:

| Calibrator | Form | Fitted on | Weights |
|---|---|---|---|
| Temperature | `p = sigmoid(s / T)` | per-frame val scores | each video equal, then each class equal |
| Video Platt | `p = sigmoid(A·S + B)`, `A > 0` | per-video val scores | each class equal |

Weighting rationale: FF++ val has ~100 videos per method and 8 frames per video,
with far more fake than real frames. Unweighted NLL would fit the fake-heavy,
long-video part of the distribution.

`A > 0` is enforced so Platt is monotone: it can move the operating point but can
never change AUC. Temperature is a pure rescaling, hence

```
p = sigmoid(s/T) = 0.5   <=>   s = 0     for every T > 0
```

so **the `p = 0.5` threshold is temperature-invariant**. This is the structural
reason temperature scaling cannot fix a threshold shift: it changes probabilities
(and ECE/NLL) but not the decision boundary in score space. Platt's boundary is
`s = -B/A`, which can move.

Reported calibration diagnostics on both domains: ECE (15 equal-width bins), NLL,
Brier, for three probability views (uncalibrated / temperature / Platt).

## 4. Threshold rules (selected on FF++ val video scores)

| Rule | Definition |
|---|---|
| `t_BA` | `argmax_t (TPR(t) + TNR(t)) / 2` |
| `t_equal` | `argmin_t |TPR(t) - TNR(t)|` |
| `t_FPR1` | smallest `t` with source `FPR(t) <= 0.01` |
| `t_FPR5` | smallest `t` with source `FPR(t) <= 0.05` |
| `t_p50_temperature` | `s = 0` (see §3) |
| `t_p50_platt` | `s = -B/A` |

Candidates are midpoints between consecutive unique scores plus outer margins;
ties resolve to the smallest threshold. Decision rule everywhere: predict fake iff
`score >= t`.

`t_BA` and `t_equal` coincide only for symmetric equal-variance classes; both are
reported. A threshold landing near `p = 0.5` is *not* evidence of good
calibration.

## 5. Transfer metrics (target domain)

For every rule, at the frozen `t`:

AUC, balanced accuracy, accuracy, TPR, TNR, FPR, FNR, precision, F1, predicted
fake rate, ECE / NLL / Brier for the three probability views, plus

```
t*_target , BA*_target  = argmax / max balanced accuracy on the target scores  (oracle, diagnostic only)
R_transfer              = BA*_target - BA_target(t_source)             >= 0
threshold_shift         = t*_target - t_source
fixed-FPR error         = FPR_target(t_FPRx_source) - x                for x in {0.01, 0.05}
```

`R_transfer` isolates *threshold* damage from *ranking* damage: it compares two
thresholds on the same target scores, so a model whose AUC dropped is not
penalised twice. The oracle uses target labels and is never used to select
anything.

Scopes: `pooled` (all 16 methods' fakes vs all reals) and `per_method` (one
method's fakes vs all reals, native counts).

## 6. Score-shift decomposition

Per class `c ∈ {real, fake}` and domain: mean, median, std, IQR, q05, q25, q75,
q95. Then

```
shift_c            = mean_target(c) - mean_source(c)
common_shift       = (shift_real + shift_fake) / 2
differential_shift = shift_fake - shift_real
scale_c            = std_target(c) / std_source(c)
separation_d       = mean_d(fake) - mean_d(real)
W_c                = Wasserstein-1(source_c, target_c)
KS_c               = two-sample KS statistic(source_c, target_c)
```

Reading:

* `common_shift` large, `differential_shift ≈ 0`, `scale ≈ 1` → rigid translation;
  ranking intact, a single re-centring would recover the source operating point.
* `differential_shift` large → classes move relative to each other; no single
  offset recovers both TPR and TNR.
* `scale_c` far from 1 → overlap changes shape; fixed-FPR quantiles move even when
  means match.
* small `R_transfer` with large ECE → calibration is bad but the *decision* still
  transfers (the two questions are separate).

## 7. Uncertainty

Video-cluster bootstrap: resample whole videos (`video_key = method_id|video_id`)
with replacement, `bootstrap_iterations = 2000`, `bootstrap_seed = 20260730`,
percentile 95% CI. Raw and direction are bootstrapped from the same wide table so
their difference CI is **paired**.

Implementation reuses `study.cluster_bootstrap.cluster_bootstrap_metric`; a fast
path (one row per cluster ⇒ resample rows) is used for the video tables and is
asserted to select identical clusters for a given seed
(`tests/test_bootstrap_equivalence.py`).

Bootstrapped quantities: `BA_target(t_BA)`, `R_transfer(t_BA)`, `FPR_target(t_FPR5)`
and the paired direction − raw differences of the first two.

## 8. Leakage rules (enforced, not just documented)

* CDF never enters a fit, a threshold search (other than the diagnostic oracle) or
  a model/seed selection step. `02_fit_calibrators.py` and
  `03_select_source_thresholds.py` load the source cache only; they physically
  cannot see the target parquet.
* DFD is excluded everywhere (`domains.exclude`).
* No target-domain quantity is used to choose among the 4 threshold rules; all
  rules are reported for all cells.
* Tests `test_no_target_domain_in_fit` / `test_threshold_selection_source_only`
  assert the first bullet by monkeypatching the target loader to raise.

## 9. Known limitations

1. FF++ val comes from the phase-2026-07-27 cache: 100 videos per method × 8
   frames, i.e. a *subsample* of FF++ val, and its frame count per video (8) is
   smaller than the target-domain videos (32 frames). Video scores are means, so
   they are comparable, but the source video score is noisier.
2. Only `raw`-trained checkpoints have both `raw` and `direction` views;
   `normtrain` checkpoints are out of scope.
3. `normtrain_norm`, `length` inference modes are out of scope.
4. CDF real videos (178 per checkpoint) are far fewer than fakes; pooled FPR
   estimates therefore have coarse granularity (~0.6% per real video), which
   bounds how tightly a 1% budget can be verified.
5. The oracle threshold is computed on the same target sample it is evaluated on,
   so `R_transfer` is slightly optimistic about the oracle (no target-side
   train/test split).
