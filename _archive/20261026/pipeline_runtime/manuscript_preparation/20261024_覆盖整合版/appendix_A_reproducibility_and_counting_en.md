# Appendix A: reproducibility protocol and counting conventions

*Preparatory journal supplement. This appendix describes frozen protocols and artifact identities, without performance results from unfinished models. Final successful-update totals, complete result tables and public archival information still require verification after completion.*

## A.1 Videos, canonical RGB inputs and draw budgets

A natural-video identity includes dataset, domain, method and video ID. Real videos use the method label real and occur once in the negative pool. Source groups encode known or proxy ancestry and are not interchangeable with video UIDs or person identities. Exact-RGB links constrain observed content duplication. Eight-frame natural videos used for scoring and canonical RGB training inputs are different levels of description.

D TRAIN contains 611 real and 12,232 fake videos: 12,843 videos and 102,744 eight-frame associations. Merging 519 redundant associations within 493 cross-method RGB groups yields 102,225 canonical RGB inputs while retaining the sampling mass contributed by each method. Missing real source 281 explains 611 rather than the nominal 612; images are not replicated to fill this gap. Each epoch samples with replacement using the frozen masses and rounds the total draw budget up to a multiple of 128, giving 102,272 draws. The extra 47 is a budget-rounding difference, not a fixed set of appended images or additional unique RGB inputs.

The two H training histories share 3,053 common-eligible natural videos and 24,424 logical mother-frame slots. B and AB are two histories of the same mother frame; versions and support masks do not create additional natural videos. The per-model replacement draw budget is 24,448, with the extra 24 again arising from rounding. Failed eligibility is not repaired with another history's image or an unselected frame. SOURCE_UNUSED contributes to the complete relationship/RGB graph and never enters training, development, calibration or testing.

## A.2 Attempts, successful updates and precision

The effective batch is fixed at 128. D and H both use a microbatch of 32 with gradient accumulation to maintain an effective batch of 128. Each of the three H backbones passed source-domain GPU qualification at microbatch 32 under both AMP and FP32, giving six qualification runs. This choice comes from actual qualification runs rather than CPU tests or a data freeze. D and H have 799 and 191 optimizer attempts per epoch, respectively, or 23,970 and 5,730 over the base 30 epochs. Any registered extension uses its actual frozen epoch count.

Let A denote attempts and S AMP skips. Successful optimizer updates satisfy U=A−S. The learning-rate clock advances by attempts, whereas Adam state advances only on successful updates. Finite-loss, nonfinite-gradient events follow the frozen scaler and stopping rules. Events, subsequent recovery and cumulative counts are retained without makeup steps, seed changes, threshold reselection or test-pool changes. Final tables report A, U and S separately; equal epoch counts do not prove equal successful updates. If H triggers its registered uniform FP32 fallback, only effective registered paired runs can supply final caches; retained AMP artifacts are not substituted.

## A.3 Artifact identity and independent recomputation

Each run preserves source TRAIN/DEV manifest hashes, configuration, initialization weights, environment lock, qualification and training-code identities, together with seeds, epoch-level draw/augmentation records, optimizer, RNG, scaler and recovery state. The primary endpoint is the frozen last epoch. Source-best weights are retained separately for registered sensitivity analyses and cannot replace the primary endpoint according to target outcomes.

Seven formal inference-role caches bind their own manifests and the same original FP32 head. Head rows and biases are promoted to FP64 before subtraction. Ranking follows the registered precision rules, while operating decisions remain in the fixed FP64 threshold space. Saved scores are reread for independent AUC and integer operating-count recomputation. CDF-full differs from the fixed TEST used by calibration analyses, and CAL scores never enter TEST AUC. H all-B and common cohorts retain separate estimands and threshold provenance.

Five training seeds, 200 calibration subsets and 2,000 test bootstrap draws describe distinct variation sources. Unresolved ordering, missing methods, unavailable budgets and inadequate connected closures retain NA and its cause. Unfavorable seeds are not dropped before aggregation. A machine-readable asset index records manifests and version identities. These identities support recomputation without certifying unrecorded generative ancestry or a population false-positive guarantee.
