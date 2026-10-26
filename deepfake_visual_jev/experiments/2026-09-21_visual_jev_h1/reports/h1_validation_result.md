# Visual-Jev H1 Validation Result

**Date:** 2026-09-21  
**Verdict:** **H1 supported on the available data**

## Claim tested

Replacing the selected CLIP model's fixed `Linear(d, 2)` output head with the
approved Jev-style semantic candidate shared scorer does not reduce in-domain
video-level AUROC by more than one percentage point.

The pre-registered primary rule was:

`AUROC_B(FF++ video) - AUROC_A(FF++ video) >= -0.01`

The observed difference was `+0.0000614`, so the rule passes. The paired
video-bootstrap 95% interval was `[-0.0001547, +0.0003143]`; it is narrow and
entirely far above the `-0.01` non-inferiority margin.

This supports H1 only. It does not establish H2, H3, open-ended VLM capability,
RL benefit, or equivalence to the undisclosed official Jev training recipe.

## Locked models

### A: selected existing baseline

- CLIP ViT-B/16 plus fixed two-output linear head
- Raw-feature cohort, seed 42
- Selected only because it had the highest saved validation AUROC among the
  eligible raw CLIP checkpoints (`0.9967121`)
- Checkpoint:
  `experiments/2026-05-05_paper_feature_geometry/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth`

### B: minimal Visual-Jev

- A's fine-tuned CLIP visual encoder, frozen
- Frozen CLIP text embeddings for the approved real/fake candidates
- One shared scorer over `[v; t; v*t; abs(v-t)]`
- MLP dimensions `2048 -> 512 -> 128 -> 1`
- Two candidate logits trained with cross entropy
- Primary seed 42 checkpoint selected at epoch 2 by validation video AUROC
- Checkpoint:
  `experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth`

## Primary paired results

All video probabilities are means of frame-level `P(fake)`. Rows are first
deduplicated by sample identity; videos use the semantic key
`method_id::video_id`, so identically numbered fakes from different methods are
not merged.

| Domain | Videos (real/fake) | A video AUROC | B video AUROC | B - A | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| FF++ test | 3,049 (140/2,909) | 0.998490 | 0.998551 | +0.000061 | [-0.000155, +0.000314] |
| Celeb-DF-v2 | 9,901 (178/9,723) | 0.901446 | 0.913479 | +0.012033 | [+0.005716, +0.018395] |

The FF++ point estimate passes the `-0.01` primary gate. Celeb-DF was a
pre-registered external-domain secondary result and also favors B in this run.

## Auxiliary metrics for the primary seed

| Domain | Model | Frame AUROC | Video AP | Video Brier ↓ | Video ACC@0.5 |
|---|---|---:|---:|---:|---:|
| FF++ | A | 0.995539 | 0.999928 | **0.020334** | 0.973762 |
| FF++ | B | **0.995857** | **0.999930** | 0.041170 | **0.989833** |
| Celeb-DF-v2 | A | 0.878546 | 0.997865 | **0.016596** | 0.981113 |
| Celeb-DF-v2 | B | **0.883592** | **0.998162** | 0.041010 | **0.982224** |

The ranking metrics support H1, but B's Brier score is substantially worse on
both domains. Therefore the result must not be described as improved probability
calibration. The shared scorer would require a separately pre-registered
calibration intervention if calibrated probabilities become a goal.

## Three-seed replication for B

All three runs reused the same locked A visual encoder and varied the scorer/data
random seed. Checkpoints were selected only by validation video AUROC.

| Seed | Best epoch | Validation AUROC | FF++ video AUROC | Celeb-DF video AUROC |
|---:|---:|---:|---:|---:|
| 42 | 2 | 0.996724 | 0.998551 | 0.913479 |
| 7 | 1 | 0.996701 | 0.998539 | 0.909243 |
| 2048 | 1 | 0.996723 | 0.998502 | 0.911080 |
| Mean ± sample SD | — | 0.996716 ± 0.000013 | 0.998531 ± 0.000026 | 0.911267 ± 0.002124 |

Every B seed is within the one-point FF++ margin relative to A. The very small
FF++ spread shows that the pass is not driven by one lucky scorer initialization.

## Correctness and resource checks

- Candidate order reversal exactly reversed the two logits.
- Reversing the target index at the same time changed cross entropy by exactly
  `0.0` in the saved-checkpoint check.
- The saved-model permutation error was exactly `0.0` on both domains.
- A has 86,193,666 total/originally trainable parameters.
- B has 87,307,521 runtime parameters, of which 1,114,881 are trainable in this
  stage. Fixed text embeddings are cached, so the frozen CLIP text tower is not
  stored in the runtime checkpoint.
- Full repository test suite: `110 passed`.

## Why the A numbers differ from older result files

The historical A metadata reported FF++ `0.999749` and Celeb-DF `0.832808`.
Those values grouped by bare `video_id`, which can merge distinct manipulation
methods, and were therefore not used for the final comparison. Re-aggregating
the existing per-frame cache with sample deduplication and
`method_id::video_id` produced the valid A values used above (`0.998490` and
`0.901446`). No A weights or frame scores were changed.

## Interpretation

On the available DF40/FF++ training setup, a semantic candidate-conditioned
shared scorer can replace the fixed binary output head while preserving
in-domain deepfake ranking performance within the pre-defined tolerance. It also
improved Celeb-DF video AUROC in this experiment. This is evidence that the
minimal Visual-Jev decision form is feasible for binary deepfake detection.

The evidence does not yet show that the candidate text is useful beyond acting
as a fixed conditioning code, because only one question and two fixed candidate
sentences were tested. It also does not show that B is better calibrated; the
Brier results show the opposite.

## Limitations

1. B inherits A's already fine-tuned visual encoder. This cleanly tests head
   replacement on a fixed representation, but is not an independent end-to-end
   retraining comparison from the original CLIP initialization.
2. Replication varies only B's scorer/data seed; it does not include multiple A
   visual seeds.
3. Video sets are strongly class-imbalanced, so AUROC is primary and accuracy/AP
   must be interpreted cautiously.
4. Candidate paraphrases, unseen candidate sets, efficiency/latency, and RL were
   outside this H1 experiment.
5. This model is a Visual-Jev adaptation of the published decision interface,
   not an official NanoJev/Jev reproduction.

## Reproducible artifacts

- Experimental design: `docs/superpowers/specs/2026-09-21-visual-jev-h1-design.md`
- Implementation plan: `docs/superpowers/plans/2026-09-21-visual-jev-h1-implementation.md`
- Baseline ranking: `experiments/2026-09-21_visual_jev_h1/baseline_selection.csv`
- Per-seed results: `experiments/2026-09-21_visual_jev_h1/results/summary/seed_results.csv`
- Seed summary: `experiments/2026-09-21_visual_jev_h1/results/summary/seed_summary.json`
- Paired comparison and verdict: `experiments/2026-09-21_visual_jev_h1/results/summary/paired_comparison.json`
- A evaluation: `experiments/2026-09-21_visual_jev_h1/results/evaluation/baseline_a/evaluation.json`
- B evaluation: `experiments/2026-09-21_visual_jev_h1/results/evaluation/visual_jev_b_seed42/evaluation.json`
- A/B video probabilities: the corresponding `*_video_predictions.csv` files
  under the two evaluation directories
