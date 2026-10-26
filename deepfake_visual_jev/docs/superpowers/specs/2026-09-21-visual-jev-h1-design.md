# Visual-Jev H1 Experimental Design

**Date:** 2026-09-21

## Objective

Test H1 only: replacing a fixed CLIP binary classification head with a Jev-style,
semantic candidate-conditioned shared scorer does not reduce basic deepfake
detection performance by more than one percentage point of video-level AUROC.

This experiment does not test generative VLM efficiency, open-ended question
answering, reinforcement learning, or the undisclosed official Jev RLCD recipe.

## Locked baseline A

Select A using validation performance only. Among the available CLIP raw
checkpoints, the selected checkpoint is:

`experiments/2026-05-05_paper_feature_geometry/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth`

It has the highest saved validation AUROC of the eligible CLIP raw runs
(0.996712109375). The original model is CLIP ViT-B/16 plus a learned `Linear(d, 2)`
head. Target-domain results were not used for selection.

Because the saved A was fully fine-tuned, B reuses A's final visual encoder and
freezes it. This holds the learned visual representation constant and isolates
the replacement of the decision head. The original A checkpoint and head remain
unchanged.

## Locked model B

Model B is the minimal Visual-Jev described in the user-provided research plan.
It is not a Qwen/NanoJev reproduction.

### Fixed inputs

- Question: `Which description best matches this facial image?`
- Candidate 0: `an authentic facial image.`
- Candidate 1: `a manipulated deepfake facial image.`

Each CLIP text input is formatted as:

`Question: {question} Candidate: {candidate}`

The text embeddings are computed once with the frozen CLIP ViT-B/16 text
encoder. The image embedding is produced by A's frozen, fine-tuned CLIP visual
encoder. To preserve A's selected `raw` representation, neither image nor text
embeddings receive an extra L2-normalization step before the shared scorer.

### Shared candidate scorer

For candidate `i`:

`u_i = [v; t_i; v * t_i; abs(v - t_i)]`

`z_i = shared_mlp(u_i)`

The MLP is exactly `4d -> 512 -> 128 -> 1`, with GELU between linear layers.
Both candidates are scored by the same module. There are no candidate-specific
heads or candidate-position parameters.

The two logits are normalized with softmax. Training uses ordinary two-class
cross entropy over the complete candidate distribution. Only the shared scorer
is trainable; both CLIP encoders and the candidate embeddings are frozen.

## Data and optimization

- Dataset root: `/Dataset/deepfake_detection/DF40_all`
- Same train/validation/test indexes, frame sampling, crop, normalization, and
  augmentation as A's resolved configuration
- Train methods: all 16 configured DF40 methods
- 586 training videos per method, 100 validation videos per method, 8 frames per
  video
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-2`
- Scheduler: one warmup epoch followed by cosine decay
- Maximum epochs: 15
- Early stopping patience: 5
- Checkpoint selection: validation video-level AUROC only
- Primary seed: 42; seeds 7 and 2048 are replication runs
- GPUs: physical devices 0, 1, and 2 for the three seeds; device 3 is reserved
  for baseline evaluation or recovery. Devices 4-7 must not be used.

No target-domain result may influence training, checkpoint selection,
hyperparameters, or model choice.

## Evaluation correctness

For video data, frame probabilities are averaged inside a semantic video key:

- real: `real::{video_id}`
- fake: `{manipulation_method}::{video_id}`

Using `video_id` alone is invalid because IDs repeat across manipulation methods.
Repeated real records are deduplicated before evaluation.

Report for FF++ test and Celeb-DF-v2:

- video-level AUROC (primary)
- frame-level AUROC
- average precision
- Brier score
- accuracy at 0.5 (reference only)
- trainable and total parameter counts

Also report paired video-bootstrap confidence intervals for the B-minus-A AUROC
difference where the evaluated video sets match.

## Required correctness test

When the candidate order is reversed, the two output logits/probabilities must
reverse exactly (within floating-point tolerance). With the target index reversed
at the same time, the loss must remain unchanged. This verifies that predictions
follow candidates rather than fixed output positions.

## Pre-registered conclusion rule

Primary H1 gate:

`AUC_B(FF++ video) - AUC_A(FF++ video) >= -0.01`

- **H1 supported:** the point estimate passes the gate. The bootstrap interval is
  reported to express uncertainty, not to redefine the gate.
- **H1 not supported:** the point estimate is below `-0.01` after correctness and
  convergence checks pass.
- **Evidence insufficient:** the run is invalid (data mismatch, failed order
  check, non-convergence, incomplete evaluation, or irreproducible failure).

Celeb-DF-v2 is a pre-registered external-domain secondary result. It can support
or qualify the interpretation but does not replace the primary FF++ gate.

## Scope of claims

A passing result supports only this statement: on the available data and fixed
CLIP representation, a semantic candidate-conditioned shared scorer can replace
the fixed binary head without more than a one-point loss in in-domain video
AUROC. It does not establish that Visual-Jev is generally more accurate, faster,
more robust, generative, or equivalent to official Jev.
