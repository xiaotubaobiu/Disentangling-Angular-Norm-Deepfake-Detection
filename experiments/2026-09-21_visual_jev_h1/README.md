# Visual-Jev H1 validation

This experiment tests whether the fixed `Linear(d, 2)` head of the selected
fine-tuned CLIP baseline can be replaced by the semantic candidate-conditioned
shared scorer specified in
[`docs/superpowers/specs/2026-09-21-visual-jev-h1-design.md`](../../docs/superpowers/specs/2026-09-21-visual-jev-h1-design.md).

## Locked baseline

- Model: CLIP ViT-B/16, raw features, seed 42
- Selection: highest validation AUROC among eligible CLIP raw checkpoints
- Checkpoint: `experiments/2026-05-05_paper_feature_geometry/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth`

The eligibility set is the raw-feature CLIP cohort because the approved H1
equations feed `v = E_v(I)` directly to both heads. The separately trained
`normtrain` cohort adds L2 feature normalization and therefore changes a second
experimental factor. The validation-only ranking used here is recorded in
`baseline_selection.csv`.

## Model B

- Frozen visual encoder loaded from baseline A
- Frozen CLIP text embeddings for the two fixed research-plan prompts
- One shared `4d -> 512 -> 128 -> 1` candidate scorer
- Cross entropy, AdamW, validation-video-AUROC selection

## Reproduction

Focused tests:

```bash
pytest -q tests/models/test_clip_visual_jev.py tests/test_sample_eval.py
```

Primary training run on physical GPUs 0-3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/2026-05-05_paper_feature_geometry/scripts/train.py \
  --config experiments/2026-09-21_visual_jev_h1/configs/visual_jev_s42.yaml
```

The seed-7 and seed-2048 configs are pre-registered replications. Devices 4-7
are intentionally excluded.
