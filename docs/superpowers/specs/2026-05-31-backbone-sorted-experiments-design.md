# Backbone-Sorted Experiments Design

## Goal

Move all timestamped experiment checkpoint directories into the final results area and organize them by backbone, training mode, and seed.

## Final Layout

The final result entry point remains `experiments/final_results/`.

Timestamped experiment directories move from `experiments/2026.../` to:

```text
experiments/final_results/checkpoints/<backbone>/<train_mode>/seed_<seed>__<timestamp>/
```

Backbones:

- `clip`
- `dinov2`
- `efficientnet`

Training modes:

- `raw`
- `normtrain`

Each moved directory keeps its original internal structure:

```text
config/
logs/
output/
experiment_name.txt
```

The existing `experiments/configs/` directory moves to `experiments/final_results/configs/`.

## Documentation

Update `README.md`, `ARTIFACTS.md`, and `experiments/final_results/README.md` so users start from `experiments/final_results/` and understand that the 30 checkpoints are grouped by backbone and training mode.

## Verification

After the move:

- No top-level `experiments/2026.../` directories remain.
- `find experiments/final_results/checkpoints -name best_model.pth` returns 30.
- Each backbone has 10 checkpoint directories.
- Each `(backbone, train_mode)` group has 5 checkpoint directories.
- The final result CSVs still have 60, 45, and 9 rows.
- The repository tests pass.
