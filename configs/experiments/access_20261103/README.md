# H ten-epoch revision

These are actual resolved six-run registries per revised backbone, fixed-budget recipes, and backbone-to-artifact-root mappings. Absolute paths identify this machine's evidence roots, rather than a portable launcher.

H-DINOv2 and H-EfficientNet-B0 are replaced by twelve models freshly initialized from public weights, trained for ten complete epochs using learning rates5e-5 and1e-4, batch128 and unchanged optimizer/augmentation/EMA. The paired source-DEV-B mean-probability criterion selects the best EMA epoch within the budget. `fixed_epoch_budget:true` disables early stopping only for these revised tasks.

Both budgets were revised after inspection of the initial H results. The recipes record `prior_target_results_observed:true`; DFD is not used for per-epoch selection or threshold calibration. D15 and H-CLIP6 retain their predictions. All three H backbones have a ten-epoch cap. Original snapshots remain in `../journal_20261102/`.
