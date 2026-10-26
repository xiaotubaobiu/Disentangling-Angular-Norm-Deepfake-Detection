# Task 4 implementation report

## Outcome

Implemented and committed the leakage-safe E2 semantic training path.

- Commit: `7fabb6ca40a20751a06cb9340b1343c8eb91521f`
- Message: `feat: add leakage-safe E2 semantic training`
- Base HEAD supplied by the task: `303b396fe9e332391d29acdc9bce51989442ef4b`

The commit contains exactly these four files:

- `src/engine/semantic_trainer.py`
- `tests/test_semantic_trainer.py`
- `experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml`
- `experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py`

The pre-existing dirty migration was preserved. A pre-existing staged rename of
`src/deepfake_detection/__init__.py` to `src/__package_init_old.py` was detected
during the staged-diff audit and removed from the index without changing its
worktree state.

## TDD evidence

1. Initial trainer RED:
   `pytest -q tests/test_semantic_trainer.py` failed during collection with
   `ModuleNotFoundError: No module named 'engine.semantic_trainer'`.
2. Trainer GREEN:
   the initial six trainer tests passed after the minimal implementation.
3. CLI/config RED:
   the expanded focused suite produced nine expected failures because
   `train_semantic.py` and `semantic_jev.yaml` did not exist.
4. CLI/config GREEN:
   the expanded focused suite passed 15 tests.
5. Smoke-duration RED:
   `test_smoke_mode_runs_exactly_one_epoch` failed with
   `AttributeError: module 'train_semantic' has no attribute 'effective_epochs'`.
6. Smoke-duration GREEN:
   that test passed after smoke mode was restricted to one epoch.

## Verification results

- Required focused and legacy command:
  `pytest -q tests/test_semantic_trainer.py tests/test_losses.py tests/test_model_comparison.py`
  -> `24 passed`, exit 0.
- Final full suite:
  `pytest -q`
  -> `154 passed`, exit 0, with 11 third-party deprecation warnings.
- Syntax:
  `python -m py_compile src/engine/semantic_trainer.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py`
  -> exit 0.
- CLI help:
  `python experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py --help`
  -> exit 0 and lists the locked seed choices, smoke flags, and overwrite flag.
- Non-smoke batch cap rejection:
  invoking `--max-train-batches 1` without `--smoke` -> exit 2 with
  `max-batch flags are allowed only with --smoke`.
- Static import audit:
  parsed the training script AST and found no eval, held-out, or control loader
  imports.
- Staged diff:
  `git diff --cached --check` -> exit 0; staged name count was exactly four
  before commit.

## CE, optimizer, EMA, and checkpoint audit

- `semantic_classification_step` computes only
  `torch.nn.functional.cross_entropy(logits, target)`. There is no auxiliary,
  contrastive, calibration, or task-weighted loss path.
- `trainable_parameters` filters on `requires_grad`; the optimizer and gradient
  clipping receive only that list.
- `TrainableEMA` snapshots and updates only named trainable parameters. Frozen
  CLIP parameters are neither optimized nor stored in EMA state, and the CLIP
  module is held in eval mode during semantic training.
- Validation runs in eval/no-grad mode and all-reduces one float64 tensor
  containing loss numerator, correct-count numerator, and sample count. The
  returned loss and accuracy are therefore derived from all-rank totals.
- Best-checkpoint selection uses only strict improvement in all-rank
  seen-language validation CE.
- Checkpoint `model_state_dict` is captured with EMA weights applied, then live
  weights are restored. Required fields are present: `epoch`,
  `model_state_dict`, `ema_state_dict`, `val_seen_ce`, `seed`,
  `language_manifest_sha256`, and `config`.

## DDP, leakage, determinism, and overwrite audit

- The reportable path is deterministic:
  `results/training/seed_<seed>/best_model.pth`; smoke output is isolated under
  `results/training/smoke/seed_<seed>/best_model.pth`.
- Existing checkpoints are rejected before model/data initialization unless
  `--allow-overwrite` is supplied. Writes use a temporary sibling followed by
  atomic replacement.
- The script supports torchrun through the existing NCCL DDP initialization and
  wraps the model with one local CUDA device per rank. Direct invocation remains
  the single-process smoke path; smoke defaults to one train batch, one
  validation batch, and one epoch.
- Every train epoch invokes both the sampler's `set_epoch` when present and the
  semantic dataset's `set_epoch`.
- The only text cache is `EncodedTextLookup.build(...,
  allowed_splits=("train", "fixed"))`.
- The only instantiated loaders are semantic train and FF++ seen-language
  validation with `split="train"`. The script does not import or construct test,
  held-out, evaluation-domain, or control loaders/caches.
- The locked configuration exactly matches the accepted Task 4 YAML, including
  the seed-42 H1 visual checkpoint.
- H1 deterministic settings are applied: Python/NumPy/Torch seeds, deterministic
  cuDNN, disabled benchmarking and TF32, fixed CUBLAS workspace configuration,
  and deterministic-algorithm enforcement with warnings.

## Concerns and limitations

- A live four-GPU end-to-end torchrun was not started; doing so would consume the
  production DF40 dataset and write a large training checkpoint. DDP reduction
  and orchestration are covered by focused tests plus static/CLI checks, but the
  actual NCCL four-rank launch remains an operator smoke step.
- The full test suite emits existing third-party deprecation warnings from
  `clip/pkg_resources`, namespace packages, and Jupyter path migration; there
  were no test failures.
- The requested report is intentionally outside the four-file commit.

## Review fix round 1

Resolved both Important findings in commit
`45b78cb36a7db767e14a4107d21041844a516877` (`fix: align E2 training
with pinned runtime`). The fix commit contains only:

- `experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml`
- `experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py`
- `tests/test_semantic_trainer.py`

### PyTorch 2.2 GradScaler compatibility

- RED: the pinned-API regression removed `torch.amp.GradScaler`, exposed only
  `torch.cuda.amp.GradScaler`, and failed with `AttributeError` because
  `build_grad_scaler` did not exist.
- GREEN: production now constructs
  `torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision ==
  "amp")`. The regression passes with the simulated PyTorch 2.2 surface and
  observes `enabled=False` for CPU.
- The trainer unit test also uses the PyTorch 2.2-compatible scaler API, so the
  test suite itself does not require `torch.amp.GradScaler`.

### Effective H1 augmentation lock

- RED: the augmentation review selection produced three failures: the
  validation function was absent and checked-in YAML still contained the
  incomplete legacy declaration, including incorrect compression `p=0.2`.
- GREEN: the YAML now records the ordered executable H1 policy:
  HorizontalFlip (`p=0.5`), Rotate (`p=0.5`, `[-10, 10]`), GaussianBlur
  (`p=0.5`, `[3, 7]`), OneOf (`p=0.5`) with RandomBrightnessContrast
  (`0.1`/`0.1`), FancyPCA, and HueSaturationValue, ImageCompression (`p=0.5`,
  `[40, 100]`), plus CLIP normalization values selected by the CLIP model.
- `validate_h1_augment_policy` requires exact equality with that effective
  policy. It runs from the locked-contract validation before DDP, model, or
  loader initialization. A mutated compression probability is rejected.
- No changes were made to `src/data/builders.py` or `src/data/transforms.py`.

### Review-fix verification

- Focused review regressions: `4 passed`.
- Required focused + legacy command:
  `pytest -q tests/test_semantic_trainer.py tests/test_losses.py tests/test_model_comparison.py`
  -> `27 passed`, exit 0.
- Final full suite: `pytest -q` -> `157 passed`, exit 0, with 12 warnings
  (third-party deprecations plus the expected PyTorch 2.6 deprecation warning
  for the PyTorch 2.2-compatible CUDA AMP API).
- `python -m py_compile` for the changed Python files -> exit 0.
- Static policy audit confirmed the checked-in YAML equals the script's locked
  policy and passes validation.
- The pre-existing staged state remains exactly
  `RD src/deepfake_detection/__init__.py -> src/__package_init_old.py` after the
  explicit-pathspec fix commit.
