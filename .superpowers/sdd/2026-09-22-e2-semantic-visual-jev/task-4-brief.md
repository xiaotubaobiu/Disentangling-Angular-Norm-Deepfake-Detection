### Task 4: Add CE-only semantic training and locked configuration

**Files:**
- Create: src/engine/semantic_trainer.py
- Create: experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml
- Create: experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py
- Create: tests/test_semantic_trainer.py

**Interfaces:**
- Consumes: semantic loaders from Task 2, model and EncodedTextLookup from Task 3
- Produces: semantic_classification_step(model, batch, text_lookup, device)
- Produces: run_semantic_train_epoch(model, dataloader, optimizer, scaler, device, text_lookup, cfg, ema=None) -> float
- Produces: run_semantic_val_epoch(model, dataloader, text_lookup, device) -> dict with loss, accuracy, count
- Checkpoint fields: epoch, model_state_dict, ema_state_dict, val_seen_ce, seed, language_manifest_sha256, config

- [ ] **Step 1: Write CE-only and optimizer-scope tests**

~~~python
def test_semantic_step_is_exact_candidate_cross_entropy(toy_model, batch, lookup):
    loss, logits = semantic_classification_step(
        toy_model, batch, lookup, torch.device("cpu")
    )
    expected = torch.nn.functional.cross_entropy(logits, batch["target_index"])
    torch.testing.assert_close(loss, expected)


def test_optimizer_scope_excludes_clip(toy_model):
    parameters = trainable_parameters(toy_model)
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert not any(
        parameter is frozen
        for parameter in parameters
        for frozen in toy_model.clip.parameters()
    )


def test_validation_is_repeatable(toy_model, deterministic_loader, lookup):
    first = run_semantic_val_epoch(
        toy_model, deterministic_loader, lookup, torch.device("cpu")
    )
    second = run_semantic_val_epoch(
        toy_model, deterministic_loader, lookup, torch.device("cpu")
    )
    assert first == second
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_semantic_trainer.py
~~~

Expected: import fails because engine.semantic_trainer is absent.

- [ ] **Step 3: Implement the semantic train and validation steps**

~~~python
def semantic_classification_step(model, batch, text_lookup, device):
    images = batch["image"].to(device, non_blocking=True)
    question_ids = batch["question_text_id"].to(device)
    candidate_ids = batch["candidate_text_ids"].to(device)
    candidate_mask = batch["candidate_mask"].to(device)
    target = batch["target_index"].to(device)
    question, candidates = text_lookup.lookup(
        question_ids, candidate_ids, candidate_mask
    )
    logits = model(images, question, candidates, candidate_mask)
    loss = F.cross_entropy(logits, target)
    return loss, logits


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
~~~

Use AMP, gradient clipping at 1.0, AdamW, EMA over trainable parameters only,
and all-reduce validation loss/count across DDP ranks.

- [ ] **Step 4: Add the locked configuration**

~~~yaml
experiment_name: e2_semantic_visual_jev
dataset:
  root: /Dataset/deepfake_detection/DF40_all
  methods: []
  train_videos_per_method: 586
  val_videos_per_method: 100
  frames_per_video: 8
  image_size: 224
augment:
  horizontal_flip_p: 0.5
  rotate_p: 0.5
  rotate_limit: 10
  brightness_contrast_p: 0.5
  brightness_limit: 0.1
  contrast_limit: 0.1
  compression_p: 0.2
  compression_quality_lower: 40
  compression_quality_upper: 100
language:
  manifest: experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml
  train_tasks: [T1, T2, T3, T4, T5]
  task_probabilities: [0.2, 0.2, 0.2, 0.2, 0.2]
model:
  name: clip_semantic_jev
  clip_model_name: ViT-B/16
  d_model: 256
  nhead: 8
  ffn_dim: 1024
  num_layers: 2
  dropout: 0.1
loss:
  name: semantic_cross_entropy
train:
  init_visual_checkpoint: experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth
  output_dir: experiments/2026-09-22_semantic_visual_jev_e2/results/training
  epochs: 15
  lr: 1.0e-4
  weight_decay: 1.0e-2
  patience: 5
  warmup_epochs: 1
  ema_decay: 0.999
  per_gpu_batch: 128
  num_workers: 8
  prefetch_factor: 4
  precision: amp
  seeds: [42, 7, 2048]
evaluation:
  primary_domains: [ffpp, cdf]
  exploratory_domains: [dfd]
  permutation_audit_samples: 4096
  bootstrap_samples: 10000
  bootstrap_seed: 20260922
gates:
  noninferiority_margin: 0.01
  heldout_macro_drop_max: 0.02
  heldout_task_drop_max: 0.05
  permutation_error_max: 1.0e-6
  t2_balanced_accuracy_min: 0.80
  t2_polarity_consistency_min: 0.95
  t4_balanced_accuracy_min: 0.65
  t4_counterfactual_order_min: 0.80
  t5_macro_balanced_accuracy_min: 0.60
  meaningless_drop_min: 0.10
~~~

- [ ] **Step 5: Implement leakage-safe DDP training**

train_semantic.py accepts:

~~~text
--config PATH --seed {42,7,2048} [--smoke] [--max-train-batches N] [--max-val-batches N]
~~~

The script:

1. establishes the same deterministic settings as H1;
2. loads the language bank and builds an EncodedTextLookup for train and fixed
   strings only;
3. builds the model and loads only the H1 visual tower;
4. wraps the model in DDP;
5. builds semantic train and seen-language validation loaders;
6. calls both DistributedSampler.set_epoch and SemanticDataset.set_epoch;
7. selects the lowest all-rank validation CE;
8. saves EMA weights and manifest SHA256;
9. never builds test, held-out, or control loaders.

If --smoke is absent, max-batch arguments are rejected.
Reportable checkpoints are written deterministically to
results/training/seed_42/best_model.pth,
results/training/seed_7/best_model.pth, and
results/training/seed_2048/best_model.pth. Refuse to overwrite a reportable
checkpoint unless the operator supplies --allow-overwrite.

- [ ] **Step 6: Run focused and legacy trainer tests**

Run:

~~~bash
pytest -q tests/test_semantic_trainer.py tests/test_losses.py tests/test_model_comparison.py
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit the training path**

~~~bash
git add src/engine/semantic_trainer.py tests/test_semantic_trainer.py experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py
git commit -m "feat: add leakage-safe E2 semantic training"
~~~

---

