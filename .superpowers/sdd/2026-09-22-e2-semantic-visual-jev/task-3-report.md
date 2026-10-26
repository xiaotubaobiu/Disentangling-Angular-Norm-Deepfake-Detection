# Task 3 report: frozen-CLIP semantic decision transformer

## Outcome

Implemented and committed `CLIPSemanticVisualJev`, split-scoped
`EncodedTextLookup`, and the `clip_semantic_jev` factory registration.

Commit: `303b396fe9e332391d29acdc9bce51989442ef4b`

Commit message: `feat: add E2 cross-attention decision model`

Committed files (and only these files):

- `src/models/clip_semantic_jev.py`
- `src/models/factory.py`
- `tests/models/test_clip_semantic_jev.py`

## TDD evidence

Initial RED:

```text
pytest -q tests/models/test_clip_semantic_jev.py
```

Result: expected failure because the feature was absent: 11
`ModuleNotFoundError` setup errors for `models.clip_semantic_jev` and one
factory failure because `CLIPSemanticVisualJev` was not registered.

First implementation run:

```text
pytest -q tests/models/test_clip_semantic_jev.py
```

Result: 11 passed, 1 failed. The failure was a test-fixture typo
(`t1_q_train_1` instead of the manifest's `t1_q_train`), which was corrected.
The next run passed: 12 passed, 10 pre-existing dependency warnings.

Self-review found that an unmasked `-1` could alias an encoded row when a
lookup included every split. Regression RED:

```text
pytest -q tests/models/test_clip_semantic_jev.py::test_lookup_rejects_unmasked_padding_id_even_when_all_splits_are_encoded
```

Result: 1 failed with `DID NOT RAISE KeyError`.

After validating unmasked text-ID bounds before indexing, the same command
passed: 1 passed, 10 pre-existing dependency warnings.

## Verification

Focused Task 3 plus H1 regression, after the lookup fix:

```text
pytest -q tests/models/test_clip_semantic_jev.py tests/models/test_clip_visual_jev.py
```

Result: 18 passed, 10 pre-existing dependency warnings.

Final post-commit full suite:

```text
pytest -q
```

Result: 138 passed, 11 pre-existing dependency warnings in 24.78 seconds.

The final verification command ran the focused regression and full suite in
sequence; both exited zero.

Scoped diff/index checks:

```text
git diff --cached --check -- src/models/clip_semantic_jev.py src/models/factory.py tests/models/test_clip_semantic_jev.py
git commit --only -m "feat: add E2 cross-attention decision model" -- src/models/clip_semantic_jev.py src/models/factory.py tests/models/test_clip_semantic_jev.py
git show --pretty=format: --name-only HEAD
```

Result: no whitespace errors; the commit contains exactly the three authorized
Task 3 files. `--only` was necessary because the dirty migration already had
an unrelated staged `src/__package_init_old.py` rename, which remains
untouched.

## Actual H1 checkpoint-format evidence

Inspected read-only:

`experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth`

Observed format:

- Top-level keys: `epoch`, `model_state_dict`, `auc`, `eer`, `acc`.
- `model_state_dict` has 159 entries.
- 152 entries use the `visual.*` prefix. Examples include
  `visual.class_embedding`, `visual.positional_embedding`, `visual.proj`,
  `visual.conv1.weight`, and transformer block weights.
- The seven non-visual entries are `candidate_embeddings` and six
  `scorer.*` tensors.

The loader reads the actual `model_state_dict`, extracts only `visual.*`,
strips that prefix, and loads the resulting mapping strictly into
`model.clip.visual`. It then re-freezes all CLIP parameters and forces CLIP
back to eval mode.

Read-only integration command:

```text
PYTHONPATH=src python - <<'PY'
# Instantiate CLIPSemanticVisualJev, clone its semantic scorer, load the real
# seed-42 best_model.pth, and assert scorer equality plus frozen/eval CLIP.
PY
```

Result:

```text
loaded actual checkpoint strictly: experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth
visual tensors loaded: 152
semantic scorer unchanged: yes
CLIP frozen/eval: yes
```

## Self-review

- Confirmed three independent LayerNorm-plus-Linear projections, with actual
  ViT-B/16 dimension 512 to default decision dimension 256.
- Confirmed candidates are decoder targets, memory is exactly image followed
  by question, and no positional encoding is introduced.
- Confirmed locked defaults: two decoder layers, eight heads, FFN 1024,
  dropout 0.1, and one shared scalar scorer.
- Confirmed two/three-candidate masking, zero gradient from padded candidates,
  and all six T5 permutations within absolute tolerance `1e-6` at eval.
- Confirmed only decision modules are trainable and parent `.train()` cannot
  put CLIP into train mode.
- Confirmed train/fixed lookup construction does not select or encode held-out
  entries, and missing/unmasked invalid IDs fail loudly.
- Confirmed the H1 scorer/candidate buffer are ignored during real checkpoint
  loading.
- Confirmed existing H1 `clip_visual_jev` behavior remains green.

No Critical, Important, or deferred Minor findings remain from self-review.

## Concerns

- The repository remains in its pre-existing dirty migration state by design;
  no unrelated working-tree or index changes were modified.
- Pytest reports the same dependency deprecation warnings already present in
  the baseline (plus the existing Jupyter platformdirs warning in the full
  suite). There are no Task 3 test failures or blockers.
