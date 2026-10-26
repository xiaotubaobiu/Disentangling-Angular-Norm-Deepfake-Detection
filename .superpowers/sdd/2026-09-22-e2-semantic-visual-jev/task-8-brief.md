### Task 8: Add preflight, toy integration, and GPU smoke workflow

**Files:**
- Create: experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py
- Create: tests/test_e2_integration.py
- Create: experiments/2026-09-22_semantic_visual_jev_e2/README.md

**Interfaces:**
- Consumes: all Tasks 1 through 7
- Produces: preflight.json with dataset counts, method coverage, hashes, checkpoint state, devices, and trainable names
- Produces: one smoke checkpoint that is clearly marked non-reportable

- [ ] **Step 1: Write the toy end-to-end integration test**

~~~python
def test_sampler_model_step_and_evaluator_round_trip(
    fake_clip, tiny_language_bank, tiny_base_dataset
):
    semantic = SemanticDataset(
        tiny_base_dataset,
        SemanticTemplateSampler(tiny_language_bank, seed=42),
        mode="suite",
        tasks=("T1", "T2", "T3", "T4", "T5"),
        split="train",
    )
    batch = next(iter(DataLoader(semantic, batch_size=5)))
    model = build_tiny_semantic_model(fake_clip)
    lookup = EncodedTextLookup.build(
        model, tiny_language_bank, allowed_splits=("train",), device="cpu"
    )
    loss, logits = semantic_classification_step(
        model, batch, lookup, torch.device("cpu")
    )
    assert torch.isfinite(loss)
    rows = frame_semantic_rows(
        batch, logits.softmax(dim=-1), condition="seen"
    )
    assert {row["task_id"] for row in rows} == {
        "T1", "T2", "T3", "T4", "T5",
    }
~~~

- [ ] **Step 2: Run the integration test and fix only interface mismatches**

Run:

~~~bash
pytest -q tests/test_e2_integration.py
~~~

Expected before wiring: failures name exact cross-task interface mismatches.
Expected after wiring: one passing integration test.

- [ ] **Step 3: Implement preflight checks**

preflight.py validates:

- all sixteen H1 methods have non-empty train, val, FF++, and CDF records;
- label/method family truth tables cover every H1 record;
- all three H1-Jev checkpoints and the baseline-A visual checkpoint exist;
- language manifest SHA256 and train/held-out isolation;
- nine training questions and all five task IDs;
- model trainable names match the allowed prefixes;
- visible CUDA device count is four when launched with
  CUDA_VISIBLE_DEVICES=0,1,2,3;
- each visible device has at least 28 GiB free before smoke training.

The script exits nonzero on any failure and writes its JSON only after every
check passes. With --lock-output it additionally records the current git
commit, dirty-file list, config SHA256, language SHA256, checkpoint SHA256,
CUDA device inventory, and timestamp in the requested execution-manifest file.

- [ ] **Step 4: Document exact smoke commands**

README.md includes:

~~~bash
cd /home/z/project/deepfake_detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --output experiments/2026-09-22_semantic_visual_jev_e2/results/preflight.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 42 --smoke --max-train-batches 4 --max-val-batches 2
~~~

- [ ] **Step 5: Run the full CPU test suite before GPU use**

Run:

~~~bash
pytest -q
~~~

Expected: all existing and E2 tests pass.

- [ ] **Step 6: Run preflight and four-GPU smoke**

Run the two README commands. Verify:

- every GPU 0 through 3 processes at least one batch;
- peak allocated memory is below 30 GiB per GPU;
- training loss and validation CE are finite;
- candidate masks contain both two- and three-candidate examples;
- the checkpoint metadata says smoke: true and reportable: false;
- the encoded-text split list excludes heldout and control.

Controller-required operational closure from the early source-only smoke:

- replace the truncated multi-worker `itertools.islice` path with a bounded
  loader/iterator that terminates cleanly on every rank; the formal smoke must
  have no worker `SIGABRT`, `_MultiProcessingDataLoaderIter.__del__`, or hidden
  teardown exception;
- lock deterministic math-only SDPA before reportable training (with the
  required CUDA determinism environment), record the effective backend, and
  add a repeatability regression; changing architecture/loss is out of scope;
- checkpoint metadata must include `smoke: true`, `reportable: false`, encoded
  split names, config/language/checkpoint provenance hashes, and per-rank
  batches, candidate-counts, and peak allocated CUDA bytes;
- the formal command must set `CUDA_VISIBLE_DEVICES=0,1,2,3`; verify every rank
  processed data, both two- and three-candidate masks occurred, and every peak
  is below 30 GiB;
- an evaluation-only float32 frozen-visual feature cache is authorized to
  remove repeated CLIP encoding across seeds/conditions.  It must fingerprint
  visual weights, preprocessing, ordered records/domain/sample IDs, fail
  closed on stale/misaligned entries, preserve every frame and metadata row,
  and have a direct-versus-cache regression.  Training augmentations must
  never use this cache.

Read `.superpowers/sdd/2026-09-22-e2-semantic-visual-jev/early-smoke-evidence.md`
before changing code.  Narrow edits/tests to the training engine/CLI and
evaluation cache/wiring are part of Task 8 and must be included in its review.

- [ ] **Step 7: Commit workflow documentation**

~~~bash
git add experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py tests/test_e2_integration.py experiments/2026-09-22_semantic_visual_jev_e2/README.md
git commit -m "test: add E2 preflight and GPU smoke workflow"
~~~

---
