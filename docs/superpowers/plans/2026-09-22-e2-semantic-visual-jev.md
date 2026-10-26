# E2 Semantic Visual-Jev Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build, train, and evaluate E2-Semantic Visual-Jev on the unchanged H1 visual protocol, then issue the preregistered supported / not supported / insufficient-evidence verdict.

**Architecture:** Wrap the existing frame dataset with a deterministic five-task semantic sampler. Feed frozen pooled CLIP image, question, and candidate embeddings into trainable 256-dimensional projections and a two-layer permutation-equivariant Transformer decoder, then score every candidate with one shared scalar head. Keep training, semantic inference, controls, paired bootstrap, and gate adjudication in focused modules so the completed H1 path remains unchanged.

**Tech Stack:** Python 3.11, PyTorch and DDP, OpenAI CLIP ViT-B/16, NumPy, pandas, scikit-learn, PyYAML, pytest, four Tesla V100 GPUs.

**Spec:** docs/superpowers/specs/2026-09-22-e2-semantic-visual-jev-design.md

## Global Constraints

- Dataset root is /Dataset/deepfake_detection/DF40_all.
- Reuse H1 real/fake labels, train/validation/test splits, sixteen methods, frame sampling, crops, augmentations, normalization, and method_id::video_id aggregation.
- The visual encoder is the exact frozen H1 CLIP ViT-B/16 visual tower.
- The CLIP text encoder is frozen; held-out text is not encoded before final evaluation.
- Only image/question/candidate projections, the two-layer decision transformer, and the shared scalar scorer are trainable.
- d_model is 256, attention heads are 8, feed-forward dimension is 1024, dropout is 0.1, and there are exactly two decoder layers.
- Loss is candidate cross entropy only.
- Candidate sets contain two items for T1 through T4 and three items for T5; padded candidates never affect logits or loss.
- Training tasks T1 through T5 are sampled uniformly in expectation.
- Final seeds are 42, 7, and 2048; GPUs 0 through 3 are the only authorized devices.
- Checkpoint selection uses deterministic FF++ validation seen-language cross entropy only.
- FF++ test, CDF, DFD, held-out text, and meaningless controls are forbidden inputs to model selection.
- G1 through G6 and their numeric thresholds are immutable after final evaluation starts.

## Review Focus

- A sample seen by different DataLoader workers or DDP ranks must render the same task and text at the same epoch; Task 2 pins stable hashing and set_epoch behavior.
- Two- and three-candidate batches must coexist without padded candidates receiving probability or gradient; Tasks 2 and 3 pin masks, target indices, and finite logits.
- Unknown-method fake records such as DFD_fake must never be silently labelled swap or reenact; Tasks 2 and 5 pin exclusion from T4/T5 and continued T1-T3 support.
- Candidate permutations must preserve canonical probabilities in evaluation mode, including all six T5 orders; Tasks 3 and 6 pin the 1e-6 invariant.
- Held-out strings and embeddings must be absent from training and validation caches; Tasks 1, 3, and 4 pin split isolation and checkpoint metadata.

## File Map

| File | Responsibility |
| --- | --- |
| experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml | Frozen train, held-out, Fixed-H1, and control language manifest |
| src/data/semantic_language.py | Parse and validate text IDs, splits, semantics, and task schemas |
| src/data/semantic_dataset.py | Stable task sampling, target derivation, padding, permutations, and dataset wrappers |
| src/data/builders.py | Expose base frame datasets and construct semantic train/validation/evaluation loaders |
| src/models/clip_semantic_jev.py | Frozen CLIP encoders, encoded-text lookup, projections, decoder, scorer, H1 visual loading |
| src/models/factory.py | Register clip_semantic_jev without changing existing model names |
| src/engine/semantic_trainer.py | Candidate CE train and deterministic seen-language validation loops |
| experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml | Locked E2 model, data, optimization, evaluation, and gate settings |
| experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py | Seeded DDP training and EMA checkpoint selection |
| src/evaluation/semantic_eval.py | Canonical probability mapping, task metrics, semantic video aggregation |
| src/evaluation/semantic_controls.py | Permutation, counterfactual, meaningless-candidate, and meaningless-question suites |
| experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py | Distributed inference and compact frame/video result export |
| src/evaluation/e2_gates.py | Three-seed summaries, paired cluster bootstrap, G1-G6 adjudication |
| experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py | Load E2/H1 artifacts and write the final machine-readable verdict |
| experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py | Validate data, checkpoints, text isolation, devices, and trainable parameters |
| experiments/2026-09-22_semantic_visual_jev_e2/README.md | Exact smoke, training, evaluation, and verdict commands |
| tests/test_semantic_language.py | Language manifest and leakage tests |
| tests/test_semantic_dataset.py | Five-task truth tables, determinism, padding, and DFD behavior |
| tests/models/test_clip_semantic_jev.py | Frozen encoders, shapes, H1 load, gradients, and permutation tests |
| tests/test_semantic_trainer.py | CE-only step, validation determinism, and optimizer-scope tests |
| tests/test_semantic_eval.py | Canonical mapping, aggregation, task metrics, and method-key tests |
| tests/test_semantic_controls.py | Exhaustive permutations and counterbalanced nonsense controls |
| tests/test_e2_gates.py | Every gate, uncertainty, and final verdict logic |
| tests/test_e2_integration.py | End-to-end toy batch through sampler, model, trainer, and evaluator |

---

### Task 1: Freeze and validate the five-task language manifest

**Files:**
- Create: experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml
- Create: src/data/semantic_language.py
- Create: tests/test_semantic_language.py

**Interfaces:**
- Produces: load_language_bank(path: str | Path) -> LanguageBank
- Produces: LanguageBank.text_id(name: str) -> int
- Produces: LanguageBank.text(name_or_id: str | int) -> str
- Produces: LanguageBank.entries(split: str, kind: str | None = None) -> tuple[TextEntry, ...]
- Produces: LanguageBank.task(task_id: str, split: str) -> TaskLanguage
- Produces: SEMANTIC_TO_ID with real, fake, no, yes, camera, swap, reenact

- [ ] **Step 1: Write failing language-isolation tests**

~~~python
from pathlib import Path

import pytest

from data.semantic_language import SEMANTIC_TO_ID, load_language_bank


MANIFEST = Path(
    "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
)


def test_language_bank_has_five_tasks_and_nine_train_questions():
    bank = load_language_bank(MANIFEST)
    assert set(bank.tasks) == {"T1", "T2", "T3", "T4", "T5"}
    assert len(bank.entries("train", kind="question")) == 9
    assert set(SEMANTIC_TO_ID) == {
        "real", "fake", "no", "yes", "camera", "swap", "reenact",
    }


def test_train_and_heldout_full_strings_are_disjoint():
    bank = load_language_bank(MANIFEST)
    train = {entry.text for entry in bank.entries("train")}
    heldout = {entry.text for entry in bank.entries("heldout")}
    assert train.isdisjoint(heldout)


def test_duplicate_text_or_id_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "texts:\n"
        "  - {id: a, split: train, kind: question, text: duplicate}\n"
        "  - {id: a, split: heldout, kind: question, text: duplicate}\n"
        "tasks: {}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_language_bank(path)
~~~

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_semantic_language.py
~~~

Expected: collection fails because data.semantic_language does not exist.

- [ ] **Step 3: Add the frozen manifest**

Create language_bank.yaml with these exact semantic strings and mappings:

~~~yaml
version: 1
fixed_h1:
  question: {id: fixed_h1_q, split: fixed, kind: question, text: "Which description best matches this facial image?"}
  candidates:
    - {id: fixed_h1_real, split: fixed, kind: candidate, semantic: real, text: "an authentic facial image."}
    - {id: fixed_h1_fake, split: fixed, kind: candidate, semantic: fake, text: "a manipulated deepfake facial image."}

answer_sets:
  train:
    - id: answer_train_short
      options:
        - {id: answer_train_yes_short, split: train, kind: candidate, semantic: yes, text: "yes."}
        - {id: answer_train_no_short, split: train, kind: candidate, semantic: no, text: "no."}
    - id: answer_train_full
      options:
        - {id: answer_train_yes_full, split: train, kind: candidate, semantic: yes, text: "yes, it is."}
        - {id: answer_train_no_full, split: train, kind: candidate, semantic: no, text: "no, it is not."}
  heldout:
    - id: answer_heldout_formal
      options:
        - {id: answer_heldout_yes_formal, split: heldout, kind: candidate, semantic: yes, text: "affirmative."}
        - {id: answer_heldout_no_formal, split: heldout, kind: candidate, semantic: no, text: "negative."}
    - id: answer_heldout_statement
      options:
        - {id: answer_heldout_yes_statement, split: heldout, kind: candidate, semantic: yes, text: "that is correct."}
        - {id: answer_heldout_no_statement, split: heldout, kind: candidate, semantic: no, text: "that is incorrect."}

tasks:
  T1:
    train:
      questions:
        - {id: t1_q_train, split: train, kind: question, text: "What is the authenticity status of this facial image?"}
      candidate_sets:
        - id: t1_c_train
          options:
            - {id: t1_real_train, split: train, kind: candidate, semantic: real, text: "an authentic face image."}
            - {id: t1_fake_train, split: train, kind: candidate, semantic: fake, text: "a manipulated face image."}
    heldout:
      questions:
        - {id: t1_q_heldout, split: heldout, kind: question, text: "Which authenticity category correctly describes this face?"}
      candidate_sets:
        - id: t1_c_heldout
          options:
            - {id: t1_real_heldout, split: heldout, kind: candidate, semantic: real, text: "a bona fide facial photograph."}
            - {id: t1_fake_heldout, split: heldout, kind: candidate, semantic: fake, text: "a facial photograph containing digital tampering."}

  T2:
    train:
      questions:
        - {id: t2_real_q1_train, split: train, kind: question, yes_means: real, text: "Is this facial image authentic?"}
        - {id: t2_real_q2_train, split: train, kind: question, yes_means: real, text: "Does this face appear unaltered?"}
        - {id: t2_fake_q1_train, split: train, kind: question, yes_means: fake, text: "Is this facial image manipulated?"}
        - {id: t2_fake_q2_train, split: train, kind: question, yes_means: fake, text: "Has this face been digitally altered?"}
      answer_sets: [answer_train_short, answer_train_full]
    heldout:
      questions:
        - {id: t2_real_q1_heldout, split: heldout, kind: question, yes_means: real, text: "Was this face captured without forensic modification?"}
        - {id: t2_real_q2_heldout, split: heldout, kind: question, yes_means: real, text: "Is the depicted face genuine and untouched?"}
        - {id: t2_fake_q1_heldout, split: heldout, kind: question, yes_means: fake, text: "Does this face contain artificial manipulation?"}
        - {id: t2_fake_q2_heldout, split: heldout, kind: question, yes_means: fake, text: "Has the facial content been tampered with?"}
      answer_sets: [answer_heldout_formal, answer_heldout_statement]

  T3:
    train:
      questions:
        - {id: t3_q_train, split: train, kind: question, text: "Which description best characterizes this face image?"}
      candidate_sets:
        - id: t3_c1_train
          options:
            - {id: t3_real_c1_train, split: train, kind: candidate, semantic: real, text: "a real human face photo."}
            - {id: t3_fake_c1_train, split: train, kind: candidate, semantic: fake, text: "a fake face photo."}
        - id: t3_c2_train
          options:
            - {id: t3_real_c2_train, split: train, kind: candidate, semantic: real, text: "a genuine facial image."}
            - {id: t3_fake_c2_train, split: train, kind: candidate, semantic: fake, text: "a manipulated facial image."}
        - id: t3_c3_train
          options:
            - {id: t3_real_c3_train, split: train, kind: candidate, semantic: real, text: "an unaltered facial photograph."}
            - {id: t3_fake_c3_train, split: train, kind: candidate, semantic: fake, text: "a forged facial photograph."}
    heldout:
      questions:
        - {id: t3_q_heldout, split: heldout, kind: question, text: "Select the phrase that accurately states this face's authenticity."}
      candidate_sets:
        - id: t3_c1_heldout
          options:
            - {id: t3_real_c1_heldout, split: heldout, kind: candidate, semantic: real, text: "a bona fide camera photograph of a face."}
            - {id: t3_fake_c1_heldout, split: heldout, kind: candidate, semantic: fake, text: "a digitally tampered facial photograph."}
        - id: t3_c2_heldout
          options:
            - {id: t3_real_c2_heldout, split: heldout, kind: candidate, semantic: real, text: "a natural untouched facial image."}
            - {id: t3_fake_c2_heldout, split: heldout, kind: candidate, semantic: fake, text: "an artificially altered facial image."}

  T4:
    train:
      questions:
        - {id: t4_swap_q_train, split: train, kind: question, yes_means: swap, text: "Is this face identity-swapped?"}
        - {id: t4_reenact_q_train, split: train, kind: question, yes_means: reenact, text: "Is this face facially reenacted?"}
      answer_sets: [answer_train_short, answer_train_full]
    heldout:
      questions:
        - {id: t4_swap_q_heldout, split: heldout, kind: question, yes_means: swap, text: "Was the person's facial identity replaced with another?"}
        - {id: t4_reenact_q_heldout, split: heldout, kind: question, yes_means: reenact, text: "Was this facial performance driven or reenacted?"}
      answer_sets: [answer_heldout_formal, answer_heldout_statement]

  T5:
    train:
      questions:
        - {id: t5_q_train, split: train, kind: question, text: "What is the likely provenance of this facial image?"}
      candidate_sets:
        - id: t5_c1_train
          options:
            - {id: t5_camera_c1_train, split: train, kind: candidate, semantic: camera, text: "a camera-captured, unmanipulated face."}
            - {id: t5_swap_c1_train, split: train, kind: candidate, semantic: swap, text: "an identity-swapped face."}
            - {id: t5_reenact_c1_train, split: train, kind: candidate, semantic: reenact, text: "a facially reenacted face."}
        - id: t5_c2_train
          options:
            - {id: t5_camera_c2_train, split: train, kind: candidate, semantic: camera, text: "an authentic face recorded by a camera."}
            - {id: t5_swap_c2_train, split: train, kind: candidate, semantic: swap, text: "a face created by identity replacement."}
            - {id: t5_reenact_c2_train, split: train, kind: candidate, semantic: reenact, text: "a face modified by expression or pose reenactment."}
    heldout:
      questions:
        - {id: t5_q_heldout, split: heldout, kind: question, text: "Which process most likely produced the face shown here?"}
      candidate_sets:
        - id: t5_c_heldout
          options:
            - {id: t5_camera_heldout, split: heldout, kind: candidate, semantic: camera, text: "a naturally recorded face with no digital facial edit."}
            - {id: t5_swap_heldout, split: heldout, kind: candidate, semantic: swap, text: "a face whose identity was replaced by a different person."}
            - {id: t5_reenact_heldout, split: heldout, kind: candidate, semantic: reenact, text: "a face whose motion or expression was artificially driven."}

controls:
  candidate_sets:
    binary:
      - {id: nonsense_binary_a, split: control, kind: candidate, text: "the daxen facial state."}
      - {id: nonsense_binary_b, split: control, kind: candidate, text: "the norvik facial state."}
    ternary:
      - {id: nonsense_ternary_a, split: control, kind: candidate, text: "the pelto image origin."}
      - {id: nonsense_ternary_b, split: control, kind: candidate, text: "the ravik image origin."}
      - {id: nonsense_ternary_c, split: control, kind: candidate, text: "the sulem image origin."}
  questions:
    - {id: nonsense_q1, split: control, kind: question, text: "Does this item satisfy the unnamed property?"}
    - {id: nonsense_q2, split: control, kind: question, text: "Which abstract condition applies to this item?"}
~~~

- [ ] **Step 4: Implement strict parsing and validation**

Implement immutable entries and deterministic integer IDs sorted by symbolic
ID. Reject duplicate symbolic IDs, duplicate full strings, unknown task
references, invalid semantics, non-binary answer sets, non-ternary T5 sets,
and any train/held-out full-string overlap.

~~~python
SEMANTIC_TO_ID = {
    "real": 0, "fake": 1, "no": 2, "yes": 3,
    "camera": 4, "swap": 5, "reenact": 6,
}


@dataclass(frozen=True)
class TextEntry:
    name: str
    text_id: int
    text: str
    split: str
    kind: str
    semantic: str | None = None
    yes_means: str | None = None


def load_language_bank(path: str | Path) -> LanguageBank:
    raw = yaml.safe_load(Path(path).read_text())
    entries = _collect_and_validate_entries(raw)
    return LanguageBank(raw=raw, text_entries=entries)
~~~

- [ ] **Step 5: Run the focused tests**

Run:

~~~bash
pytest -q tests/test_semantic_language.py
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit the frozen language contract**

~~~bash
git add src/data/semantic_language.py tests/test_semantic_language.py experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml
git commit -m "feat: freeze E2 semantic language bank"
~~~

---

### Task 2: Build deterministic five-task semantic datasets

**Files:**
- Create: src/data/semantic_dataset.py
- Modify: src/data/builders.py
- Create: tests/test_semantic_dataset.py

**Interfaces:**
- Consumes: LanguageBank and SEMANTIC_TO_ID from Task 1
- Produces: method_family(method_id: str, label: int) -> str
- Produces: SemanticTemplateSampler.render(sample: dict, task_id: str | None, split: str, epoch: int, permutation: tuple[int, ...] | None = None) -> dict
- Produces: SemanticDataset(base_dataset, sampler, mode: str, tasks: tuple[str, ...])
- Produces: SemanticDataset.set_epoch(epoch: int) -> None
- Produces: build_semantic_train_loader, build_semantic_val_loader, and build_semantic_eval_loader
- Batch tensors: question_text_id [B], candidate_text_ids [B,3], candidate_semantic_ids [B,3], candidate_mask [B,3], target_index [B], task_index [B], question_yes_means [B]

- [ ] **Step 1: Write truth-table, determinism, and padding tests**

~~~python
import torch

from data.semantic_dataset import SemanticTemplateSampler, method_family
from data.semantic_language import load_language_bank


def _sample(label, method_id, sample_id="sample-1"):
    return {
        "image": torch.zeros(3, 224, 224),
        "label": torch.tensor(label),
        "method_id": method_id,
        "video_id": "001",
        "video_key": f"{method_id}::001",
        "image_path": "/tmp/frame.png",
        "sample_id": sample_id,
    }


def test_method_family_never_guesses_unknown_fake():
    assert method_family("real", 0) == "camera"
    assert method_family("simswap", 1) == "swap"
    assert method_family("sadtalker", 1) == "reenact"
    assert method_family("DFD_fake", 1) == "unknown"


def test_t2_target_flips_with_question_polarity(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    real = _sample(0, "real")
    positive = sampler.render(
        real, "T2", "train", 0, force_question="t2_real_q1_train",
        force_candidate_set="answer_train_short", permutation=(0, 1),
    )
    negative = sampler.render(
        real, "T2", "train", 0, force_question="t2_fake_q1_train",
        force_candidate_set="answer_train_short", permutation=(0, 1),
    )
    assert positive["target_semantic"] == "yes"
    assert negative["target_semantic"] == "no"


def test_t4_and_t5_truth_tables(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    swap = _sample(1, "simswap")
    assert sampler.render(
        swap, "T4", "train", 0, force_question="t4_swap_q_train"
    )["target_semantic"] == "yes"
    assert sampler.render(
        swap, "T4", "train", 0, force_question="t4_reenact_q_train"
    )["target_semantic"] == "no"
    t5 = sampler.render(swap, "T5", "train", 0)
    assert t5["target_semantic"] == "swap"
    assert t5["candidate_mask"].tolist() == [True, True, True]


def test_binary_candidates_are_padded_without_targeting_padding(bank):
    row = SemanticTemplateSampler(bank, seed=42).render(
        _sample(1, "fsgan"), "T1", "train", 0
    )
    assert row["candidate_mask"].tolist().count(True) == 2
    assert row["candidate_text_ids"].shape == (3,)
    assert row["target_index"] < 2


def test_hashing_is_repeatable_and_epoch_sensitive(bank):
    sampler_a = SemanticTemplateSampler(bank, seed=2048)
    sampler_b = SemanticTemplateSampler(bank, seed=2048)
    item = _sample(1, "fomm", sample_id="stable-id")
    first = sampler_a.render(item, None, "train", 3)
    repeated = sampler_b.render(item, None, "train", 3)
    changed = sampler_a.render(item, None, "train", 4)
    assert first["render_id"] == repeated["render_id"]
    torch.testing.assert_close(
        first["candidate_text_ids"], repeated["candidate_text_ids"]
    )
    assert first["render_id"] != changed["render_id"]


def test_language_render_is_identical_with_zero_or_two_workers(bank, base_dataset):
    zero_worker = SemanticDataset(
        base_dataset, SemanticTemplateSampler(bank, seed=42), mode="train"
    )
    two_workers = SemanticDataset(
        base_dataset, SemanticTemplateSampler(bank, seed=42), mode="train"
    )
    zero_worker.set_epoch(6)
    two_workers.set_epoch(6)
    ids_zero = [
        render_id
        for batch in DataLoader(zero_worker, batch_size=8, num_workers=0)
        for render_id in batch["render_id"]
    ]
    ids_two = [
        render_id
        for batch in DataLoader(two_workers, batch_size=8, num_workers=2)
        for render_id in batch["render_id"]
    ]
    assert ids_zero == ids_two


def test_unknown_fake_supports_t1_to_t3_but_rejects_t4_t5(bank):
    sampler = SemanticTemplateSampler(bank, seed=42)
    dfd = _sample(1, "DFD_fake")
    sampler.render(dfd, "T1", "heldout", 0)
    for task in ("T4", "T5"):
        with pytest.raises(ValueError, match="unknown method family"):
            sampler.render(dfd, task, "heldout", 0)
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~bash
pytest -q tests/test_semantic_dataset.py
~~~

Expected: import fails because data.semantic_dataset is absent.

- [ ] **Step 3: Implement stable hashing and method-family truth**

~~~python
TASKS = ("T1", "T2", "T3", "T4", "T5")
MAX_CANDIDATES = 3


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def method_family(method_id: str, label: int) -> str:
    if int(label) == 0:
        return "camera"
    if method_id in FACE_SWAP_METHODS:
        return "swap"
    if method_id in REENACTMENT_METHODS:
        return "reenact"
    return "unknown"
~~~

Task selection uses _stable_u64(seed, epoch, sample_id) modulo five.
Question, candidate-set, and permutation choices use an independent
random.Random seeded by _stable_u64(seed, epoch, sample_id, task_id).

- [ ] **Step 4: Implement render logic and fixed-size tensors**

The renderer must compute canonical truth before permutation, then permute
text IDs and semantic IDs together:

~~~python
def _pack_candidates(options, target_semantic, order):
    ordered = [options[index] for index in order]
    target_index = next(
        index for index, option in enumerate(ordered)
        if option.semantic == target_semantic
    )
    text_ids = torch.full((MAX_CANDIDATES,), -1, dtype=torch.long)
    semantic_ids = torch.full((MAX_CANDIDATES,), -1, dtype=torch.long)
    mask = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
    for index, option in enumerate(ordered):
        text_ids[index] = option.text_id
        semantic_ids[index] = SEMANTIC_TO_ID[option.semantic]
        mask[index] = True
    return text_ids, semantic_ids, mask, target_index
~~~

For T2, target yes exactly when question.yes_means equals real/fake visual
truth. For T4, target yes exactly when question.yes_means equals swap/reenact
method family. For T5, target is camera/swap/reenact.

- [ ] **Step 5: Add train and deterministic suite wrappers**

~~~python
class SemanticDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, sampler, mode="train", tasks=TASKS, split="train"):
        self.base_dataset = base_dataset
        self.sampler = sampler
        self.mode = mode
        self.tasks = tuple(tasks)
        self.split = split
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.mode == "train":
            return len(self.base_dataset)
        return len(self.base_dataset) * len(self.tasks)

    def __getitem__(self, index):
        if self.mode == "train":
            base_index, task_id = index, None
        else:
            base_index = index // len(self.tasks)
            task_id = self.tasks[index % len(self.tasks)]
        sample = self.base_dataset[base_index]
        return self.sampler.render(sample, task_id, self.split, self.epoch)
~~~

Suite builders must omit T4/T5 for DFD and preserve the existing H1
DistributedSampler, worker seeding, pinning, and shuffle behavior.

- [ ] **Step 6: Refactor builders without changing old loader outputs**

Extract build_train_dataset, build_val_dataset, and build_eval_dataset from
the existing builders. Existing build_train_loader, build_val_loader, and
build_eval_loader call those functions, so H1 behavior remains byte-for-byte
equivalent at the sample-dictionary boundary. Add semantic loader functions
that wrap the base dataset and use default collation on fixed-shape tensors.

- [ ] **Step 7: Run semantic and legacy data tests**

Run:

~~~bash
pytest -q tests/test_semantic_dataset.py tests/test_sample_eval.py tests/test_video_aggregation.py
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit the semantic data layer**

~~~bash
git add src/data/semantic_dataset.py src/data/builders.py tests/test_semantic_dataset.py
git commit -m "feat: add deterministic E2 semantic sampler"
~~~

---

### Task 3: Implement the frozen-CLIP semantic decision transformer

**Files:**
- Create: src/models/clip_semantic_jev.py
- Modify: src/models/factory.py
- Create: tests/models/test_clip_semantic_jev.py

**Interfaces:**
- Consumes: LanguageBank text IDs from Task 1
- Produces: CLIPSemanticVisualJev
- Produces: CLIPSemanticVisualJev.extract_features(images) -> [B,512]
- Produces: CLIPSemanticVisualJev.encode_texts(texts) -> [N,512]
- Produces: CLIPSemanticVisualJev.classify_features(image_features, question_embeddings, candidate_embeddings, candidate_mask) -> [B,N]
- Produces: CLIPSemanticVisualJev.load_h1_visual_checkpoint(path) -> None
- Produces: EncodedTextLookup.build(model, bank, allowed_splits, device)
- Produces: EncodedTextLookup.lookup(question_ids, candidate_ids, candidate_mask)

- [ ] **Step 1: Write model contract tests with a fake CLIP**

~~~python
def test_only_decision_modules_are_trainable(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("image_projection.", "question_projection.",
                         "candidate_projection.", "decoder.", "scorer."))
        for name in trainable
    )
    assert not model.clip.training


@pytest.mark.parametrize("candidate_count", [2, 3])
def test_two_and_three_candidate_shapes_and_padding(patched_clip, candidate_count):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    image = torch.randn(4, 4)
    question = torch.randn(4, 4)
    candidates = torch.randn(4, 3, 4)
    mask = torch.arange(3)[None, :] < candidate_count
    mask = mask.expand(4, -1)
    logits = model.classify_features(image, question, candidates, mask)
    assert logits.shape == (4, 3)
    assert torch.isneginf(logits[:, candidate_count:]).all()
    assert torch.isfinite(logits[:, :candidate_count]).all()


def test_all_t5_permutations_are_equivariant(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16, dropout=0.0
    ).eval()
    image = torch.randn(2, 4)
    question = torch.randn(2, 4)
    candidates = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    reference = model.classify_features(image, question, candidates, mask)
    for order in itertools.permutations(range(3)):
        order = list(order)
        actual = model.classify_features(
            image, question, candidates[:, order], mask[:, order]
        )
        torch.testing.assert_close(actual, reference[:, order], atol=1e-6, rtol=0)


def test_padded_candidate_has_no_gradient(patched_clip):
    model = patched_clip.CLIPSemanticVisualJev(
        "fake", d_model=8, nhead=2, ffn_dim=16, dropout=0.0
    )
    image = torch.randn(2, 4)
    question = torch.randn(2, 4)
    candidates = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    logits = model.classify_features(image, question, candidates, mask)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert candidates.grad[:, 2].abs().max().item() == 0.0


def test_h1_checkpoint_loads_only_visual_weights(patched_clip, tmp_path):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    expected = torch.full_like(model.clip.visual.projection.weight, 2.5)
    path = tmp_path / "h1.pth"
    torch.save({"model_state_dict": {
        "visual.projection.weight": expected,
        "scorer.0.weight": torch.randn(2, 2),
    }}, path)
    model.load_h1_visual_checkpoint(path)
    torch.testing.assert_close(model.clip.visual.projection.weight, expected)


def test_train_lookup_rejects_heldout_text_id(patched_clip, language_bank):
    model = patched_clip.CLIPSemanticVisualJev("fake", d_model=8, nhead=2, ffn_dim=16)
    lookup = patched_clip.EncodedTextLookup.build(
        model, language_bank, allowed_splits=("train", "fixed"),
        device=torch.device("cpu"),
    )
    heldout_id = language_bank.text_id("t1_q_heldout")
    with pytest.raises(KeyError, match="not encoded"):
        lookup.lookup(
            torch.tensor([heldout_id]),
            torch.tensor([[language_bank.text_id("t1_real_train"), -1, -1]]),
            torch.tensor([[True, False, False]]),
        )
~~~

- [ ] **Step 2: Run the model tests and see the import failure**

Run:

~~~bash
pytest -q tests/models/test_clip_semantic_jev.py
~~~

Expected: import fails because models.clip_semantic_jev is absent.

- [ ] **Step 3: Implement frozen encoders and H1 visual loading**

Keep the complete CLIP module to retain encode_text, freeze every CLIP
parameter, and force it into evaluation mode:

~~~python
class CLIPSemanticVisualJev(nn.Module):
    def __init__(self, clip_model_name="ViT-B/16", d_model=256, nhead=8,
                 ffn_dim=1024, num_layers=2, dropout=0.1):
        super().__init__()
        self.clip, _ = clip.load(clip_model_name, device="cpu")
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()
        feature_dim = int(self.clip.visual.output_dim)
        self.image_projection = Projection(feature_dim, d_model)
        self.question_projection = Projection(feature_dim, d_model)
        self.candidate_projection = Projection(feature_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.scorer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def extract_features(self, images):
        with torch.no_grad():
            return self.clip.encode_image(images).float()

    def encode_texts(self, texts):
        tokens = clip.tokenize(list(texts), truncate=True).to(next(self.parameters()).device)
        with torch.no_grad():
            return self.clip.encode_text(tokens).float()
~~~

load_h1_visual_checkpoint extracts state keys beginning with visual. and loads
them strictly into self.clip.visual. It rejects a checkpoint with missing
visual keys.

- [ ] **Step 4: Implement masked permutation-equivariant scoring**

~~~python
def classify_features(self, image_features, question_embeddings,
                      candidate_embeddings, candidate_mask):
    memory = torch.stack((
        self.image_projection(image_features.float()),
        self.question_projection(question_embeddings.float()),
    ), dim=1)
    queries = self.candidate_projection(candidate_embeddings.float())
    decoded = self.decoder(
        tgt=queries,
        memory=memory,
        tgt_key_padding_mask=~candidate_mask,
    )
    logits = self.scorer(decoded).squeeze(-1)
    return logits.masked_fill(~candidate_mask, float("-inf"))

def forward(self, images, question_embeddings, candidate_embeddings,
            candidate_mask):
    return self.classify_features(
        self.extract_features(images),
        question_embeddings,
        candidate_embeddings,
        candidate_mask,
    )
~~~

- [ ] **Step 5: Implement split-scoped encoded text lookup**

EncodedTextLookup builds embeddings only for explicitly allowed splits and
uses a dense text-ID-to-row tensor initialized to -1. lookup raises if an
unmasked ID is absent, which makes accidental held-out access during training
fail loudly.

~~~python
@dataclass
class EncodedTextLookup:
    row_for_text_id: torch.Tensor
    embeddings: torch.Tensor

    def lookup(self, question_ids, candidate_ids, candidate_mask):
        q_rows = self.row_for_text_id[question_ids]
        c_rows = self.row_for_text_id[candidate_ids.clamp_min(0)]
        if (q_rows < 0).any() or (c_rows[candidate_mask] < 0).any():
            raise KeyError("text ID is not encoded in this split-scoped lookup")
        questions = self.embeddings[q_rows]
        candidates = self.embeddings[c_rows.clamp_min(0)]
        candidates = candidates.masked_fill(~candidate_mask[..., None], 0.0)
        return questions, candidates
~~~

- [ ] **Step 6: Register the model factory name**

Add clip_semantic_jev to build_model and pass the five architecture values
from configuration. Do not alter clip_visual_jev.

- [ ] **Step 7: Run model and H1 regression tests**

Run:

~~~bash
pytest -q tests/models/test_clip_semantic_jev.py tests/models/test_clip_visual_jev.py
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit the model**

~~~bash
git add src/models/clip_semantic_jev.py src/models/factory.py tests/models/test_clip_semantic_jev.py
git commit -m "feat: add E2 cross-attention decision model"
~~~

---

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

### Task 5: Implement canonical semantic evaluation and video metrics

**Files:**
- Create: src/evaluation/semantic_eval.py
- Create: experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
- Create: tests/test_semantic_eval.py

**Interfaces:**
- Consumes: semantic eval loaders, CLIPSemanticVisualJev, held-out EncodedTextLookup
- Produces: map_candidate_probabilities(probabilities, semantic_ids, mask) -> [B,7]
- Produces: canonical_fake_probability(row: dict) -> float
- Produces: validate_domain_task(domain: str, task_id: str) -> None
- Produces: frame_semantic_rows(batch, probabilities, condition) -> list[dict]
- Produces: aggregate_semantic_videos(rows) -> pandas.DataFrame
- Produces: compute_task_metrics(video_rows) -> pandas.DataFrame
- Output: results/evaluation/seed_{42,7,2048}/{condition}/{domain}_frames.npz
- Output: results/evaluation/seed_{42,7,2048}/{condition}/{domain}_videos.csv
- Output: results/evaluation/seed_{42,7,2048}/{condition}/{domain}_metrics.json

- [ ] **Step 1: Write canonical-mapping and aggregation tests**

~~~python
def test_t2_fake_probability_depends_on_question_polarity():
    authentic = {"task_id": "T2", "question_yes_means": "real",
                 "p_yes": 0.2, "p_no": 0.8}
    manipulated = {"task_id": "T2", "question_yes_means": "fake",
                   "p_yes": 0.8, "p_no": 0.2}
    assert canonical_fake_probability(authentic) == pytest.approx(0.8)
    assert canonical_fake_probability(manipulated) == pytest.approx(0.8)


def test_t5_fake_probability_is_one_minus_camera():
    row = {"task_id": "T5", "p_camera": 0.15,
           "p_swap": 0.60, "p_reenact": 0.25}
    assert canonical_fake_probability(row) == pytest.approx(0.85)


def test_video_aggregation_keeps_same_id_from_different_methods_separate():
    rows = [
        frame_row("simswap", "001", 1, 0.9),
        frame_row("blendface", "001", 1, 0.2),
        frame_row("real", "001", 0, 0.1),
    ]
    videos = aggregate_semantic_videos(rows)
    assert set(videos.video_key) == {
        "simswap::001", "blendface::001", "real::001",
    }


def test_dfd_metrics_exclude_t4_and_t5():
    with pytest.raises(ValueError, match="DFD supports only T1-T3"):
        validate_domain_task("dfd", "T4")
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_semantic_eval.py
~~~

Expected: import fails because evaluation.semantic_eval is absent.

- [ ] **Step 3: Implement canonical mapping**

Scatter each valid candidate probability into a seven-column semantic tensor.
Rows without a semantic receive zero; duplicate semantics in one candidate set
are rejected.

~~~python
def map_candidate_probabilities(probabilities, semantic_ids, mask):
    mapped = probabilities.new_zeros((probabilities.shape[0], 7))
    for column in range(probabilities.shape[1]):
        valid = mask[:, column]
        mapped[valid, semantic_ids[valid, column]] = probabilities[valid, column]
    return mapped
~~~

T1/T3 fake probability is p_fake. T2 uses p_no when yes_means is real and
p_yes when yes_means is fake. T5 uses 1 - p_camera. T4 stores p_yes for each
question and combines swap/reenact pairs after matching condition, seed,
sample_id, and video key.

- [ ] **Step 4: Implement semantic video aggregation and metrics**

Group frame rows by:

~~~python
GROUP_KEYS = [
    "seed", "condition", "task_id", "variant_id", "question_yes_means",
    "method_id", "video_id", "video_key", "label", "method_family",
]
~~~

Average canonical probabilities over frames. Reject duplicate sample IDs
inside one semantic condition. Compute:

- T1/T3 and Fixed-H1 video AUROC, AP, Brier, accuracy, and ECE;
- T2/T4 video balanced accuracy and F1;
- T5 video macro balanced accuracy and macro F1;
- task counts and per-method slices.

- [ ] **Step 5: Implement distributed inference without selection leakage**

evaluate_semantic.py accepts config, checkpoint, seed, domains, conditions,
batch size, and output directory. It loads the selected checkpoint, encodes
only the text splits requested by the condition, gathers rows across ranks,
deduplicates by semantic condition plus sample_id, and writes compressed frame
arrays plus human-readable video and metric files.

The script refuses heldout or control evaluation when checkpoint metadata lacks
the expected train-manifest SHA256.

- [ ] **Step 6: Run evaluation and H1 aggregation regression tests**

Run:

~~~bash
pytest -q tests/test_semantic_eval.py tests/test_sample_eval.py tests/test_video_aggregation.py
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit semantic evaluation**

~~~bash
git add src/evaluation/semantic_eval.py tests/test_semantic_eval.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
git commit -m "feat: add canonical E2 semantic evaluation"
~~~

---

### Task 6: Add permutation, counterfactual, and meaningless-text controls

**Files:**
- Create: src/evaluation/semantic_controls.py
- Modify: experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
- Create: tests/test_semantic_controls.py

**Interfaces:**
- Produces: enumerate_candidate_permutations(candidate_count: int)
- Produces: permutation_audit(model, image_features, question_embeddings, candidate_embeddings, candidate_mask) -> dict
- Produces: build_counterfactual_requests(task_id, split, sample)
- Produces: build_meaningless_candidate_request(meaningful_request, sample_id)
- Produces: build_meaningless_question_request(meaningful_request, sample_id)

- [ ] **Step 1: Write exhaustive control tests**

~~~python
def test_permutation_enumeration_is_complete():
    assert enumerate_candidate_permutations(2) == ((0, 1), (1, 0))
    assert len(enumerate_candidate_permutations(3)) == 6


def test_nonsense_assignment_is_position_and_label_balanced(bank):
    assignments = [
        build_meaningless_candidate_request(
            meaningful_binary_request(label=index % 2), f"sample-{index}", bank
        )
        for index in range(400)
    ]
    counts = Counter(
        (request.target_index, request.label) for request in assignments
    )
    assert max(counts.values()) - min(counts.values()) <= 20


def test_t2_counterfactual_keeps_implied_fake_probability():
    pair = build_counterfactual_requests("T2", "heldout", fake_sample())
    assert {request.question_yes_means for request in pair} == {"real", "fake"}
    assert {request.target_semantic for request in pair} == {"no", "yes"}


def test_t4_counterfactual_asks_both_families():
    pair = build_counterfactual_requests("T4", "heldout", swap_sample())
    assert {request.question_yes_means for request in pair} == {"swap", "reenact"}
    assert {request.target_semantic for request in pair} == {"yes", "no"}
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_semantic_controls.py
~~~

Expected: import fails because evaluation.semantic_controls is absent.

- [ ] **Step 3: Implement exhaustive permutation auditing**

For each audited batch, cache image features once. Evaluate every valid order,
invert the order on output, and compare canonical softmax probabilities with
the identity order.

~~~python
error = (remapped_probabilities - reference_probabilities).abs().max().item()
consistent = (
    remapped_probabilities.argmax(dim=-1)
    == reference_probabilities.argmax(dim=-1)
).all().item()
~~~

Aggregate maximum error and total consistency count across DDP ranks.

- [ ] **Step 4: Implement paired question controls**

T2 pairs one held-out real-polarity question with one held-out fake-polarity
question for the same sample and answer set. T4 pairs swap and reenactment
questions. render_id records the common pair ID so evaluation joins exactly,
never by row order.

- [ ] **Step 5: Implement counterbalanced meaningless controls**

Use stable hashing of control name and sample_id to permute the two or three
pseudowords independently of image label and output position. Retain a hidden
canonical semantic assignment only in evaluation metadata. Meaningless
questions replace the meaningful question while retaining its hidden
yes_means field, so the paired score drop is well-defined but the model cannot
recover polarity from the text.

- [ ] **Step 6: Extend the evaluator with explicit conditions**

Accepted conditions are:

~~~text
fixed_h1 seen heldout permutation counterfactual
meaningless_candidates meaningless_questions
~~~

The permutation condition runs on the first 4,096 deterministically sorted
samples per domain. Every other condition uses the full eligible domain.

- [ ] **Step 7: Run control and model invariance tests**

Run:

~~~bash
pytest -q tests/test_semantic_controls.py tests/models/test_clip_semantic_jev.py
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit controls**

~~~bash
git add src/evaluation/semantic_controls.py tests/test_semantic_controls.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py
git commit -m "feat: add E2 semantic control suites"
~~~

---

### Task 7: Implement three-seed bootstrap and G1-G6 adjudication

**Files:**
- Create: src/evaluation/e2_gates.py
- Create: experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py
- Create: tests/test_e2_gates.py

**Interfaces:**
- Produces: paired_seed_video_bootstrap(baseline, candidate, metric, n_resamples, seed) -> BootstrapResult
- Produces: evaluate_gates(metric_rows, control_rows, permutation_rows, baseline_rows, candidate_rows, cfg) -> GateReport
- Produces: GateReport.verdict in supported, not_supported, insufficient_evidence

- [ ] **Step 1: Write synthetic pass, fail, and uncertainty tests**

~~~python
def test_all_gates_pass_returns_supported(passing_artifacts, gate_config):
    report = evaluate_gates(**passing_artifacts, cfg=gate_config)
    assert report.verdict == "supported"
    assert all(gate.status == "pass" for gate in report.gates.values())


def test_point_failure_returns_not_supported(passing_artifacts, gate_config):
    passing_artifacts["metric_rows"].loc[
        lambda frame: frame.task_id == "T5", "heldout_score"
    ] = 0.55
    report = evaluate_gates(**passing_artifacts, cfg=gate_config)
    assert report.gates["G5"].status == "fail"
    assert report.verdict == "not_supported"


def test_ci_crossing_margin_returns_insufficient_evidence(
    passing_artifacts, gate_config
):
    passing_artifacts["baseline_rows"] = noisy_noninferiority_fixture()
    report = evaluate_gates(**passing_artifacts, cfg=gate_config)
    assert report.gates["G1"].point_passed is True
    assert report.gates["G1"].ci_passed is False
    assert report.verdict == "insufficient_evidence"


def test_video_key_mismatch_is_rejected(passing_artifacts, gate_config):
    passing_artifacts["candidate_rows"] = (
        passing_artifacts["candidate_rows"].iloc[:-1]
    )
    with pytest.raises(ValueError, match="video-key sets differ"):
        evaluate_gates(**passing_artifacts, cfg=gate_config)
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_e2_gates.py
~~~

Expected: import fails because evaluation.e2_gates is absent.

- [ ] **Step 3: Implement stratified paired bootstrap**

Within each seed and domain, align on method_id::video_id and verify identical
labels. Resample real videos and each fake method stratum with replacement.
Apply the same sampled keys to baseline and E2, compute the seed metric
difference, then average the three seed differences in each replicate.

~~~python
for bootstrap_index in range(n_resamples):
    seed_deltas = []
    for experiment_seed in (42, 7, 2048):
        indices = stratified_video_resample(aligned[experiment_seed], rng)
        seed_deltas.append(
            metric(candidate[indices]) - metric(baseline[indices])
        )
    bootstrap_delta[bootstrap_index] = statistics.fmean(seed_deltas)
~~~

The G1 one-sided lower bound is the fifth percentile. G6 uses the 2.5th
percentile of meaningful minus meaningless degradation.

Represent outputs with immutable records so gate fields cannot drift between
the library and CLI:

~~~python
@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    n_resamples: int
    n_videos: int


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    point_passed: bool
    ci_passed: bool | None
    evidence: dict[str, float | int | str]


@dataclass(frozen=True)
class GateReport:
    verdict: str
    gates: dict[str, GateResult]
~~~

- [ ] **Step 4: Encode all immutable gate rules**

Gates are computed exactly as follows:

- G1: each of FF++ and CDF has mean Fixed-H1 AUROC delta at least -0.01
  and one-sided lower bound at least -0.01;
- G2: each domain has macro seen-minus-heldout drop at most 0.02 and every
  task drop at most 0.05;
- G3: maximum permutation error at most 1e-6 and consistency exactly 1.0;
- G4: held-out T2 balanced accuracy at least 0.80 and polarity consistency
  at least 0.95 in each primary domain;
- G5: held-out T4 balanced accuracy at least 0.65, true-family ordering at
  least 0.80, and T5 macro balanced accuracy at least 0.60 in each primary
  domain;
- G6: meaningful-minus-meaningless candidate score and T2/T4 question score
  are each at least 0.10 in each primary domain, with paired 95 percent lower
  bounds greater than zero.

Any point failure yields not_supported. With no point failures, any required
CI failure yields insufficient_evidence. Only all passes yield supported.

- [ ] **Step 5: Implement the artifact-loading CLI**

assess_gates.py requires all three seed directories, all seven conditions, and
three matching H1 seed prediction directories. It writes:

~~~text
results/summary/seed_metrics.csv
results/summary/bootstrap_intervals.csv
results/summary/gate_status.json
results/summary/gate_status.csv
reports/e2_validation_result.md
~~~

The Markdown report prints exact thresholds, point estimates, confidence
bounds, per-method diagnostics, calibration diagnostics, and the final verdict.

- [ ] **Step 6: Run gate and existing comparison tests**

Run:

~~~bash
pytest -q tests/test_e2_gates.py tests/test_model_comparison.py
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit adjudication**

~~~bash
git add src/evaluation/e2_gates.py tests/test_e2_gates.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py
git commit -m "feat: preregister E2 gate adjudication"
~~~

---

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

- [ ] **Step 7: Commit workflow documentation**

~~~bash
git add experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py tests/test_e2_integration.py experiments/2026-09-22_semantic_visual_jev_e2/README.md
git commit -m "test: add E2 preflight and GPU smoke workflow"
~~~

---

### Task 9: Run the locked three-seed experiment and issue the verdict

**Files:**
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_42/
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_7/
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_2048/
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1/
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation/
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/results/summary/
- Create during execution: experiments/2026-09-22_semantic_visual_jev_e2/reports/e2_validation_result.md

**Interfaces:**
- Consumes: locked code/config from Tasks 1 through 8 and completed H1 checkpoints
- Produces: all required outputs in spec Section 15 and the final E2 verdict

- [ ] **Step 1: Record the locked execution state**

Run:

~~~bash
cd /home/z/project/deepfake_detection
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --output experiments/2026-09-22_semantic_visual_jev_e2/results/preflight.json --lock-output experiments/2026-09-22_semantic_visual_jev_e2/results/execution_manifest.txt
~~~

Expected: both files are created atomically only after every lock and preflight
check passes.

- [ ] **Step 2: Train seed 42**

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 42
~~~

Expected: a non-smoke best checkpoint selected by FF++ validation
seen-language CE only.

- [ ] **Step 3: Train seeds 7 and 2048 without changing configuration**

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 7
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 2048
~~~

Expected: all three checkpoints carry the same config and language hashes and
different declared seeds.

- [ ] **Step 4: Recreate missing H1 per-video predictions for all seeds**

Seed 42 already has cached predictions, but rerun all three through the same
current evaluator so the G1 input contract is uniform:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-21_visual_jev_h1/scripts/evaluate_model.py --config experiments/2026-09-21_visual_jev_h1/configs/visual_jev_s42.yaml --checkpoint experiments/2026-09-21_visual_jev_h1/results/training/seed_42/20260921_200626/output/best_model.pth --output-dir experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1/seed_42 --model-id h1_jev_seed42 --domains ffpp cdf

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-21_visual_jev_h1/scripts/evaluate_model.py --config experiments/2026-09-21_visual_jev_h1/configs/visual_jev_s7.yaml --checkpoint experiments/2026-09-21_visual_jev_h1/results/training/seed_7/20260921_203731/output/best_model.pth --output-dir experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1/seed_7 --model-id h1_jev_seed7 --domains ffpp cdf

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-21_visual_jev_h1/scripts/evaluate_model.py --config experiments/2026-09-21_visual_jev_h1/configs/visual_jev_s2048.yaml --checkpoint experiments/2026-09-21_visual_jev_h1/results/training/seed_2048/20260921_203738/output/best_model.pth --output-dir experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1/seed_2048 --model-id h1_jev_seed2048 --domains ffpp cdf
~~~

- [ ] **Step 5: Evaluate every E2 seed and required condition**

Run the three explicit commands:

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 42 --checkpoint experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_42/best_model.pth --domains ffpp cdf dfd --conditions fixed_h1 seen heldout permutation counterfactual meaningless_candidates meaningless_questions --output-dir experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation/seed_42

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 7 --checkpoint experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_7/best_model.pth --domains ffpp cdf dfd --conditions fixed_h1 seen heldout permutation counterfactual meaningless_candidates meaningless_questions --output-dir experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation/seed_7

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 2048 --checkpoint experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_2048/best_model.pth --domains ffpp cdf dfd --conditions fixed_h1 seen heldout permutation counterfactual meaningless_candidates meaningless_questions --output-dir experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation/seed_2048
~~~

The script itself enforces that DFD produces only Fixed-H1 and T1-T3 outputs.

- [ ] **Step 6: Compute gates and render the report**

~~~bash
python experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --evaluation-root experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation --baseline-root experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1 --output-root experiments/2026-09-22_semantic_visual_jev_e2/results/summary --report experiments/2026-09-22_semantic_visual_jev_e2/reports/e2_validation_result.md
~~~

Expected: gate_status.json contains G1 through G6, every threshold and
uncertainty field, plus exactly one verdict.

- [ ] **Step 7: Perform the completion audit**

Run:

~~~bash
pytest -q
python experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --output experiments/2026-09-22_semantic_visual_jev_e2/results/final_preflight.json
python -m json.tool experiments/2026-09-22_semantic_visual_jev_e2/results/summary/gate_status.json
~~~

Then verify:

- three reportable checkpoints exist and are readable;
- all requested conditions exist for FF++ and CDF;
- DFD contains only permitted tasks;
- H1 and E2 G1 video-key sets match for every seed/domain;
- every gate cites its source files and sample/video counts;
- the Markdown verdict matches gate_status.json;
- no target or held-out metric appears in any training checkpoint-selection
  history.

- [ ] **Step 8: Commit only compact reproducibility artifacts**

Do not add checkpoints, frame caches, or large prediction files. Add the
report, compact summaries, configuration hashes, and updated README:

~~~bash
git add experiments/2026-09-22_semantic_visual_jev_e2/reports/e2_validation_result.md experiments/2026-09-22_semantic_visual_jev_e2/results/summary experiments/2026-09-22_semantic_visual_jev_e2/results/execution_manifest.txt experiments/2026-09-22_semantic_visual_jev_e2/README.md
git commit -m "results: report E2 semantic Visual-Jev verdict"
~~~
