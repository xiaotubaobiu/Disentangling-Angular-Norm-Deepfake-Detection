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

