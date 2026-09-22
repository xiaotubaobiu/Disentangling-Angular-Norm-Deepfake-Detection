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
