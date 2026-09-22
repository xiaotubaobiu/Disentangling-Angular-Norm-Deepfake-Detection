"""Frozen semantic-language manifest parsing and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SEMANTIC_TO_ID = {
    "real": 0,
    "fake": 1,
    "no": 2,
    "yes": 3,
    "camera": 4,
    "swap": 5,
    "reenact": 6,
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


@dataclass(frozen=True)
class CandidateSet:
    name: str
    options: tuple[TextEntry, ...]


@dataclass(frozen=True)
class TaskLanguage:
    task_id: str
    split: str
    questions: tuple[TextEntry, ...]
    candidate_sets: tuple[CandidateSet, ...]


class LanguageBank:
    def __init__(self, raw: dict[str, Any], text_entries: tuple[TextEntry, ...]):
        self.raw = raw
        self.text_entries = text_entries
        self._by_name = {entry.name: entry for entry in text_entries}
        self._by_id = {entry.text_id: entry for entry in text_entries}
        self.tasks = raw.get("tasks", {})
        self._task_languages = _build_task_languages(raw, self._by_name)

    def text_id(self, name: str) -> int:
        return self._by_name[name].text_id

    def text(self, name_or_id: str | int) -> str:
        if isinstance(name_or_id, str):
            return self._by_name[name_or_id].text
        return self._by_id[name_or_id].text

    def entries(self, split: str, kind: str | None = None) -> tuple[TextEntry, ...]:
        return tuple(
            entry for entry in self.text_entries
            if entry.split == split and (kind is None or entry.kind == kind)
        )

    def task(self, task_id: str, split: str) -> TaskLanguage:
        return self._task_languages[(task_id, split)]


def load_language_bank(path: str | Path) -> LanguageBank:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a mapping")
    entries = _collect_and_validate_entries(raw)
    return LanguageBank(raw=raw, text_entries=entries)


def _collect_and_validate_entries(raw: dict[str, Any]) -> tuple[TextEntry, ...]:
    records = list(_walk_text_records(raw))
    names = [record["id"] for record in records]
    texts = [record["text"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate symbolic text id")
    if len(texts) != len(set(texts)):
        raise ValueError("duplicate full text string")
    for record in records:
        semantic = _semantic(record.get("semantic"))
        yes_means = record.get("yes_means")
        if semantic is not None and semantic not in SEMANTIC_TO_ID:
            raise ValueError(f"invalid semantic: {semantic}")
        if yes_means is not None and yes_means not in {"real", "fake", "swap", "reenact"}:
            raise ValueError(f"invalid yes_means semantic: {yes_means}")
    _validate_schema(raw)
    train = {record["text"] for record in records if record["split"] == "train"}
    heldout = {record["text"] for record in records if record["split"] == "heldout"}
    if train & heldout:
        raise ValueError("train and heldout text strings overlap")
    ordered = sorted(records, key=lambda record: record["id"])
    return tuple(
        TextEntry(
            name=record["id"], text_id=index, text=record["text"],
            split=record["split"], kind=record["kind"],
            semantic=_semantic(record.get("semantic")), yes_means=record.get("yes_means"),
        )
        for index, record in enumerate(ordered)
    )


def _walk_text_records(value: Any):
    if isinstance(value, dict):
        if {"id", "split", "kind", "text"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from _walk_text_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_text_records(child)


def _semantic(value: Any) -> str | None:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return value


def _validate_schema(raw: dict[str, Any]) -> None:
    answer_sets = {
        item["id"]: item
        for items in raw.get("answer_sets", {}).values()
        for item in items
    }
    for name, answer_set in answer_sets.items():
        options = answer_set.get("options", [])
        if len(options) != 2 or {_semantic(option.get("semantic")) for option in options} != {"yes", "no"}:
            raise ValueError(f"non-binary answer set: {name}")
    for task_id, task in raw.get("tasks", {}).items():
        for split, language in task.items():
            for reference in language.get("answer_sets", []):
                if reference not in answer_sets:
                    raise ValueError(f"unknown task reference: {reference}")
            candidate_sets = language.get("candidate_sets", [])
            if task_id == "T5":
                for candidate_set in candidate_sets:
                    options = candidate_set.get("options", [])
                    if len(options) != 3 or {_semantic(item.get("semantic")) for item in options} != {"camera", "swap", "reenact"}:
                        raise ValueError(f"non-ternary T5 set: {candidate_set.get('id')}")
    controls = raw.get("controls", {}).get("candidate_sets", {})
    if len(controls.get("binary", [])) != 2:
        raise ValueError("non-binary control set")
    if len(controls.get("ternary", [])) != 3:
        raise ValueError("non-ternary control set")


def _build_task_languages(raw: dict[str, Any], by_name: dict[str, TextEntry]) -> dict[tuple[str, str], TaskLanguage]:
    answer_sets = {
        item["id"]: item["options"]
        for items in raw.get("answer_sets", {}).values()
        for item in items
    }
    result = {}
    for task_id, task in raw.get("tasks", {}).items():
        for split, language in task.items():
            candidate_sets = [
                CandidateSet(item["id"], tuple(by_name[option["id"]] for option in item["options"]))
                for item in language.get("candidate_sets", [])
            ]
            candidate_sets.extend(
                CandidateSet(name, tuple(by_name[option["id"]] for option in answer_sets[name]))
                for name in language.get("answer_sets", [])
            )
            result[(task_id, split)] = TaskLanguage(
                task_id=task_id,
                split=split,
                questions=tuple(by_name[item["id"]] for item in language.get("questions", [])),
                candidate_sets=tuple(candidate_sets),
            )
    return result
