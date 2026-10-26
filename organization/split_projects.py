"""One-time, non-destructive split of the local research workspace.

Run from any cwd. Refuses existing destinations; never overwrites source files.
The original working tree (including untracked artifacts) is the source.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD_PHASES = (
    "2026-05-05_paper_feature_geometry", "2026-07-21_frame_midterm_diagnosis",
    "2026-07-23_four_questions", "2026-07-27_pooling_saturation",
    "2026-07-30_threshold_transfer",
)
NEW_PHASES = ("2026-09-21_visual_jev_h1", "2026-09-22_semantic_visual_jev_e2")
NEW_SOURCE = {
    "data/semantic_dataset.py", "data/semantic_language.py",
    "engine/semantic_trainer.py", "models/clip_visual_jev.py",
    "models/clip_semantic_jev.py", "evaluation/e2_feature_cache.py",
    "evaluation/e2_gates.py", "evaluation/e2_provenance.py",
    "evaluation/semantic_controls.py", "evaluation/semantic_eval.py",
    "evaluation/model_comparison.py",
}
OLD_TESTS = {"test_audit_fixes.py", "test_infer_modes_length.py",
             "test_video_aggregation.py", "test_stage2_layout.py"}
SHARED_TESTS = {"test_sample_eval.py", "test_losses.py"}
SKIP = {"__pycache__", ".pytest_cache", ".DS_Store"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def files(directory):
    return sorted(p for p in directory.rglob("*") if p.is_file()
                  and not SKIP.intersection(p.parts) and not p.suffix == ".pyc")


def remove_nodes(path, predicate):
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    spans = sorted({(n.lineno - 1, n.end_lineno) for n in ast.walk(ast.parse(source))
                    if predicate(n)}, reverse=True)
    for start, end in spans:
        del lines[start:end]
    path.write_text("".join(lines))
    ast.parse(path.read_text())


def main():
    destinations = {name: ROOT / "projects" / name for name in ("feature_geometry", "visual_jev")}
    if any(path.exists() for path in destinations.values()):
        raise SystemExit("Destination exists; inspect the existing migration instead of rerunning.")
    source_hashes = {}
    imported = {}

    def copy(source, destination, entries):
        if source.is_symlink():
            raise ValueError(f"Refusing implicit symlink dependency: {source}")
        rel = source.relative_to(ROOT).as_posix()
        digest = source_hashes.setdefault(rel, {"sha256": sha(source), "bytes": source.stat().st_size})
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != digest["sha256"] or source.stat().st_ino == destination.stat().st_ino:
            raise ValueError(f"Copy integrity/isolation failed: {source}")
        entries.append({"source": rel, "path": destination.relative_to(project).as_posix(), **digest})

    for name, project in destinations.items():
        entries = []
        phases = OLD_PHASES if name == "feature_geometry" else NEW_PHASES
        for source in files(ROOT / "src"):
            rel = source.relative_to(ROOT / "src")
            is_old = rel.parts[0] in {"diagnosis", "study", "study_pooling", "study_threshold"} or rel.as_posix() == "evaluation/norm_correction.py"
            if (name == "feature_geometry" and rel.as_posix() in NEW_SOURCE) or (name == "visual_jev" and is_old):
                continue
            copy(source, project / "src" / rel, entries)
        for source in files(ROOT / "configs"):
            copy(source, project / source.relative_to(ROOT), entries)
        for phase in phases:
            print(f"Copying {name}/{phase}", flush=True)
            for source in files(ROOT / "experiments" / phase):
                copy(source, project / source.relative_to(ROOT), entries)
        for source in files(ROOT / "tests"):
            if source.suffix != ".py":
                continue
            old = source.name in OLD_TESTS
            shared = source.name in SHARED_TESTS
            if shared or (name == "feature_geometry" and old) or (name == "visual_jev" and not old):
                copy(source, project / source.relative_to(ROOT), entries)
        for source in files(ROOT / "docs"):
            if ("superpowers" in source.parts) == (name == "visual_jev"):
                copy(source, project / source.relative_to(ROOT), entries)
        if name == "feature_geometry":
            copy(ROOT / "README.md", project / "docs/legacy_workspace_readme.md", entries)
            copy(ROOT / "experiments/README.md", project / "experiments/README.md", entries)
            copy(ROOT / "reports/threshold_transfer_final_report.md", project / "reports/threshold_transfer_final_report.md", entries)
            remove_nodes(project / "src/data/builders.py", lambda n:
                isinstance(n, ast.ImportFrom) and n.module == "data.semantic_dataset"
                or isinstance(n, ast.FunctionDef) and (n.name.startswith("build_semantic_") or n.name == "_build_semantic_suite_loader"))
            remove_nodes(project / "src/models/factory.py", lambda n:
                isinstance(n, ast.ImportFrom) and n.module in {"models.clip_visual_jev", "models.clip_semantic_jev"}
                or isinstance(n, ast.If) and ast.get_source_segment((project / "src/models/factory.py").read_text(), n.test) in {'name == "clip_visual_jev"', 'name == "clip_semantic_jev"'})
        else:
            baseline = ROOT / "experiments/2026-05-05_paper_feature_geometry/results/checkpoints/clip/raw/seed_42__20260505_130308"
            for source in files(baseline):
                copy(source, project / source.relative_to(ROOT), entries)
            copy(ROOT / "experiments/2026-05-05_paper_feature_geometry/scripts/train.py", project / "scripts/train_h1.py", entries)
            train = project / "scripts/train_h1.py"
            train.write_text(train.read_text().replace("Path(__file__).resolve().parents[3]", "Path(__file__).resolve().parents[1]"))
            readme = project / "experiments/2026-09-21_visual_jev_h1/README.md"
            readme.write_text(readme.read_text().replace("experiments/2026-05-05_paper_feature_geometry/scripts/train.py", "scripts/train_h1.py"))
        for source in (ROOT / "requirements.txt", ROOT / "environment.yml", ROOT / "pyproject.toml"):
            copy(source, project / "provenance/original_setup" / source.name, entries)
        imported[name] = entries
        dump(project / "provenance/import_manifest.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "original_root": str(ROOT),
            "source_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_is_dirty": True, "copied_and_hash_verified": len(entries),
            "note": "Original disk bytes, including untracked results; hashes describe imports before documented adaptations.",
            "files": entries,
        })
        print(f"Verified {len(entries)} independent file copies for {name}", flush=True)
    archive = ROOT / "organization/legacy_navigation"
    archive.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "README.md", ROOT / "docs/ARCHITECTURE.md"):
        shutil.copy2(source, archive / source.name)
    for source in files(ROOT / "reports"):
        shutil.copy2(source, archive / source.name)
    dump(ROOT / "organization/source_inventory.json", source_hashes)
    common = []
    old, new = destinations.values()
    for path in files(old / "src"):
        rel = path.relative_to(old)
        other = new / rel
        if other.is_file() and sha(path) == sha(other):
            common.append({"path": rel.as_posix(), "sha256": sha(path)})
    shared = {"version": "2026-09-23", "runtime_dependency": False,
              "projects": list(destinations), "files": common,
              "intentional_divergence": ["src/data/builders.py", "src/models/factory.py"]}
    dump(ROOT / "shared/common_code.json", shared)
    tested = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    for project in destinations.values():
        dump(project / "provenance/common_code.json", shared)
        dump(project / "provenance/environment_observed.json", {"python": sys.version, "packages": tested,
             "note": "Observed host environment; a record, not a clean-install lockfile."})
    print("Split copies complete; configure entrypoints and replay before declaring success.", flush=True)


if __name__ == "__main__":
    main()
