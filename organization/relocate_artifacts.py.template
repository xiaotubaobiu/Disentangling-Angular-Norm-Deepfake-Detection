"""Rebind copied E2 path metadata to this project after verifying local bytes.

Original metadata is retained in provenance/path_metadata_original. This is a
relocation record, not a new experiment or a changed scientific fingerprint.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "experiments/2026-09-22_semantic_visual_jev_e2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relocate():
    imported = json.loads((ROOT / "provenance/import_manifest.json").read_text())
    journal_path = ROOT / "provenance/path_relocations.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else []
    roots = {imported["original_root"], *(entry["project_root"] for entry in journal)}
    roots.discard(str(ROOT))

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            # The standalone project can live inside the original workspace.
            # Already-local paths must not match that shorter ancestor again.
            if value == str(ROOT) or value.startswith(str(ROOT) + "/"):
                return value
            for old in sorted(roots, key=len, reverse=True):
                if value == old or value.startswith(old + "/"):
                    return str(ROOT) + value[len(old):]
        return value

    # Validate every referenced local input before changing any path metadata.
    records = [E2 / "results/execution_manifest.json", E2 / "results/preflight.json"]
    for manifest_path in records:
        manifest = replace(json.loads(manifest_path.read_text()))
        for checkpoint in manifest["checkpoints"].values():
            path = Path(checkpoint["path"])
            if not path.is_relative_to(ROOT) or sha(path) != checkpoint["sha256"]:
                raise ValueError(f"Copied H1 checkpoint mismatch: {path}")
    for seed in (42, 7, 2048):
        record = E2 / f"results/evaluation_primary_v2/seed_{seed}/run_provenance.json"
        manifest = replace(json.loads(record.read_text()))
        checkpoint = E2 / f"results/training/seed_{seed}/best_model.pth"
        if Path(manifest["checkpoint_path"]) != checkpoint or sha(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError(f"Copied E2 checkpoint mismatch: {checkpoint}")
        cache = E2 / "results/feature_cache"
        if Path(manifest["feature_cache_root"]) != cache:
            raise ValueError("Copied feature cache path mismatch")
        for domain, expected in manifest["feature_cache_manifest_sha256"].items():
            if sha(cache / domain / "manifest.json") != expected:
                raise ValueError(f"Copied feature cache manifest mismatch: {domain}")
        records.extend((record, E2 / f"results/baseline_h1/seed_{seed}/evaluation.json"))
    changes = []
    for path in records:
        before = json.loads(path.read_text())
        after = replace(before)
        if before == after:
            continue
        relative = path.relative_to(ROOT)
        backup = ROOT / "provenance/path_metadata_original" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        old_sha = sha(path)
        path.write_text(json.dumps(after, indent=2, ensure_ascii=False) + "\n")
        changes.append({"path": str(relative), "before_sha256": old_sha, "after_sha256": sha(path)})
    if changes:
        journal.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "project_root": str(ROOT), "previous_roots": sorted(roots),
                        "changed_files": changes})
        journal_path.write_text(json.dumps(journal, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"relocated_metadata_files": len(changes), "project_root": str(ROOT)}))


if __name__ == "__main__":
    relocate()
