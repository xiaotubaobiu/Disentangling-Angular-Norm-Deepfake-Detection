"""Read-only source preservation, independent-copy, and common-code audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
H1 = "experiments/2026-09-21_visual_jev_h1"
E2 = "experiments/2026-09-22_semantic_visual_jev_e2"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    originals = json.loads((ROOT / "organization/source_inventory.json").read_text())
    errors = []
    source_preserved = 0
    navigation = {"README.md": "organization/legacy_navigation/README.md",
                  "docs/ARCHITECTURE.md": "organization/legacy_navigation/ARCHITECTURE.md"}
    for relative, entry in originals.items():
        path = ROOT / navigation.get(relative, relative)
        if sha(path) != entry["sha256"]:
            errors.append(f"Original changed: {relative}")
        else:
            source_preserved += 1
    result = {"source_files_preserved": source_preserved,
              "navigation_archived_before_replacement": navigation, "projects": {}}
    for name in ("feature_geometry", "visual_jev"):
        root = ROOT / "projects" / name
        manifest = json.loads((root / "provenance/import_manifest.json").read_text())
        allowed = ({"src/data/builders.py", "src/models/factory.py", "docs/legacy_workspace_readme.md"}
                   if name == "feature_geometry" else {"scripts/train_h1.py", f"{H1}/README.md", f"{E2}/README.md",
                        f"{E2}/results/execution_manifest.json", f"{E2}/results/preflight.json",
                        *(f"{E2}/results/evaluation_primary_v2/seed_{s}/run_provenance.json" for s in (42, 7, 2048)),
                        *(f"{E2}/results/baseline_h1/seed_{s}/evaluation.json" for s in (42, 7, 2048))})
        changed, identical = [], 0
        for entry in manifest["files"]:
            relative = entry["path"]
            path = root / relative
            source = ROOT / entry["source"]
            if path.is_symlink() or path.stat().st_ino == source.stat().st_ino:
                errors.append(f"Non-independent file: {path}")
            if sha(path) != entry["sha256"]:
                changed.append(relative)
                if relative not in allowed:
                    errors.append(f"Unexplained changed copy: {path}")
            else:
                identical += 1
        common = json.loads((root / "provenance/common_code.json").read_text())
        for entry in common["files"]:
            if sha(root / entry["path"]) != entry["sha256"]:
                errors.append(f"Common source drift: {name}/{entry['path']}")
        links = 0
        for path in root.rglob("*.md"):
            if {".venv", "build", "dist", "runs", "results"}.intersection(path.parts):
                continue
            # Match Markdown links, not prose such as mathematical arrays.
            for target in re.findall(r'!?\[[^\]\n]+\]\(([^)\n]+)\)', path.read_text()):
                target = target.split("#")[0].strip("<>")
                if not target or "://" in target or " " in target or target.startswith("mailto:"):
                    continue
                links += 1
                if not (path.parent / target).exists():
                    errors.append(f"Broken document link: {path}: {target}")
        result["projects"][name] = {"imported_files": len(manifest["files"]),
            "byte_identical_imports": identical, "documented_adaptations": changed,
            "common_source_files": len(common["files"]), "document_links_checked": links}
    spec = importlib.util.spec_from_file_location("split_source_fingerprint", ROOT / "projects/visual_jev/src/evaluation/e2_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = module.evaluation_source_bundle_sha256(ROOT)
    after = module.evaluation_source_bundle_sha256(ROOT / "projects/visual_jev")
    if before != after:
        errors.append("E2 scientific source fingerprint changed")
    result.update(e2_source_fingerprint=after, e2_source_unchanged=before == after,
                  errors=errors, status="pass" if not errors else "fail")
    (ROOT / "organization/integrity_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
