"""Verify that this project's modules resolve locally; does not load model weights."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
modules = ["config_io", "data.builders", "engine.trainers", "models.factory", "evaluation.sample_eval"]
if (ROOT / "src/study_threshold").is_dir():
    modules += ["diagnosis.diagnostics", "study.io", "study_pooling.config", "study_threshold"]
else:
    modules += ["models.clip_visual_jev", "models.clip_semantic_jev", "evaluation.e2_gates"]
for name in modules:
    importlib.import_module(name)
paths = {}
for name, module in list(sys.modules.items()):
    if name.split(".")[0] not in {"config_io", "data", "engine", "models", "losses", "evaluation", "diagnosis", "study", "study_pooling", "study_threshold"}:
        continue
    file = getattr(module, "__file__", None)
    if file and not Path(file).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(f"Non-local import: {name}: {file}")
    if file:
        paths[name] = str(Path(file).resolve().relative_to(ROOT))
print(json.dumps({"project_root": str(ROOT), "python": sys.executable,
                  "local_modules": paths}, indent=2))
