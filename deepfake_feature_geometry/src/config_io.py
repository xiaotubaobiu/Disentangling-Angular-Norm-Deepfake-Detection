"""YAML config loading with `_base_` inheritance.

Shared by the training entrypoint and by every experiment script that needs to
rebuild a model from a checkpoint's stored config. It used to live in the
repo-root `train.py`; it moved here when the training entrypoint was filed under
`experiments/<date>_<phase>/scripts/`, so that library code never has to import
an experiment script.

`_base_` values are resolved against, in order:

1. the value as given (absolute, or relative to the current working directory)
2. the directory of the config that declared it
3. the repository root
4. the declaring config's directory, by file name only
5. the repository root `configs/` directory, by file name only

Rules 4 and 5 exist so that historical checkpoint config snapshots (which
declare paths such as ``experiments/configs/base.yaml`` from an older repository
layout) still resolve after the reorganization.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_config_path(path, config_path=None):
    candidates = [path]
    if config_path is not None:
        candidates.append(os.path.join(os.path.dirname(config_path), path))
    candidates.append(str(REPO_ROOT / path))
    name = os.path.basename(path)
    if config_path is not None:
        candidates.append(os.path.join(os.path.dirname(config_path), name))
    candidates.append(str(REPO_ROOT / "configs" / name))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def load_config(path, _visited=None):
    if _visited is None:
        _visited = set()
    resolved_path = os.path.realpath(path)
    if resolved_path in _visited:
        raise ValueError(f"Circular config inheritance: {path}")
    _visited.add(resolved_path)
    with open(resolved_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = _resolve_config_path(cfg.pop("_base_"), resolved_path)
        base = load_config(base_path, _visited)
        merged = {**base, **cfg}
        for k in set(list(base.keys()) + list(cfg.keys())):
            if k in base and k in cfg and isinstance(base[k], dict) and isinstance(cfg[k], dict):
                merged[k] = {**base[k], **cfg[k]}
        cfg = merged
    return cfg
