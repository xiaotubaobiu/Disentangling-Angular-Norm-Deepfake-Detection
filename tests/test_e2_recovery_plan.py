from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments/2026-09-22_semantic_visual_jev_e2/scripts/primary_v2_recovery_supervisor.py"
)


def test_recovery_plan_covers_only_missing_primary_groups(tmp_path):
    """Resume the twenty missing groups without overwriting the twenty-two completed groups."""
    for seed in (42, 7):
        for condition in ("fixed_h1", "seen", "heldout", "permutation", "counterfactual"):
            for domain in ("ffpp", "cdf"):
                path = tmp_path / f"seed_{seed}" / condition / f"{domain}_metrics.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]")
        path = tmp_path / f"seed_{seed}/meaningless_candidates/ffpp_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")

    spec = importlib.util.spec_from_file_location("e2_recovery_supervisor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate_recovery_plan(tmp_path) == 20

    collision = tmp_path / "seed_7/meaningless_candidates/cdf_metrics.json"
    collision.write_text("[]")
    with pytest.raises(ValueError, match="already exists"):
        module.validate_recovery_plan(tmp_path)
