from __future__ import annotations

from datetime import timedelta
import importlib.util
from pathlib import Path

from torch.distributed import Backend
import torch.distributed.distributed_c10d as c10d


WRAPPER = (
    Path(__file__).parents[1]
    / "experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluation_runtime_timeout.py"
)


def test_evaluation_wrapper_extends_nccl_timeout_before_evaluator(monkeypatch):
    """Recovery runs must survive rank-zero CDF output processing past ten minutes."""
    spec = importlib.util.spec_from_file_location("e2_runtime_wrapper", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_timeout = c10d.default_pg_nccl_timeout
    observed = []

    def evaluator_probe(path, *, run_name):
        observed.append((path, run_name, c10d._get_default_timeout(Backend.NCCL)))

    monkeypatch.setattr(module.runpy, "run_path", evaluator_probe)
    try:
        module.main()
        assert observed == [(str(WRAPPER.with_name("evaluate_semantic.py")), "__main__", timedelta(hours=1))]
    finally:
        c10d.default_pg_nccl_timeout = original_timeout
