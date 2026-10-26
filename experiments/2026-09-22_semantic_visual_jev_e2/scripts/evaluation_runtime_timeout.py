"""Run the unchanged E2 evaluator with a longer NCCL collective timeout.

Rank zero serializes the large CDF control result while other ranks wait at a
status broadcast. The PyTorch NCCL default of ten minutes is too short for
that CPU-only phase. This wrapper changes only the process-group timeout.
"""

from datetime import timedelta
from pathlib import Path
import runpy
import sys

from torch.distributed import Backend
import torch.distributed.distributed_c10d as c10d


def main() -> None:
    c10d.default_pg_nccl_timeout = timedelta(hours=1)
    if c10d._get_default_timeout(Backend.NCCL) != timedelta(hours=1):
        raise RuntimeError("NCCL collective timeout override was not applied")
    print("E2 recovery NCCL collective timeout: 3600 seconds", file=sys.stderr, flush=True)
    runpy.run_path(str(Path(__file__).with_name("evaluate_semantic.py")), run_name="__main__")


if __name__ == "__main__":
    main()
