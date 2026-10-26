"""CPU replay of the frozen FF++/CDF adjudication and independent artifact audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from relocate_artifacts import relocate

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "experiments/2026-09-22_semantic_visual_jev_e2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory (must not exist)")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "runs" / f"e2_replay_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    relocate()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}

    def run(script, *arguments):
        subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=ROOT, env=env, check=True)

    run(E2 / "scripts/assess_gates.py",
        "--config", E2 / "configs/semantic_jev.yaml",
        "--evaluation-root", E2 / "results/evaluation_primary_v2",
        "--baseline-root", E2 / "results/baseline_h1",
        "--output-root", output / "summary", "--report", output / "gate_report.md")
    frozen = json.loads((E2 / "results/summary/gate_status.json").read_text())
    replay = json.loads((output / "summary/gate_status.json").read_text())
    # Paths are the only permitted difference; every metric and CI must match.
    frozen.pop("sources", None)
    replay.pop("sources", None)
    if frozen != replay:
        raise ValueError("Replayed scientific adjudication differs from frozen results")
    audit_path = E2 / "results/progress" / f"audit_relocation_{stamp}.json"
    run(E2 / "scripts/audit_final_artifacts.py", "--primary-only",
        "--evaluation-root", E2 / "results/evaluation_primary_v2", "--output", audit_path)
    shutil.copy2(audit_path, output / "artifact_audit.json")
    audit = json.loads(audit_path.read_text())
    result = {"scientific_results_identical": True, "verdict": replay["verdict"],
              "domains": audit["domains"], "groups": audit["inventory_complete_groups"],
              "audit": audit["status"], "checks": audit["checks_by_status"],
              "output": str(output)}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
