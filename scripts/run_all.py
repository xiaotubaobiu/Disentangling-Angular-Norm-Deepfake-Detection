#!/usr/bin/env python3
# scripts/run_all.py
"""End-to-end orchestrator for the four-questions study (§7.1).

Runs every stage in order as a subprocess with cwd=ROOT, teeing combined
stdout/stderr to outputs/logs/run_all.log and stopping on the first nonzero
return code. The individual stages already ran during the study; this script
exists so the whole pipeline is reproducible from a single command.

NOTE: the full 15-checkpoint Q4 extraction (`04c_q4_run.py --full`) is
deliberately NOT in this list. Per the spec's smoke-first gate, the three
backbone smokes all came back negative (same-frame chord alignment <= mismatch
negative-control across clip/dinov2/efficientnet x ffpp/cdf, 6/6), so the
expensive full run is skipped. Only the smoke variants are orchestrated here.
"""
from __future__ import annotations
import os, subprocess, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs" / "run_all.log"

# (stage name, command) tuples. Commands are python invocations relative to ROOT.
# Steps marked CUDA get CUDA_VISIBLE_DEVICES=0 injected into their env.
STEPS = [
    ("00 schema",
     ["python", "scripts/00_inspect_project.py"]),
    ("Q1",
     ["python", "scripts/01_rank_flip_analysis.py"]),
    ("Q2",
     ["python", "scripts/02_radial_stability.py"]),
    ("Q3",
     ["python", "scripts/03_length_robustness.py"]),
    ("midterm report",
     ["python", "scripts/05_generate_report.py", "--midterm"]),
    ("Q4 pairing audit",
     ["python", "scripts/04_audit_pairs.py"]),
    ("Q4 consistency gate",
     ["env", "CUDA_VISIBLE_DEVICES=0", "python", "scripts/04b_q4_gate.py"]),
    ("Q4 smoke clip",
     ["env", "CUDA_VISIBLE_DEVICES=0", "python", "scripts/04c_q4_run.py",
      "--smoke", "--backbone", "clip"]),
    ("Q4 smoke dinov2",
     ["env", "CUDA_VISIBLE_DEVICES=0", "python", "scripts/04c_q4_run.py",
      "--smoke", "--backbone", "dinov2"]),
    ("Q4 smoke efficientnet",
     ["env", "CUDA_VISIBLE_DEVICES=0", "python", "scripts/04c_q4_run.py",
      "--smoke", "--backbone", "efficientnet"]),
    ("final report",
     ["python", "scripts/05_generate_report.py", "--final"]),
]


def _build_cmd(cmd: list[str]) -> tuple[list[str], dict]:
    """Return (argv, env). Translate `env CUDA_VISIBLE_DEVICES=0 python ...`
    into a plain argv with CUDA_VISIBLE_DEVICES=0 set in the subprocess env."""
    env = os.environ.copy()
    if cmd and cmd[0] == "env" and len(cmd) >= 3 and cmd[1].startswith("CUDA_VISIBLE_DEVICES="):
        env["CUDA_VISIBLE_DEVICES"] = cmd[1].split("=", 1)[1]
        cmd = cmd[2:]
    return cmd, env


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.datetime.now()
    header = f"run_all.py started {start.isoformat()}\nROOT={ROOT}\n"
    print(header, end="")
    with open(LOG, "w") as f:
        f.write(header)

    for name, cmd in STEPS:
        argv, env = _build_cmd(cmd)
        banner = f"\n=== [{name}] => {' '.join(argv)} ===\n"
        print(banner, end="")
        with open(LOG, "a") as f:
            f.write(banner)
        proc = subprocess.Popen(
            argv, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            with open(LOG, "a") as f:
                f.write(line)
        rc = proc.wait()
        with open(LOG, "a") as f:
            f.write(f"--- [{name}] rc={rc} ---\n")
        if rc != 0:
            err = f"\n!! [{name}] FAILED with rc={rc}; stopping.\n"
            print(err, end="")
            with open(LOG, "a") as f:
                f.write(err)
            return rc

    elapsed = datetime.datetime.now() - start
    done = f"\nrun_all.py completed in {elapsed}\n"
    print(done, end="")
    with open(LOG, "a") as f:
        f.write(done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
