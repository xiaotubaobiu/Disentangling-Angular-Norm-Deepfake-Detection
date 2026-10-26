#!/usr/bin/env python3
"""Stage runner — the whole threshold-transfer pipeline, in order.

Inputs : configs/threshold_study.yaml (plus whatever each stage needs).
Outputs: everything under results/, figures/, reports/ + results/logs/run_all*.log
GPU    : only if `--extract-missing` is forwarded to stage 01.
Target-domain labels: stages 04/05 only (see their headers).

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/run_all.py --dry-run
    python experiments/2026-07-30_threshold_transfer/scripts/run_all.py
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]

STAGES = [
    ("00 audit inputs", "00_audit_inputs.py", []),
    ("01 FF++ validation", "01_prepare_ffpp_validation.py", ["dry", "extract"]),
    ("02 fit calibrators", "02_fit_calibrators.py", ["dry"]),
    ("03 select thresholds", "03_select_source_thresholds.py", ["dry"]),
    ("04 evaluate transfer", "04_evaluate_transfer.py", ["dry", "skip_bootstrap"]),
    ("05 score shift", "05_analyze_score_shift.py", ["dry"]),
    ("06 report", "06_generate_report.py", ["dry"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="one backbone / one seed / two methods, fewer bootstrap iterations")
    ap.add_argument("--extract-missing", action="store_true",
                    help="let stage 01 re-infer FF++ val for uncached checkpoints (GPU 0-5)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--config", default=None,
                    help="alternative threshold_study.yaml (forwarded to every stage)")
    ap.add_argument("--from-stage", default=None, help="resume at this script name")
    args = ap.parse_args()

    log_dir = (Path(args.config).resolve().parents[1] / "results" / "logs"
               if args.config else PHASE / "results" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "dry" if args.dry_run else "full"
    log_path = log_dir / f"run_all_{tag}.log"
    start = datetime.datetime.now()
    header = (f"run_all.py ({tag}) started {start.isoformat(timespec='seconds')}\n"
              f"PHASE={PHASE}\nROOT={ROOT}\n")
    print(header, end="")
    log_path.write_text(header)

    started = args.from_stage is None
    for name, script, flags in STAGES:
        if not started:
            if script == args.from_stage:
                started = True
            else:
                continue
        cmd = [sys.executable, "-u", str(PHASE / "scripts" / script)]
        if args.config:
            cmd += ["--config", str(Path(args.config).resolve())]
        if args.dry_run and "dry" in flags:
            cmd.append("--dry-run")
        if args.extract_missing and "extract" in flags:
            cmd += ["--extract-missing", "--gpu", str(args.gpu)]
        if args.skip_bootstrap and "skip_bootstrap" in flags:
            cmd.append("--skip-bootstrap")

        banner = f"\n=== [{name}] {' '.join(cmd[2:])} ===\n"
        print(banner, end="")
        with log_path.open("a") as fh:
            fh.write(banner)
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            with log_path.open("a") as fh:
                fh.write(line)
        rc = proc.wait()
        with log_path.open("a") as fh:
            fh.write(f"--- [{name}] rc={rc} ---\n")
        if rc != 0:
            msg = f"\n!! [{name}] failed with rc={rc}; stopping.\n"
            print(msg, end="")
            with log_path.open("a") as fh:
                fh.write(msg)
            return rc

    done = f"\nrun_all.py ({tag}) finished in {datetime.datetime.now() - start}\n"
    print(done, end="")
    with log_path.open("a") as fh:
        fh.write(done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
