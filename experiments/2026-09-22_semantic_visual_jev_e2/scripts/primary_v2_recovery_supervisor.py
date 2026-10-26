"""Resume only the missing FF++/CDF E2 groups after the NCCL timeout.

This operational launcher leaves completed predictions untouched. Each child
has its exact command, wrapper hash, console/rank logs, and real wait() exit
status recorded independently of the scientific evaluation source fingerprint.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
E2 = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = E2 / "results/evaluation_primary_v2"
LOG_ROOT = E2 / "results/evaluation_logs"
WRAPPER = Path(__file__).with_name("evaluation_runtime_timeout.py")
CONFIG = E2 / "configs/semantic_jev.yaml"
CACHE = E2 / "results/feature_cache"
SOURCE_SHA256 = "0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7"
CONDITIONS = ("fixed_h1", "seen", "heldout", "permutation", "counterfactual",
              "meaningless_candidates", "meaningless_questions")
DOMAINS = ("ffpp", "cdf")
SEEDS = (42, 7, 2048)
PLANS = {
    7: (("meaningless_candidates", ("cdf",)),
        ("meaningless_questions", DOMAINS)),
    2048: ((CONDITIONS, DOMAINS),),
    42: (("meaningless_candidates", ("cdf",)),
         ("meaningless_questions", DOMAINS)),
}
ORDER = (7, 2048, 42)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def groups_for_step(seed: int, step: tuple) -> tuple:
    conditions, domains = step
    conditions = (conditions,) if isinstance(conditions, str) else conditions
    return tuple((seed, condition, domain) for condition in conditions for domain in domains)


def recovery_groups() -> tuple:
    return tuple(group for seed in ORDER for step in PLANS[seed]
                 for group in groups_for_step(seed, step))


def validate_recovery_plan(root: Path) -> int:
    """Require an exact, disjoint partition of 22 old and 20 recovery groups."""
    target = recovery_groups()
    if len(target) != 20 or len(set(target)) != 20:
        raise ValueError("recovery plan must contain exactly 20 unique groups")
    all_groups = {(seed, condition, domain) for seed in SEEDS
                  for condition in CONDITIONS for domain in DOMAINS}
    if not set(target).issubset(all_groups):
        raise ValueError("recovery plan contains a non-primary group")
    for seed, condition, domain in sorted(all_groups):
        path = root / f"seed_{seed}" / condition / f"{domain}_metrics.json"
        if (seed, condition, domain) in target:
            if path.exists():
                raise ValueError(f"planned recovery metric already exists: {path}")
        elif not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"previously completed metric is missing: {path}")
    return len(target)


def check_scientific_source() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from evaluation.e2_provenance import evaluation_source_bundle_sha256
    actual = evaluation_source_bundle_sha256(REPO)
    if actual != SOURCE_SHA256:
        raise ValueError(f"evaluation source bundle changed: {actual}")


def required_outputs(root: Path, groups: tuple) -> list[Path]:
    paths = []
    for seed, condition, domain in groups:
        directory = root / f"seed_{seed}" / condition
        paths.append(directory / f"{domain}_metrics.json")
        if condition != "permutation":
            paths += [directory / f"{domain}_videos.csv", directory / f"{domain}_frames.npz"]
        if condition == "counterfactual":
            paths.append(directory / f"{domain}_controls.csv")
        if condition.startswith("meaningless_"):
            paths.append(directory / f"{domain}_paired_videos.csv")
    return paths


def launch(seed: int, step_index: int, run_dir: Path) -> dict:
    check_scientific_source()
    step = PLANS[seed][step_index]
    groups = groups_for_step(seed, step)
    for path in required_outputs(EVALUATION_ROOT, groups):
        if path.exists():
            raise ValueError(f"planned recovery output already exists: {path}")
    conditions, domains = step
    conditions = (conditions,) if isinstance(conditions, str) else conditions
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise ValueError("torchrun is unavailable")
    job_dir = run_dir / f"seed_{seed}_step_{step_index + 1}"
    job_dir.mkdir(parents=True, exist_ok=False)
    argv = [torchrun, "--standalone", "--nproc_per_node=4", "--log-dir",
            str(job_dir / "rank_logs"), "--redirects", "3", str(WRAPPER),
            "--config", str(CONFIG), "--seed", str(seed),
            "--checkpoint", str(E2 / f"results/training/seed_{seed}/best_model.pth"),
            "--domains", *domains, "--conditions", *conditions,
            "--feature-cache-root", str(CACHE),
            "--output-dir", str(EVALUATION_ROOT / f"seed_{seed}")]
    overrides = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    env = os.environ.copy()
    env.update(overrides)
    manifest = {
        "seed": seed, "step": step_index + 1, "groups": groups, "argv": argv,
        "cwd": str(REPO), "environment_overrides": overrides,
        "configured_per_gpu_batch": 128, "nccl_timeout_seconds": 3600,
        "runtime_wrapper_sha256": sha256(WRAPPER),
        "recovery_supervisor_sha256": sha256(Path(__file__)),
        "evaluator_sha256": sha256(WRAPPER.with_name("evaluate_semantic.py")),
        "evaluation_source_bundle_sha256": SOURCE_SHA256,
        "config_sha256": sha256(CONFIG),
        "checkpoint_sha256": sha256(E2 / f"results/training/seed_{seed}/best_model.pth"),
        "started_at_utc": now(),
    }
    atomic_json(job_dir / "command.json", manifest)
    with (job_dir / "launcher_console.log").open("w") as stream:
        process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(job_dir / "launcher_pid.json", {"pid": process.pid, "started_at_utc": now()})
    return {"process": process, "seed": seed, "step": step_index,
            "groups": groups, "job_dir": job_dir}


def finish(job: dict) -> dict:
    process = job["process"]
    rc = process.wait()
    missing = [str(path) for path in required_outputs(EVALUATION_ROOT, job["groups"])
               if not path.is_file() or path.stat().st_size == 0]
    completion = {"seed": job["seed"], "step": job["step"] + 1,
                  "groups": job["groups"], "launcher_pid": process.pid,
                  "launcher_exit_code": rc, "missing_or_empty_outputs": missing,
                  "finished_at_utc": now()}
    atomic_json(job["job_dir"] / "completion.json", completion)
    return completion


def run() -> int:
    check_scientific_source()
    validate_recovery_plan(EVALUATION_ROOT)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "primary_v2_recovery.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir = LOG_ROOT / ("primary_v2_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        run_dir.mkdir(exist_ok=False)
        atomic_json(LOG_ROOT / "current_recovery_run.json", {"run_dir": str(run_dir),
                    "supervisor_pid": os.getpid(), "supervisor_sha256": sha256(Path(__file__)),
                    "started_at_utc": now()})
        pending = deque(ORDER)
        active = {}
        completed = []
        failed = False
        try:
            while pending or active:
                for seed, job in list(active.items()):
                    if job["process"].poll() is None:
                        continue
                    result = finish(job)
                    completed.append(result)
                    del active[seed]
                    if result["launcher_exit_code"] != 0 or result["missing_or_empty_outputs"]:
                        failed = True
                    elif job["step"] + 1 < len(PLANS[seed]):
                        active[seed] = launch(seed, job["step"] + 1, run_dir)
                while not failed and len(active) < 2 and pending:
                    seed = pending.popleft()
                    active[seed] = launch(seed, 0, run_dir)
                atomic_json(run_dir / "state.json", {
                    "at_utc": now(), "active": {str(s): j["process"].pid for s, j in active.items()},
                    "pending_seeds": list(pending), "completed": completed, "failed": failed})
                if failed and not active:
                    break
                time.sleep(5)
        except BaseException:
            for job in active.values():
                if job["process"].poll() is None:
                    job["process"].terminate()
                completed.append(finish(job))
            raise
        success = not failed and not pending and validate_completed_groups()
        atomic_json(run_dir / "completion.json", {"success": success,
                    "completed": completed, "finished_at_utc": now()})
        return 0 if success else 1


def validate_completed_groups() -> bool:
    for seed, condition, domain in recovery_groups():
        if any(not path.is_file() or path.stat().st_size == 0 for path in
               required_outputs(EVALUATION_ROOT, ((seed, condition, domain),))):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        check_scientific_source()
        count = validate_recovery_plan(EVALUATION_ROOT)
        print(json.dumps({"missing_groups": count, "plan": {
            str(seed): [groups_for_step(seed, step) for step in PLANS[seed]] for seed in ORDER},
            "runtime_wrapper_sha256": sha256(WRAPPER),
            "evaluation_source_bundle_sha256": SOURCE_SHA256}, indent=2))
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
