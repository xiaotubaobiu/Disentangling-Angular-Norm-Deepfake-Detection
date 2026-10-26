"""Serial seed supervisor with durable child IDs and actual launcher exits."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('only GPUs 0,1,2,3 are authorized')
    output = PHASE / 'results/training_logs'
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'series_status.json'
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {'supervisor_pid': os.getpid(), 'complete': False, 'runs': []}
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)
    save()
    for seed in (42, 7, 2048):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   str(PHASE / 'scripts/train_candidate.py'), '--seed', str(seed)]
        path = output / f'seed_{seed}.log'
        with path.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            run = {'seed': seed, 'pid': child.pid, 'command': command,
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': None,
                   'log': str(path)}
            status['runs'].append(run)
            save()
            print(json.dumps(run), flush=True)
            run['exit_code'] = child.wait()
        run['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(run), flush=True)
        if run['exit_code'] != 0:
            return run['exit_code']
    status['complete'] = True
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
