"""Run all 66 E3 groups after all formal training launchers have exited zero."""
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
    training = json.loads((PHASE / 'results/training_logs/series_status.json').read_text())
    if not training.get('complete') or [r['seed'] for r in training['runs']] != [42, 7, 2048] or any(r['exit_code'] != 0 for r in training['runs']):
        raise ValueError('three-seed training must complete successfully before formal evaluation')
    output = PHASE / 'results/evaluation_logs'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'series_status.json'
    if path.exists():
        raise FileExistsError('preserving previous evaluation series; inspect it before recovery')
    status = {'supervisor_pid': os.getpid(), 'complete': False, 'runs': []}
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(path)
    save()
    for seed in (42, 7, 2048):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   str(PHASE / 'scripts/evaluate_candidate.py'), '--seed', str(seed)]
        log = output / f'seed_{seed}.log'
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            run = {'seed': seed, 'pid': child.pid, 'command': command, 'log': str(log),
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': None}
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
