"""Continue the authorized evaluation when the existing training job finishes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().split()
        command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return {'start_ticks': fields[21], 'command': command, 'state': fields[2]}
    except FileNotFoundError:
        return None


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('only GPUs 0,1,2,3 are authorized')
    path = PHASE / 'results/progress/stage1_pipeline.json'
    if path.exists():
        raise FileExistsError('preserving existing pipeline state')
    training_path = PHASE / 'results/training_logs/series_status.json'
    training = json.loads(training_path.read_text())
    pid = training['supervisor_pid']
    identity = process_identity(pid)
    if not training.get('complete') and (identity is None or 'train_series.py' not in identity['command']):
        raise ValueError('training supervisor is not currently live')
    status = {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'phase': 'waiting_for_existing_training', 'training_pid': pid,
              'training_process_identity': identity, 'runs': [], 'complete': False}
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(path)
    save()
    try:
        while True:
            training = json.loads(training_path.read_text())
            if training.get('complete'):
                if [run['seed'] for run in training['runs']] != [42, 7, 2048] or any(run['exit_code'] != 0 for run in training['runs']):
                    raise RuntimeError('training completed without three successful launchers')
                break
            current = process_identity(pid)
            if current is None or current['state'] == 'Z' or current['start_ticks'] != identity['start_ticks']:
                raise RuntimeError('training supervisor stopped before complete success; no automatic restart')
            time.sleep(30)
        for phase, filename in [('evaluation', 'evaluate_series.py'), ('assessment', 'assess_candidate.py')]:
            script = PHASE / 'scripts' / filename
            log = PHASE / 'results/progress' / f'pipeline_{phase}.log'
            command = [sys.executable, str(script)]
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                run = {'phase': phase, 'pid': child.pid, 'command': command, 'log': str(log),
                       'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'exit_code': None,
                       'started_utc': datetime.now(timezone.utc).isoformat()}
                status['phase'] = phase
                status['runs'].append(run)
                save()
                print(json.dumps(run), flush=True)
                run['exit_code'] = child.wait()
            run['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
            if run['exit_code'] != 0:
                raise RuntimeError(f'{phase} exited {run["exit_code"]}; inspect its recorded log')
        status['complete'] = True
        status['phase'] = 'stage1_evidence_ready_for_report'
        save()
        return 0
    except Exception as error:
        status['phase'] = 'failed'
        status['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
