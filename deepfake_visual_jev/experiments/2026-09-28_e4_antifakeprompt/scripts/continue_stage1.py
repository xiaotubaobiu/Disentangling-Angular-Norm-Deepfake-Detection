"""Finish authorized local jobs after training; no agents, no notifications."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PHASE = Path(__file__).resolve().parents[1]
PYTHON = str(PHASE / '.venv/bin/python')


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'ExecMainStatus'],
                            text=True, capture_output=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def wait_success(unit, completion):
    while True:
        state = unit_state(unit)
        if state['ActiveState'] not in ('active', 'activating', 'deactivating'):
            if state['ExecMainStatus'] != '0' or not completion.exists():
                raise RuntimeError(f'{unit} stopped without successful completion: {state}')
            return
        time.sleep(20)


def continue_arm(arm, gpu):
    unit = f'visual-jev-e4-train-{arm.lower()}-20260928.service'
    wait_success(unit, PHASE / f'results/training/{arm}/complete.json')
    for domain in ('ffpp', 'cdf'):
        cache = PHASE / f'results/visual_cache/{domain}/manifest.json'
        while not cache.exists():
            state = unit_state(f'visual-jev-e4-eval-a0-{domain}-20260928.service')
            if state['ActiveState'] not in ('active', 'activating'):
                raise RuntimeError(f'official baseline stopped before {domain} cache completed')
            time.sleep(20)
        print('starting evaluation', arm, domain, 'GPU', gpu, flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4')
        log = PHASE / f'results/logs/evaluate_{arm}_{domain}.log'
        command = [PYTHON, '-u', 'scripts/evaluate.py', '--arm', arm, '--domain', domain]
        with log.open('a') as stream:
            result = subprocess.run(command, cwd=PHASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{arm}/{domain} evaluation failed; see {log}')
        print('finished evaluation', arm, domain, flush=True)


def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(continue_arm, arm, gpu) for arm, gpu in [('A1', 0), ('A2', 1)]]
        for future in futures:
            future.result()
    for domain in ('ffpp', 'cdf'):
        wait_success(f'visual-jev-e4-eval-a0-{domain}-20260928.service',
                     PHASE / f'results/evaluation/A0/permutation/{domain}_completion.json')
    print('all evaluations finished; running audited report', flush=True)
    subprocess.run([PYTHON, '-u', 'scripts/report.py'], cwd=PHASE, check=True,
                   env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    output = PHASE / 'results/stage1_pipeline_complete.json'
    output.write_text(json.dumps({'completed': True, 'report': 'reports/stage1_report_zh.md',
                                  'next_stage_started': False}, indent=2) + '\n')
    print('STAGE1 PIPELINE COMPLETE; no further experiment launched', flush=True)


if __name__ == '__main__':
    main()
