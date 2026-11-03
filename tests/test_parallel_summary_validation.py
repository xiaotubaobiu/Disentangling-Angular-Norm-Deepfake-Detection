"""Fresh acceptance rounds must preserve the original fail-closed validation."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_parallel_summary_validation.py'


def runtime(monkeypatch):
    assert SCRIPT.is_file(), 'Parallel strict-summary acceptance runtime is not implemented'
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module('run_parallel_summary_validation')


def probe(tmp_path, monkeypatch):
    """A real spawned validator reads mutable inputs; no research data is mocked."""
    name = 'acceptance_probe_' + tmp_path.name.replace('-', '_')
    (tmp_path / (name + '.py')).write_text('''
from pathlib import Path
import os
import time

def _accept(path, seed):
    path = Path(path)
    value = path.read_text()
    with path.with_suffix('.calls').open('a') as stream:
        stream.write(str(seed) + '\\n')
    path.with_suffix('.entered').write_text('entered')
    if value.startswith('barrier:'):
        other = path.parent / (value.split(':', 1)[1] + '.entered')
        deadline = time.monotonic() + 10
        while not other.exists():
            if time.monotonic() > deadline:
                raise RuntimeError('validators did not run concurrently')
            time.sleep(.01)
    if value == 'FAIL':
        raise ValueError('bad frozen input')
    return dict(seed=seed, value=value, pid=os.getpid(), cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    a.write_text('original-a'); b.write_text('original-b')
    return name, a, b, {7: (str(a), 7), 42: (str(b), 42)}


def test_acceptance_round_rereads_changed_inputs_in_real_children(tmp_path, monkeypatch):
    m = runtime(monkeypatch)
    name, a, b, arguments = probe(tmp_path, monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    with m.AcceptanceRounds(name, arguments, workers=2) as rounds:
        first = [rounds.accept(*arguments[s]) for s in [7, 42]]
        a.write_text('changed-after-initial-acceptance')
        second = [rounds.accept(*arguments[s]) for s in [7, 42]]
    assert [x['value'] for x in first] == ['original-a', 'original-b']
    assert [x['value'] for x in second] == ['changed-after-initial-acceptance', 'original-b']
    assert all(x['pid'] != os.getpid() and x['cuda'] == '' for x in first + second)
    assert a.with_suffix('.calls').read_text().splitlines() == ['7', '7']
    assert b.with_suffix('.calls').read_text().splitlines() == ['42', '42']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'


def test_all_seed_acceptors_are_submitted_before_waiting_for_the_first(tmp_path, monkeypatch):
    m = runtime(monkeypatch)
    name, a, b, arguments = probe(tmp_path, monkeypatch)
    a.write_text('barrier:b'); b.write_text('barrier:a')
    with m.AcceptanceRounds(name, arguments, workers=2) as rounds:
        results = [rounds.accept(*arguments[s]) for s in [7, 42]]
    assert len({x['pid'] for x in results}) == 2


def test_acceptor_failure_is_not_omitted_or_retried(tmp_path, monkeypatch):
    m = runtime(monkeypatch)
    name, a, b, arguments = probe(tmp_path, monkeypatch)
    b.write_text('FAIL')
    with m.AcceptanceRounds(name, arguments, workers=2) as rounds:
        assert rounds.accept(*arguments[7])['value'] == 'original-a'
        with pytest.raises(ValueError, match='bad frozen input'):
            rounds.accept(*arguments[42])
    assert b.with_suffix('.calls').read_text().splitlines() == ['42']


def test_repeated_seed_before_round_complete_is_rejected(tmp_path, monkeypatch):
    m = runtime(monkeypatch)
    name, a, b, arguments = probe(tmp_path, monkeypatch)
    with m.AcceptanceRounds(name, arguments, workers=2) as rounds:
        rounds.accept(*arguments[7])
        with pytest.raises(ValueError, match='repeated|round'):
            rounds.accept(*arguments[7])


def test_unregistered_arguments_are_rejected_before_any_input_is_read(tmp_path, monkeypatch):
    m = runtime(monkeypatch)
    name, a, b, arguments = probe(tmp_path, monkeypatch)
    with m.AcceptanceRounds(name, arguments, workers=2) as rounds:
        with pytest.raises(ValueError, match='registered|arguments'):
            rounds.accept(str(a), 123)
    assert not a.with_suffix('.calls').exists()
    assert not b.with_suffix('.calls').exists()


@pytest.mark.parametrize('workers', [0, 4])
def test_worker_bound_rejects_empty_or_unbounded_pools(tmp_path, monkeypatch, workers):
    m = runtime(monkeypatch)
    name, a, b, arguments = probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='worker|process'):
        m.AcceptanceRounds(name, arguments, workers=workers)


def test_cli_rejects_changed_runtime_before_reading_a_config(tmp_path, monkeypatch):
    runtime(monkeypatch)
    result = subprocess.run([sys.executable, str(SCRIPT), 'wrong-hash', 'journal_detector.summary_h',
        '--summary-sha', '0'*64, '--config', str(tmp_path/'missing.json'), '--backbone', 'efficientnet_b0'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'runtime' in result.stderr.lower() and 'hash' in result.stderr.lower()
    assert not (tmp_path/'analysis').exists()


def test_cli_rejects_changed_original_summary_before_reading_a_config(tmp_path, monkeypatch):
    runtime(monkeypatch)
    expected = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    result = subprocess.run([sys.executable, str(SCRIPT), expected, 'journal_detector.summary_h',
        '--summary-sha', '0'*64, '--config', str(tmp_path/'missing.json'), '--backbone', 'efficientnet_b0'],
        capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    assert 'summary' in result.stderr.lower() and 'hash' in result.stderr.lower()
    assert not (tmp_path/'analysis').exists()
