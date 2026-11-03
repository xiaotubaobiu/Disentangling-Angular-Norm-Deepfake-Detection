"""Fresh acceptance rounds must preserve the original fail-closed validation."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import json

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


def test_d_child_rejects_unqualified_helper_before_import_side_effects(tmp_path):
    sentinel = tmp_path / 'helper_imported.txt'
    (tmp_path / 'run_parallel_d_validation.py').write_text(
        'from pathlib import Path\n'
        f'Path({str(sentinel)!r}).write_text("changed helper was imported")\n'
        'def install_validation_threads(threads): pass\n')
    code = '''
from pathlib import Path
import journal_detector.summary_d as summary
import run_parallel_summary_validation as runtime
runtime._invoke_original('journal_detector.summary_d',
    (Path('missing-config.json'), Path('.'), 'missing-run', 'efficientnet_b0', 7),
    runtime.digest(summary.__file__), runtime.digest(runtime.__file__))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=os.pathsep.join(
        [str(tmp_path), str(SCRIPT.parent), str(SCRIPT.parents[1] / 'src')]))
    result = subprocess.run([sys.executable, '-c', code], env=env,
        capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    assert not sentinel.exists(), 'unqualified helper was executed before its identity was checked'
    assert 'validation helper hash' in result.stderr.lower()


def qualification_probe(tmp_path):
    """Exercise the real CLI around a small completed-summary boundary fixture."""
    package = tmp_path / 'modules' / 'journal_detector'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    summary = package / 'summary_h.py'
    summary.write_text('''
from pathlib import Path
import json
SEEDS = (7, 42, 123)
def _registry(config, backbone):
    return Path(json.loads(Path(config).read_text())['artifact_root']), {}
def _accept(config, backbone, seed):
    root, _ = _registry(config, backbone)
    (root / 'acceptance_started.txt').write_text('acceptance ran')
    return seed
def summarize_backbone(config, backbone):
    root, _ = _registry(config, backbone)
    for _ in range(2):
        for seed in SEEDS: _accept(config, backbone, seed)
    marker = root / 'analysis/H/efficientnet_b0/three_seed_summary/complete.json'
    return json.loads(marker.read_text())
''')
    root = tmp_path / 'experiment'
    output = root / 'analysis/H/efficientnet_b0/three_seed_summary'
    output.mkdir(parents=True)
    artifact = output / 'metrics.csv'
    artifact.write_text('seed,auc\n7,0.9\n')
    marker = output / 'complete.json'
    marker.write_text(json.dumps({'artifact_sha256': {
        'metrics.csv': hashlib.sha256(artifact.read_bytes()).hexdigest()}}))
    config = root / 'config.json'
    config.write_text(json.dumps({'artifact_root': str(root)}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
        PYTHONPATH=os.pathsep.join([str(package.parent), str(SCRIPT.parent)]))
    command = [sys.executable, str(SCRIPT), hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        'journal_detector.summary_h', '--summary-sha', hashlib.sha256(summary.read_bytes()).hexdigest(),
        '--config', str(config), '--backbone', 'efficientnet_b0', '--qualify']
    return root, output, marker, artifact, config, env, command


@pytest.mark.parametrize('destination', ['marker', 'artifact', 'new_summary_file', 'config',
    'outside_namespace', 'symlink_namespace', 'existing_hardlink'])
def test_qualification_rejects_unsafe_report_before_acceptance(tmp_path, destination):
    root, output, marker, artifact, config, env, command = qualification_probe(tmp_path)
    namespace = root / 'operations/summary_validation_qualification'
    targets = {'marker': marker, 'artifact': artifact, 'new_summary_file': output / 'qualification.json',
        'config': config, 'outside_namespace': root / 'unregistered-report.json'}
    if destination == 'symlink_namespace':
        namespace.parent.mkdir(parents=True)
        namespace.symlink_to(output, target_is_directory=True)
        report = namespace / 'qualification.json'
    elif destination == 'existing_hardlink':
        namespace.mkdir(parents=True)
        report = namespace / 'report.json'
        os.link(artifact, report)
    else:
        report = targets[destination]
    before = {p: p.read_bytes() for p in [marker, artifact, config]}
    result = subprocess.run(command + ['--qualification-report', str(report)], env=env,
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'qualification report' in result.stderr.lower()
    assert not (root / 'acceptance_started.txt').exists()
    assert all(path.read_bytes() == value for path, value in before.items())
    assert set(p.name for p in output.iterdir()) == {'complete.json', 'metrics.csv'}


def test_qualification_exclusively_writes_report_outside_summary_and_preserves_artifacts(tmp_path):
    root, output, marker, artifact, config, env, command = qualification_probe(tmp_path)
    report = root / 'operations/summary_validation_qualification/report.json'
    before = {p: p.read_bytes() for p in [marker, artifact, config]}
    first = subprocess.run(command + ['--qualification-report', str(report)], env=env,
        capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    payload = json.loads(report.read_text())
    assert payload['status'] == 'QUALIFIED_ORIGINAL_COMPLETED_SUMMARY_UNCHANGED'
    assert payload['acceptance_rounds'] == 2 and payload['original_acceptance_calls'] == 6
    assert all(path.read_bytes() == value for path, value in before.items())
    assert set(p.name for p in output.iterdir()) == {'complete.json', 'metrics.csv'}
    report_bytes = report.read_bytes()
    (root / 'acceptance_started.txt').unlink()
    second = subprocess.run(command + ['--qualification-report', str(report)], env=env,
        capture_output=True, text=True, timeout=30)
    assert second.returncode != 0
    assert not (root / 'acceptance_started.txt').exists()
    assert report.read_bytes() == report_bytes
