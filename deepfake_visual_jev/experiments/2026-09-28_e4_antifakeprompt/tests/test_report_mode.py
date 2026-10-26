import importlib.util
import json
from pathlib import Path
import sys


def test_partial_audit_cannot_launch_final_stage_when_nothing_missing(tmp_path, monkeypatch):
    phase = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('report_mode_test', phase / 'scripts/report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'results/protocol').mkdir(parents=True)
    # An empty synthetic inventory exercises the no-missing-items control path
    # without manufacturing reportable scientific results or loading a model.
    (tmp_path / 'configs/stage1.yaml').write_text('evaluation:\n  conditions: []\n')
    (tmp_path / 'results/protocol/manifest.json').write_text('{"files": {}}')
    monkeypatch.setattr(report, 'PHASE', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['report.py', '--partial'])

    def forbidden_final_stage(*args, **kwargs):
        raise AssertionError('partial audit tried to start the final stage')

    monkeypatch.setattr(report.subprocess, 'run', forbidden_final_stage)
    report.main()
    audit = json.loads((tmp_path / 'results/summary/artifact_audit.json').read_text())
    assert audit['missing_conditions'] == []
    assert not (tmp_path / 'results/summary/gate_status.json').exists()
