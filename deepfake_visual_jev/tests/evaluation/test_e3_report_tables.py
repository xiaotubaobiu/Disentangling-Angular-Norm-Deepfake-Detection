"""Report supplements must expose threshold failures and require final evidence."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'experiments/2026-09-23_e3_smolvlm_candidate/scripts/export_report_tables.py'


def load_exporter():
    spec = importlib.util.spec_from_file_location('e3_report_tables_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_predictions(path):
    rows = []
    for i, (label, score) in enumerate([(0, .6), (0, .7), (1, .8), (1, .9),
                                      (1, .95), (1, .99), (1, .85), (1, .92)]):
        method = 'fake' if label else 'real'
        rows.append({'seed': 42, 'domain': 'cdf', 'condition': 'fixed_h1', 'task_id': 'T1',
                     'method_id': method, 'video_id': str(i), 'video_key': f'{method}::{i}',
                     'label': label, 'prob_fake': score, 'num_frames': 4})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_perfect_ranking_does_not_hide_zero_real_recall(tmp_path):
    exporter = load_exporter()
    path = tmp_path / 'predictions.csv'
    write_predictions(path)
    row, identities = exporter.fixed_report_row(path, seed=42, domain='cdf', model='E3')
    assert row['model'] == 'E3' and len(identities) == 8
    assert row['auroc'] == 1
    assert row['accuracy_at_0_5'] == .75
    assert row['balanced_accuracy_at_0_5'] == .5
    assert row['real_recall_at_0_5'] == 0
    assert row['fake_recall_at_0_5'] == 1
    assert row['brier'] == pytest.approx(.1164375)
    assert row['ece'] == pytest.approx(.23625)
    assert row['nll'] == pytest.approx(.34450147148125065)


def test_report_rejects_wrong_seed_context(tmp_path):
    exporter = load_exporter()
    path = tmp_path / 'predictions.csv'
    write_predictions(path)
    with pytest.raises(ValueError, match='context'):
        exporter.fixed_report_row(path, seed=7, domain='cdf', model='E3')


def test_missing_evidence_does_not_emit_report_tables(tmp_path, monkeypatch):
    exporter = load_exporter()
    monkeypatch.setattr(exporter, 'PHASE', tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        exporter.main()
    assert not (tmp_path / 'results/summary/report_fixed_video_by_seed.csv').exists()
    assert not (tmp_path / 'results/summary/report_tables_manifest.json').exists()
