"""Export supplementary video tables only after all E3 evidence is verified.

This does not change any scientific gate. All metrics use the existing physical
video predictions; seeds are summarized separately, never ensembled. Only FF++
and CDF are read. NLL uses sklearn's standard log_loss numerical clipping.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import log_loss

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.semantic_eval import _ece


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_script(ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/scripts/export_report_tables.py',
                     'e3_report_legacy_helpers')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixed_report_row(path, *, seed, domain, model):
    if model not in ('H1', 'E3') or domain not in ('ffpp', 'cdf'):
        raise ValueError('report supports only H1/E3 and FF++/CDF')
    # Reuse the validated E2 CSV layout checks for E3, then name it correctly.
    row, identities = LEGACY.read_fixed(path, seed, domain, 'E2' if model == 'E3' else 'H1')
    frame = pd.read_csv(path)
    labels = frame.label.to_numpy(int)
    probabilities = frame.prob_fake.to_numpy(float)
    row.update(model=model, ece=float(_ece(labels, probabilities)),
               nll=float(log_loss(labels, probabilities, labels=[0, 1])))
    return row, identities


def main():
    evaluator = load_script(PHASE / 'scripts/evaluate_candidate.py', 'e3_report_evaluator')
    assessment = load_script(PHASE / 'scripts/assess_candidate.py', 'e3_report_assessment')
    evaluator.verify_lock(PHASE / 'configs/assessment_lock.json')
    _, audit = assessment.validate_all_evidence(evaluator, PHASE / 'results/evaluation')
    output = PHASE / 'results/summary'
    gate_path, audit_path = output / 'gate_status.json', output / 'artifact_audit.json'
    status = json.loads(gate_path.read_text())
    if status.get('adjudicated') is not True or json.loads(audit_path.read_text())['groups'] != audit:
        raise ValueError('a completed assessment of the current 66 groups is required')
    if len(audit) != 66:
        raise ValueError('report requires all 66 declared groups')
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in (gate_path, audit_path)}
    fixed, semantic = [], []
    metrics = (*LEGACY.FIXED_METRICS, 'ece', 'nll')
    for seed in (42, 7, 2048):
        for domain in ('ffpp', 'cdf'):
            h1_path = ROOT / f'experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1/seed_{seed}/{domain}_video_predictions.csv'
            directory = PHASE / f'results/evaluation/seed_{seed}'
            e3_path = directory / f'fixed_h1/{domain}_videos.csv'
            for path in (h1_path, e3_path):
                inputs[str(path.relative_to(ROOT))] = sha256(path)
            h1, h1_ids = fixed_report_row(h1_path, seed=seed, domain=domain, model='H1')
            e3, e3_ids = fixed_report_row(e3_path, seed=seed, domain=domain, model='E3')
            if h1_ids != e3_ids:
                raise ValueError('paired models differ in video identity, label or unique frame count')
            metric_path = directory / f'fixed_h1/{domain}_metrics.json'
            inputs[str(metric_path.relative_to(ROOT))] = sha256(metric_path)
            records = LEGACY.read_metric_json(metric_path, seed, domain, 'fixed_h1')
            for metric, column in [('auroc', 'auroc'), ('accuracy', 'accuracy_at_0_5'),
                                   ('brier', 'brier'), ('ap', 'ap'), ('ece', 'ece')]:
                stored = LEGACY.one_overall(records, 'T1', metric, metric_path)
                if stored['count'] != e3['count'] or abs(stored['value'] - e3[column]) > 1e-10:
                    raise ValueError(f'saved/recomputed video metric differs: {metric_path}/{metric}')
            fixed.extend((h1, e3))
            for condition in ('seen', 'heldout'):
                path = directory / condition / f'{domain}_metrics.json'
                inputs[str(path.relative_to(ROOT))] = sha256(path)
                rows = LEGACY.semantic_video_rows(path, seed, domain, condition)
                semantic.extend({**row, 'model': 'E3'} for row in rows)
    fixed_long = [{**{k: row[k] for k in ('model', 'seed', 'domain', 'condition', 'task_id', 'count', 'unit')},
                   'metric': metric, 'value': row[metric]} for row in fixed for metric in metrics]
    tables = {
        'report_fixed_video_by_seed.csv': fixed,
        'report_fixed_video_summary.csv': LEGACY.summarize(fixed_long),
        'report_semantic_video_by_seed.csv': semantic,
        'report_semantic_video_summary.csv': LEGACY.summarize(semantic),
    }
    for name, expected in inputs.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'input changed while exporting report: {name}')
    for name, rows in tables.items():
        LEGACY.write_csv(output / name, rows)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'reportable': True,
        'audited_groups': len(audit), 'domains': ['ffpp', 'cdf'], 'seeds': [42, 7, 2048],
        'aggregation': 'arithmetic mean and sample SD across three fixed seeds, no ensemble',
        'threshold': .5, 'ece_bins': 15, 'gate_results_modified': False,
        'nll_definition': 'sklearn.metrics.log_loss on float64 P(fake), labels=[0,1]',
        'inputs': inputs, 'outputs': {name: sha256(output / name) for name in tables},
        'reporter_sha256': sha256(Path(__file__)),
        'legacy_helpers_sha256': sha256(Path(LEGACY.__file__)),
        'evaluation_lock_sha256': sha256(PHASE / 'configs/evaluation_lock.json'),
        'assessment_lock_sha256': sha256(PHASE / 'configs/assessment_lock.json'),
    }
    evaluator.atomic_json(output / 'report_tables_manifest.json', manifest)
    print(json.dumps({'audited_groups': len(audit), 'tables': list(tables)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
