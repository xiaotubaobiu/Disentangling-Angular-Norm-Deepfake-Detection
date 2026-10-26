"""Require complete E3 evidence before applying original and additional gates."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config_io import load_config
from evaluation.e2_gates import evaluate_gates, SEEDS, DOMAINS, CONDITIONS
from evaluation.smolvlm_gates import negation_results, composition_results, matched_control_results


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_all_evidence(evaluator, evaluation_root):
    config_path = PHASE / 'configs/candidate.yaml'
    cfg = load_config(config_path)
    training_sha = evaluator.verify_lock(PHASE / 'configs/protocol_lock.json')
    evaluation_sha = evaluator.verify_lock(PHASE / 'configs/evaluation_lock.json')
    for name in ('training_logs', 'evaluation_logs'):
        series = json.loads((PHASE / f'results/{name}/series_status.json').read_text())
        if series.get('complete') is not True or [r['seed'] for r in series['runs']] != list(SEEDS):
            raise ValueError(f'incomplete seed series: {name}')
        if any(r['exit_code'] != 0 for r in series['runs']):
            raise ValueError(f'failed launcher in {name}')
    audit = []
    for seed in SEEDS:
        train_dir = PHASE / f'results/training/seed_{seed}'
        checkpoint_path = train_dir / 'best_model.pth'
        checkpoint_sha = evaluator.sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        completion = json.loads((train_dir / 'completion.json').read_text())
        evaluator.validate_checkpoint(checkpoint, completion, {
            'config_sha256': evaluator.sha256(config_path), 'protocol_sha256': training_sha,
            'language_manifest_sha256': evaluator.sha256(ROOT / cfg['language']['manifest']),
            'pretrained_manifest_sha256': evaluator.sha256(PHASE / 'results/pretrained/manifest.json')},
            seed=seed, checkpoint_sha=checkpoint_sha)
        del checkpoint
        directory = evaluation_root / f'seed_{seed}'
        provenance = json.loads((directory / 'run_provenance.json').read_text())
        for key, value in {'seed': seed, 'smoke': False, 'reportable': True, 'world_size': 4,
                           'visible_devices': '0,1,2,3', 'checkpoint_sha256': checkpoint_sha,
                           'protocol_sha256': training_sha, 'evaluation_lock_sha256': evaluation_sha,
                           'config_sha256': evaluator.sha256(config_path), 'precision': 'float32'}.items():
            if provenance.get(key) != value:
                raise ValueError(f'invalid evaluation provenance: seed={seed}, {key}')
        for domain in DOMAINS:
            cache_path = PHASE / f'results/native_cache/{domain}/manifest.json'
            if provenance['native_caches'][domain] != evaluator.sha256(cache_path):
                raise ValueError('native cache provenance changed')
            for condition in (*CONDITIONS, *evaluator.ADDED_CONDITIONS):
                path = directory / condition / f'{domain}_completion.json'
                completed = json.loads(path.read_text())
                for key, value in {'seed': seed, 'domain': domain, 'condition': condition,
                                   'smoke': False, 'reportable': True, 'checkpoint_sha256': checkpoint_sha,
                                   'evaluation_lock_sha256': evaluation_sha}.items():
                    if completed.get(key) != value:
                        raise ValueError(f'invalid group completion: {path}, {key}')
                required = [f'{domain}_metrics.json'] if condition == 'permutation' else [f'{domain}_metrics.json', f'{domain}_frames.npz', f'{domain}_videos.csv']
                if condition.startswith('meaningless_'):
                    required.append(f'{domain}_paired_videos.csv')
                if condition == 'counterfactual':
                    required.append(f'{domain}_controls.csv')
                if not all(f'{condition}/{file}' in completed['files'] for file in required):
                    raise ValueError(f'missing group artifacts: {path}')
                for relative, expected in completed['files'].items():
                    if evaluator.sha256(directory / relative) != expected:
                        raise ValueError(f'group artifact changed: {relative}')
                audit.append({'seed': seed, 'domain': domain, 'condition': condition,
                              'completion_sha256': evaluator.sha256(path), 'verified': True})
    return cfg, audit


def assess_complete_run():
    evaluator = load_script(PHASE / 'scripts/evaluate_candidate.py', 'e3_evaluation_for_assessment')
    evaluator.verify_lock(PHASE / 'configs/assessment_lock.json')
    directory = PHASE / 'results/evaluation'
    output = PHASE / 'results/summary'
    output.mkdir(parents=True, exist_ok=True)
    cfg, audit = validate_all_evidence(evaluator, directory)
    legacy = load_script(ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py', 'e3_legacy_gate_loader')
    artifacts = legacy.load_artifacts(directory, ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1')
    primary = evaluate_gates(**artifacts, cfg=cfg)
    def read_condition(condition, suffix='videos'):
        return pd.concat([pd.read_csv(directory / f'seed_{seed}' / condition / f'{domain}_{suffix}.csv', keep_default_na=False)
                          for seed in SEEDS for domain in DOMAINS], ignore_index=True)
    config = yaml.safe_load((ROOT / cfg['evaluation']['diagnostics_manifest']).read_text())
    additional = {
        'negation': negation_results(read_condition('negation'), config),
        'composition': composition_results(read_condition('composition'), read_condition('heldout'), config),
        'matched_controls': matched_control_results(pd.concat([
            read_condition('meaningless_matched_candidates', 'paired_videos'),
            read_condition('meaningless_matched_questions', 'paired_videos')], ignore_index=True), config)}
    succeeded = primary.verdict == 'supported' and all(row['passed'] for rows in additional.values() for row in rows)
    result = {'adjudicated': True, 'stage_1_succeeded': succeeded, 'primary': asdict(primary),
              'additional': additional, 'audited_groups': len(audit), 'bootstrap_resamples': 10000,
              'next_stage_triggered': succeeded,
              'interpretation': 'Whole-model replacement tests feasibility; it does not isolate the causal explanation for E2 failure.'}
    evaluator.atomic_json(output / 'gate_status.json', result)
    evaluator.atomic_json(output / 'artifact_audit.json', {'groups': audit})
    pd.DataFrame(primary.bootstrap_intervals).to_csv(output / 'bootstrap_intervals.csv', index=False)
    pd.DataFrame(primary.seed_metrics).to_csv(output / 'seed_metrics.csv', index=False)
    for name, rows in additional.items():
        pd.DataFrame(rows).to_csv(output / f'{name}_gates.csv', index=False)
    print(json.dumps({'stage_1_succeeded': succeeded, 'primary_verdict': primary.verdict,
                      'audited_groups': len(audit)}), flush=True)
    return 0


def main():
    try:
        return assess_complete_run()
    except (ValueError, OSError, KeyError, TypeError) as error:
        # A previous successful verdict must never survive a failed re-audit.
        output = PHASE / 'results/summary/gate_status.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'adjudicated': False, 'stage_1_succeeded': None,
                                        'error': str(error)}, indent=2) + '\n')
        temporary.replace(output)
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
