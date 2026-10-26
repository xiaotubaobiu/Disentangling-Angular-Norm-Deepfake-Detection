from pathlib import Path
from dataclasses import asdict
from collections import Counter
import json
import sys

import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from experiment_data import select_records, assert_disjoint, atomic_json, sha256, language_bank, ExhaustiveSuite
from data.builders import build_train_dataset, build_val_dataset, build_eval_dataset

cfg = yaml.safe_load((PHASE / 'configs/stage1.yaml').read_text())
folder = PHASE / 'results/protocol'
if (folder / 'manifest.json').exists():
    raise RuntimeError('protocol already frozen; verify instead of overwriting')
splits = {}
splits['train'] = select_records(build_train_dataset(cfg).records, max_fake_videos=None, max_real_videos=None, frames=2)
splits['val'] = select_records(build_val_dataset(cfg).records, max_fake_videos=8, max_real_videos=32, frames=1)
for domain in ('ffpp', 'cdf'):
    records = build_eval_dataset(cfg, domain=domain).records
    splits[domain + '_full'] = select_records(records, max_fake_videos=None, max_real_videos=None, frames=4)
    splits[domain + '_semantic'] = select_records(records, max_fake_videos=32, max_real_videos=None, frames=4)
assert_disjoint({k: splits[k] for k in ('train', 'val', 'ffpp_full')})
# Across source domains compare physical paths; video-id strings can repeat.
source_paths = {r.frame_path for k in ('train', 'val', 'ffpp_full') for r in splits[k]}
if source_paths & {r.frame_path for r in splits['cdf_full']}:
    raise ValueError('cross-domain physical overlap')
manifest = {'scope': 'single_seed_architecture_stage1', 'seed': 42, 'splits': {}, 'files': {},
            'data_definition': 'DF40-derived ff/cdf, same 16 manipulation methods',
            'train_val_ff_test_overlap': 0, 'cdf_physical_overlap': 0}
for name, records in splits.items():
    path = folder / (name + '.json')
    atomic_json(path, [asdict(r) for r in records])
    videos = {(r.method, r.video_id, r.label) for r in records}
    manifest['splits'][name] = {'path': str(path.relative_to(PHASE)), 'sha256': sha256(path),
        'frames': len(records), 'videos': len(videos), 'videos_by_method': dict(Counter(m for m, _, _ in videos))}
    print(name, len(records), 'frames', len(videos), 'videos', flush=True)
for path in [PHASE / 'configs/stage1.yaml',
             ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml',
             ROOT / 'experiments/2026-09-23_e3_smolvlm_candidate/configs/diagnostics.yaml']:
    manifest['files'][str(path)] = sha256(path)
bank = language_bank()
manifest['language_slots'] = {s: len(ExhaustiveSuite([], bank, split=s).slots) for s in ('train', 'heldout')}
atomic_json(folder / 'manifest.json', manifest)
print('PROTOCOL FROZEN', manifest['language_slots'], flush=True)
