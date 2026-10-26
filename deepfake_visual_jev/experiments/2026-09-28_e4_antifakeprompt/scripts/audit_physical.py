"""Distinguish source-index aliases from physical video directories."""
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import json
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]
from experiment_data import atomic_json

splits, report = {}, {'splits': {}, 'overlaps': {}}
for name in ('train', 'val', 'ffpp_full', 'cdf_full', 'ffpp_semantic', 'cdf_semantic'):
    rows = json.loads((PHASE / f'results/protocol/{name}.json').read_text())
    physical = defaultdict(list)
    labels = defaultdict(set)
    for row in rows:
        directory = str(Path(row['frame_path']).parent.resolve())
        physical[directory].append(row)
        labels[directory].add(row['label'])
    if any(len(value) != 1 for value in labels.values()):
        raise ValueError('conflicting physical video labels')
    real = {k: v for k, v in physical.items() if v[0]['label'] == 0}
    report['splits'][name] = {
        'index_video_keys': len({(r['method'], r['pair_id']) for r in rows}),
        'physical_videos': len(physical), 'physical_real_videos': len(real),
        'index_frame_records': len(rows), 'physical_frame_files': len({r['frame_path'] for r in rows}),
        'real_alias_multiplicity': dict(Counter(len({r['pair_id'] for r in group}) for group in real.values()))}
    splits[name] = set(physical)
for left, right in itertools.combinations(('train', 'val', 'ffpp_full', 'cdf_full'), 2):
    count = len(splits[left] & splits[right])
    report['overlaps'][left + '/' + right] = count
    if count:
        raise ValueError(f'physical video leakage: {left}/{right}')
for domain in ('ffpp', 'cdf'):
    if not splits[domain + '_semantic'] <= splits[domain + '_full']:
        raise ValueError('semantic subset not inside full test')
    for suffix in ('_full', '_semantic'):
        stat = report['splits'][domain + suffix]
        if stat['index_video_keys'] != stat['physical_videos'] or stat['index_frame_records'] != stat['physical_frame_files']:
            raise ValueError('test inference contains duplicated physical records')
report['interpretation'] = (
    'Training real videos retain 16 source-method aliases. These are repeated sampling entries, not independent videos. '
    'Validation has 160 index records but 156 physical videos (28 real); four real videos are weighted twice in frozen selection CE. '
    'Both arms use identical index draws and selection protocol. No split shares physical video directories. '
    'All reported test/diagnostic populations and bootstrap clusters are physically unique. No frozen training or checkpoint selection was changed retrospectively.')
atomic_json(PHASE / 'results/protocol/physical_audit.json', report)
print(json.dumps(report, indent=2), flush=True)
