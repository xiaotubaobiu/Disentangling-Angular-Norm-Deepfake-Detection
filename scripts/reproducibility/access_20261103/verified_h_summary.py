"""Run the original H summary with explicitly recorded, sealed digest reuse."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from journal_detector import analysis_d, geometry_h, summary_h, trainer
from .verified_digest import guarded_digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--backbone', required=True, choices=summary_h.analysis.BACKBONES)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['artifact_root']).resolve()
    target = root / f'operations/{args.backbone}_summary_digest_transaction.json'
    if target.exists():
        raise ValueError('transaction receipt already exists; inspect before rerunning')
    original_hash = trainer.digest
    with guarded_digests([(trainer, 'digest'), (geometry_h, 'sha256_file')]) as memo:
        result = summary_h.summarize_backbone(args.config, args.backbone)
    marker = root / f'analysis/H/{args.backbone}/three_seed_summary/complete.json'
    receipt = dict(status='PASS_SEALED_LARGE_FILE_DIGEST_TRANSACTION', sealed=memo.sealed,
        created=datetime.now(timezone.utc).isoformat(), backbone=args.backbone,
        original_summary_function='journal_detector.summary_h.summarize_backbone',
        original_summary_code_sha256=original_hash(summary_h.__file__),
        original_training_code_sha256=trainer.code_hash('H'),
        summary_complete_sha256=original_hash(marker), wrapper_sha256=original_hash(__file__),
        digest_helper_sha256=original_hash(Path(__file__).with_name('verified_digest.py')),
        minimum_cached_bytes=memo.min_bytes, repeated_large_reads_avoided=memo.hits,
        large_inputs=memo.entries,
        guarantee='Each large input fully hashed at first use and fully reread before success; metadata guard on every reuse. Small code/metadata hashes and all original numeric/identity validations remain active.',
        statistical_code_modified=False, model_or_score_modified=False)
    trainer.save_json(target, receipt)
    print(json.dumps(analysis_d._jsonable(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
