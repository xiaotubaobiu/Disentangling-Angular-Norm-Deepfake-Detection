"""Run unchanged strict-summary acceptance rounds in bounded CPU processes."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _limit_loaded_libraries():
    if globals().get('_LIMITS_SET'):
        return
    if 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(4)
        sys.modules['torch'].set_num_interop_threads(1)
    if 'cv2' in sys.modules:
        sys.modules['cv2'].setNumThreads(1)
    globals()['_LIMITS_SET'] = True


def _invoke_original(module_name, args, module_sha, runtime_sha):
    if digest(__file__) != runtime_sha:
        raise ValueError('runtime script hash changed in validation child')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '4'
    module = importlib.import_module(module_name)
    if digest(module.__file__) != module_sha:
        raise ValueError('original summary module hash changed in validation child')
    # Each spawned process has its own numerical-library and precision state.
    if not globals().get('_CHILD_SETUP'):
        _limit_loaded_libraries()
        if module_name == 'journal_detector.summary_d':
            from run_parallel_d_validation import install_validation_threads
            install_validation_threads(32)
        print(json.dumps(dict(runtime='SUMMARY_VALIDATION_CHILD', pid=os.getpid(),
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
            torch_threads=sys.modules['torch'].get_num_threads() if 'torch' in sys.modules else None,
            torch_interop_threads=sys.modules['torch'].get_num_interop_threads() if 'torch' in sys.modules else None,
            opencv_threads=sys.modules['cv2'].getNumThreads() if 'cv2' in sys.modules else None)), flush=True)
        globals()['_CHILD_SETUP'] = True
    return module._accept(*args)


class AcceptanceRounds:
    """Supply each original round once; the next round starts fresh validation."""

    def __init__(self, module_name, arguments_by_seed, workers=3):
        if type(workers) is not int or not 1 <= workers <= 3:
            raise ValueError('validation process workers must be between one and three')
        self.arguments = {seed: tuple(args) for seed, args in arguments_by_seed.items()}
        if not self.arguments or len(set(self.arguments.values())) != len(self.arguments):
            raise ValueError('nonempty unique registered acceptance arguments required')
        self.module_name = module_name
        self.module_sha = digest(importlib.util.find_spec(module_name).origin)
        self.runtime_sha = digest(__file__)
        self.workers = workers
        self.pool = None
        self.pending = {}
        self.remaining = set()
        self.round_count = 0
        self.accepted_call_count = 0
        self.failed = False

    def __enter__(self):
        self.pool = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context('spawn'))
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pool = None

    def accept(self, *args):
        if self.pool is None or self.failed:
            raise RuntimeError('acceptance pool is closed or has a failed validation')
        candidates = [seed for seed, registered in self.arguments.items() if registered == args]
        if len(candidates) != 1:
            raise ValueError('unregistered acceptance arguments')
        seed = candidates[0]
        if not self.remaining:
            self.pending = {s: self.pool.submit(_invoke_original, self.module_name,
                registered, self.module_sha, self.runtime_sha) for s, registered in self.arguments.items()}
            self.remaining = set(self.arguments)
            self.round_count += 1
        elif seed not in self.remaining:
            raise ValueError('repeated seed before the acceptance round is complete')
        try:
            result = self.pending[seed].result()
        except BaseException:
            self.failed = True
            raise
        self.remaining.remove(seed)
        self.accepted_call_count += 1
        return result


def main():
    expected_runtime = sys.argv.pop(1)
    if digest(__file__) != expected_runtime:
        raise ValueError('runtime script hash mismatch')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', choices=['journal_detector.summary_d', 'journal_detector.summary_h'])
    parser.add_argument('--summary-sha', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--backbone', required=True,
        choices=['clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--qualification-report', type=Path)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    module = importlib.import_module(args.module)
    if digest(module.__file__) != args.summary_sha:
        raise ValueError('original summary module hash mismatch')
    _limit_loaded_libraries()
    config = args.config.resolve()
    is_d = args.module.endswith('summary_d')
    if is_d:
        root, _, runs = module._registry(config, args.backbone)
        arguments = {seed: (config, root, run, args.backbone, seed) for seed, run in runs.items()}
        marker = root / f'analysis/D/{args.backbone}/five_seed_summary/complete.json'
    else:
        root, _ = module._registry(config, args.backbone)
        arguments = {seed: (config, args.backbone, seed) for seed in module.SEEDS}
        marker = root / f'analysis/H/{args.backbone}/three_seed_summary/complete.json'
    before = None
    artifact_hashes = {}
    if args.qualify:
        if args.qualification_report is None or not marker.is_file():
            raise ValueError('qualification requires a completed strict summary and report destination')
        before = json.loads(marker.read_text())
        artifact_hashes = {name: digest(marker.parent / name) for name in before['artifact_sha256']}
        if artifact_hashes != before['artifact_sha256']:
            raise ValueError('qualification input artifacts do not match the completed summary')
        marker_sha = digest(marker)
    started = time.monotonic()
    original = module._accept
    with AcceptanceRounds(args.module, arguments, args.workers) as rounds:
        print(json.dumps(dict(runtime='PARALLEL_STRICT_SUMMARY_ACCEPTANCE', module=args.module,
            runtime_sha256=expected_runtime, summary_sha256=args.summary_sha,
            processes=args.workers, CUDA_VISIBLE_DEVICES='', original_collection_unchanged=True)), flush=True)
        module._accept = rounds.accept
        try:
            result = module.summarize_backbone(config, args.backbone)
        finally:
            module._accept = original
    if rounds.remaining or rounds.round_count != 2 or rounds.accepted_call_count != 2 * len(arguments):
        raise ValueError('original initial and final complete acceptance rounds required')
    if args.qualify:
        if (result != before or digest(marker) != marker_sha
                or any(digest(marker.parent / name) != sha for name, sha in artifact_hashes.items())):
            raise ValueError('completed summary changed during parallel qualification')
        record = dict(status='QUALIFIED_ORIGINAL_COMPLETED_SUMMARY_UNCHANGED',
            created=datetime.now(timezone.utc).isoformat(), module=args.module, backbone=args.backbone,
            processes=args.workers, acceptance_rounds=rounds.round_count,
            original_acceptance_calls=rounds.accepted_call_count, elapsed_seconds=time.monotonic()-started,
            original_summary_sha256=args.summary_sha, runtime_sha256=expected_runtime,
            summary_complete_sha256=marker_sha, artifact_sha256=artifact_hashes,
            original_collection_statistics_and_cross_seed_checks_unchanged=True,
            training_or_inference=False)
        args.qualification_report.parent.mkdir(parents=True, exist_ok=True)
        args.qualification_report.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2), flush=True)
    else:
        print(json.dumps(module.io._jsonable(result), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
