"""Parallel scheduling of the unchanged D input validator; all checks retained."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import sys

THREADS=32

def install_validation_threads(threads=THREADS):
    import numpy as np
    import pandas as pd
    from journal_detector import cache_d
    current=cache_d._validate_inputs
    original=getattr(current,'_original_input_validator',current)
    def validate(frame):
        if not {'image_path','content_sha256'}.issubset(frame) or frame.empty:
            return original(frame)
        # Exact equal paths always share a shard, so the original conflicting-
        # metadata check still sees every duplicate of a path together.
        if frame.image_path.isna().any():return original(frame)
        codes,unique=pd.factorize(frame.image_path,sort=False)
        count=min(int(threads),len(unique))
        if count<=1:return original(frame)
        assignment=codes%count
        partitions=[frame.iloc[np.flatnonzero(assignment==i)] for i in range(count)]
        assert sum(len(part) for part in partitions)==len(frame)
        with ThreadPoolExecutor(max_workers=count) as pool:
            for _ in pool.map(original,partitions):pass
    validate._original_input_validator=original
    cache_d._validate_inputs=validate
    return original,validate

def main():
    expected=sys.argv.pop(1)
    actual=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if expected!=actual:raise ValueError('runtime validation wrapper changed after registration')
    module=sys.argv.pop(1)
    if module not in ['journal_detector.cache_d','journal_detector.analysis_d','journal_detector.summary_d']:
        raise ValueError('D runtime wrapper accepts only registered D cache/analysis/summary modules')
    check='--runtime-check' in sys.argv
    if check:sys.argv.remove('--runtime-check')
    config_path=Path(sys.argv[sys.argv.index('--config')+1]);config=json.loads(config_path.read_text())
    workers=config.get('cache_workers',config.get('workers',4))
    if module=='journal_detector.cache_d' and (type(workers) is not int or not 0<=workers<=6):
        raise ValueError('effective cache loader count exceeds the six-worker bound')
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original,_=install_validation_threads()
    from journal_detector import cache_d
    print(json.dumps(dict(runtime='PARALLEL_D_INPUT_VALIDATION',module=module,
        validation_threads=THREADS,validator=original.__module__+'.'+original.__name__,
        original_validator_file_sha256=hashlib.sha256(Path(cache_d.__file__).read_bytes()).hexdigest(),
        cache_workers=workers,opencv_threads=cv2.getNumThreads(),torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),wrapper_sha256=actual,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),check_only=check)),flush=True)
    if check:return
    if module=='journal_detector.cache_d':
        parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
        parser.add_argument('--run-id',required=True);parser.add_argument('--device',default='cuda')
        args=parser.parse_args()
        print(json.dumps(cache_d.extract(args.config,args.run_id,args.device),ensure_ascii=False,allow_nan=False))
    else:
        importlib.import_module(module).main()

if __name__=='__main__':main()
