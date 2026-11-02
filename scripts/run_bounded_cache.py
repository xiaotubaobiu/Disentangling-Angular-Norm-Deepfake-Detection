"""Runtime thread bounds for unchanged FP32 cache modules and frozen configs."""
from pathlib import Path
import hashlib
import json
import runpy
import sys

def main():
    expected=sys.argv.pop(1)
    actual=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if expected!=actual:raise ValueError('runtime cache wrapper changed after registration')
    module=sys.argv.pop(1)
    if module not in ['journal_detector.cache_d','journal_detector.cache_h']:
        raise ValueError('bounded cache wrapper accepts only registered cache modules')
    check='--runtime-check' in sys.argv
    if check:sys.argv.remove('--runtime-check')
    config_path=Path(sys.argv[sys.argv.index('--config')+1])
    config=json.loads(config_path.read_text())
    workers=config.get('cache_workers',config.get('workers',4))
    if type(workers) is not int or not 0<=workers<=6:
        raise ValueError('effective cache loader count exceeds the six-worker bound')
    import cv2,torch
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(json.dumps(dict(runtime='BOUNDED_FP32_CACHE',module=module,cache_workers=workers,
        opencv_threads=cv2.getNumThreads(),torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),wrapper_sha256=actual,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),check_only=check)),flush=True)
    if check:return
    runpy.run_module(module,run_name='__main__',alter_sys=True)

if __name__=='__main__':main()
