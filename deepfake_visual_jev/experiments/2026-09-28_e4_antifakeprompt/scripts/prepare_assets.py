"""Download/verify public pinned model files; never execute upstream setup."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

PHASE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--verify-only', action='store_true')
args = parser.parse_args()
spec = json.loads((PHASE / 'configs/assets.json').read_text())
for entry in spec['files']:
    path = PHASE / 'results/assets' / entry['path']
    if not path.exists():
        if args.verify_only:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.part')
        with requests.get(entry['url'], stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for block in response.iter_content(8 * 1024 * 1024):
                    stream.write(block)
        with partial.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if partial.stat().st_size != entry['bytes'] or digest != entry['sha256']:
            raise ValueError(f'asset checksum mismatch: {path.name}')
        partial.replace(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != entry['bytes'] or digest != entry['sha256']:
        raise ValueError(f'asset checksum mismatch: {path.name}')
    print('verified', entry['path'], digest, flush=True)
