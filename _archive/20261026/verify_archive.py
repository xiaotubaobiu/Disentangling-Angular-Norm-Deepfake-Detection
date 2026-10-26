#!/usr/bin/env python3
"""Verify copied bytes using the archived manifest; never reads original roots."""
from pathlib import Path
import hashlib
import json
import sys

here = Path(__file__).resolve().parent
root = here.parent.parent
manifest = json.loads((here / 'source_manifest.json').read_text())
errors = []
total = 0
for entry in manifest['files']:
    path = root / entry['archive']
    if not path.is_file() or path.is_symlink():
        errors.append({'path': entry['archive'], 'error': 'missing/nonregular'})
        continue
    data = path.read_bytes()
    total += len(data)
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        errors.append({'path': entry['archive'], 'error': 'size/hash mismatch'})
print(json.dumps({'files': len(manifest['files']), 'bytes': total, 'errors': errors}))
sys.exit(bool(errors))
