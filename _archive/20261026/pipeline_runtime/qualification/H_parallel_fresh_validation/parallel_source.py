"""Isolated source-verification scheduling candidate; imports have no side effects.

Never caches a receipt across invocations. Production does not import this file.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import json

def collect_fresh(config_path, run_ids, verify_complete, *, workers=2):
    ids=list(run_ids)
    if type(workers) is not int or workers not in (1,2):
        raise ValueError("Fresh verification permits one or two workers")
    if not ids or any(not isinstance(r,str) or not r for r in ids) or len(set(ids))!=len(ids):
        raise ValueError("Expected nonempty unique model identifiers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values=list(executor.map(lambda run: verify_complete(config_path,run),ids))
    return dict(zip(ids,values))

class FreshLookup:
    """One original-ledger invocation may consume each current receipt once."""
    def __init__(self,config_path,receipts):
        self._config=Path(config_path).resolve()
        self._receipts=deepcopy(receipts)
        self._seen=set()
    def __call__(self,config_path,run_id):
        if Path(config_path).resolve()!=self._config or run_id not in self._receipts or run_id in self._seen:
            raise ValueError("Fresh receipt context/selection/reuse mismatch")
        self._seen.add(run_id)
        return deepcopy(self._receipts[run_id])
    def finish(self):
        if self._seen!=set(self._receipts):
            raise ValueError("Original ledger did not consume every fresh receipt")

def assert_same_ledger(reference,candidate):
    for value in (reference,candidate):
        if not isinstance(value,dict) or not isinstance(value.get("utc"),str):
            raise ValueError("Original ledger timestamp missing")
    def canonical(value):
        return json.dumps({k:v for k,v in value.items() if k!="utc"},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
    if canonical(reference)!=canonical(candidate):
        raise ValueError("Serial and parallel original ledgers differ")


class FrozenHashes:
    """Cryptographically bracket one operation over all selected source files."""
    def __init__(self,paths):
        import hashlib
        self._digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.hashes={str(Path(p).resolve()):self._digest(p) for p in paths}
    def check(self):
        if any(self._digest(path)!=digest for path,digest in self.hashes.items()):
            raise ValueError("Frozen source/input drift during operation")
    def run(self,operation):
        self.check()
        value=operation()
        self.check()
        return value

def publish_json_new(path,result):
    """Publish complete fsynced JSON atomically, never overwrite prior evidence.

    A failed write/sync leaves only a unique .partial as failure evidence.
    """
    import os,tempfile
    path=Path(path)
    if path.exists() or path.is_symlink():raise FileExistsError(path)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,prefix="."+path.name+".",suffix=".partial",delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)
    temporary.unlink()
