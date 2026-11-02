import gzip,hashlib,importlib
from pathlib import Path
import pytest

def api():return importlib.import_module('reliability_v2.sensitivity_io')
def sha(data):return hashlib.sha256(data).hexdigest()
def ids(p,r):return dict(compressed_sha256=r['compressed_sha256'],uncompressed_sha256=r['uncompressed_sha256'])

def test_postcommit_cleanup_failure_returns_receipt_and_preserves_both_names(tmp_path,monkeypatch):
    m=api();p=tmp_path/'result.json.gz';unlink=Path.unlink
    def fail(path,*args,**kwargs):
        if path.name.endswith('.partial'):raise OSError('injected duplicate cleanup failure')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',fail);r=m.write_json(p,{'a':1})
    assert p.exists() and gzip.decompress(p.read_bytes())==b'{\n  "a": 1\n}\n'
    assert r['temporary_cleanup_pending'] is True and Path(r['duplicate_temporary_path']).exists()
    assert sha(p.read_bytes())==r['compressed_sha256']

def test_reader_rejects_implicit_unbounded_inflation_budget(tmp_path):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,{'a':1})
    with pytest.raises(ValueError,match='budget'):m.read_json(p,**ids(p,r))

def test_inflated_limit_is_checked_incrementally_before_bulk_allocation(tmp_path,monkeypatch):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,{'payload':'x'*1024*1024});original=m.gzip.GzipFile.read;sizes=[]
    def read(stream,size=-1):sizes.append(size);return original(stream,size)
    monkeypatch.setattr(m.gzip.GzipFile,'read',read)
    with pytest.raises(ValueError,match='budget'):m.read_json(p,**ids(p,r),max_uncompressed_bytes=64)
    assert sizes and all(0<n<=65 for n in sizes)

def test_exact_raw_byte_budget_is_accepted(tmp_path):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,{'a':1});size=len(gzip.decompress(p.read_bytes()))
    assert m.read_json(p,**ids(p,r),max_uncompressed_bytes=size)=={'a':1}

@pytest.mark.parametrize('limit',[True,False,0,-1,1.5])
def test_invalid_read_budgets_are_rejected(tmp_path,limit):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,{'a':1})
    with pytest.raises(ValueError,match='budget'):m.read_json(p,**ids(p,r),max_uncompressed_bytes=limit)

def test_finite_JSON_exponent_cannot_manufacture_nonfinite_float(tmp_path):
    m=api();p=tmp_path/'result.json.gz';raw=b'{"score":1e400}';p.write_bytes(gzip.compress(raw,mtime=0))
    with pytest.raises(ValueError,match='nonfinite'):m.read_json(p,compressed_sha256=sha(p.read_bytes()),uncompressed_sha256=sha(raw),max_uncompressed_bytes=1024)
