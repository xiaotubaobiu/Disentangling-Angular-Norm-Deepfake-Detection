"""Lossless secondary-output transport: tests name the break they catch."""
import gzip,hashlib,importlib,json,math,os
from pathlib import Path
import pytest

def api():
    try:return importlib.import_module('reliability_v2.sensitivity_io')
    except ModuleNotFoundError:pytest.fail('Missing selected-M lossless transport implementation')
def sha(data):return hashlib.sha256(data).hexdigest()
def fixture():
    return {'draws':[{'draw_id':i,'multiplicity':{'g0':2,'g1':0},'metric':None,'reason':'missing fixed method','precision':'0.000000000000000000000000001'} for i in range(2000)],'repeats':2000,'big':2**80,'negative_zero':-0.0,'unicode':'源域/零值','valid':False}
def load(m,p,r):return m.read_json(p,compressed_sha256=r['compressed_sha256'],uncompressed_sha256=r['uncompressed_sha256'],max_uncompressed_bytes=8*1024**2)

def test_roundtrip_preserves_all_2000_draws_types_NA_and_precision_strings(tmp_path):
    m=api();value=fixture();p=tmp_path/'result.json.gz';r=m.write_json(p,value);decoded=load(m,p,r)
    assert decoded==value and len(decoded['draws'])==2000
    assert type(decoded['big']) is int and type(decoded['valid']) is bool
    assert math.copysign(1,decoded['negative_zero'])==-1
    assert r['codec']=='gzip' and r['compressed_bytes']==p.stat().st_size
    raw=gzip.decompress(p.read_bytes());assert sha(raw)==r['uncompressed_sha256'] and len(raw)==r['uncompressed_bytes']

def test_compressed_bytes_do_not_depend_on_output_name_or_wall_clock(tmp_path):
    m=api();a=tmp_path/'a.json.gz';b=tmp_path/'b.json.gz';m.write_json(a,fixture());m.write_json(b,fixture());assert a.read_bytes()==b.read_bytes()

def test_existing_commit_is_never_overwritten(tmp_path):
    m=api();p=tmp_path/'result.json.gz';m.write_json(p,{'a':1});before=p.read_bytes()
    with pytest.raises(FileExistsError):m.write_json(p,{'a':2})
    assert p.read_bytes()==before

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),set([1])])
def test_unserializable_or_nonfinite_values_cannot_create_committed_output(tmp_path,bad):
    m=api();p=tmp_path/'result.json.gz'
    with pytest.raises((ValueError,TypeError)):m.write_json(p,{'prefix':[1,2],'bad':bad})
    assert not p.exists()

def test_interrupted_fsync_cannot_publish_partial_output(tmp_path,monkeypatch):
    m=api();p=tmp_path/'result.json.gz'
    def fail(fd):raise OSError('injected storage failure')
    monkeypatch.setattr(m.os,'fsync',fail)
    with pytest.raises(OSError,match='storage failure'):m.write_json(p,{'a':1})
    assert not p.exists()

def test_tampered_compressed_bytes_fail_before_data_use(tmp_path):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,fixture());raw=bytearray(p.read_bytes());raw[15]^=1;p.write_bytes(raw)
    with pytest.raises(ValueError,match='compressed'):load(m,p,r)

def test_truncated_stream_rejected_even_if_archive_hash_was_rebound(tmp_path):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,fixture());p.write_bytes(p.read_bytes()[:-5]);r['compressed_sha256']=sha(p.read_bytes())
    with pytest.raises(ValueError,match='gzip'):load(m,p,r)

def test_wrong_uncompressed_identity_rejected(tmp_path):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,fixture());r['uncompressed_sha256']='0'*64
    with pytest.raises(ValueError,match='uncompressed'):load(m,p,r)

def test_reader_rejects_nonfinite_JSON_from_external_stream(tmp_path):
    m=api();p=tmp_path/'bad.json.gz';raw=b'{"bad":NaN}';p.write_bytes(gzip.compress(raw,mtime=0))
    with pytest.raises(ValueError):m.read_json(p,compressed_sha256=sha(p.read_bytes()),uncompressed_sha256=sha(raw),max_uncompressed_bytes=8*1024**2)

def test_pack_existing_retains_exact_original_JSON_bytes_and_source(tmp_path):
    m=api();source=tmp_path/'original.json';raw=' { "score" : -0.0, "missing": null, "name":"源域" }\n'.encode();source.write_bytes(raw);p=tmp_path/'copy.json.gz';r=m.pack_existing(source,p,expected_source_sha256=sha(raw))
    assert source.read_bytes()==raw and gzip.decompress(p.read_bytes())==raw
    assert r['uncompressed_sha256']==sha(raw) and load(m,p,r)==json.loads(raw)

def test_pack_rejects_wrong_source_hash_without_committing(tmp_path):
    m=api();source=tmp_path/'original.json';source.write_bytes(b'{"a":1}');p=tmp_path/'copy.json.gz'
    with pytest.raises(ValueError,match='source'):m.pack_existing(source,p,expected_source_sha256='0'*64)
    assert not p.exists() and source.read_bytes()==b'{"a":1}'

def test_source_changed_during_pack_cannot_be_committed(tmp_path,monkeypatch):
    m=api();source=tmp_path/'original.json';raw=b'{"a":1}';source.write_bytes(raw);p=tmp_path/'copy.json.gz';original=m.gzip.GzipFile.write;changed=[]
    def write(stream,data):
        result=original(stream,data)
        if not changed:source.write_bytes(b'{"a":2}');changed.append(True)
        return result
    monkeypatch.setattr(m.gzip.GzipFile,'write',write)
    with pytest.raises(ValueError,match='source'):m.pack_existing(source,p,expected_source_sha256=sha(raw))
    assert not p.exists()

@pytest.mark.parametrize('name',['result.json','result.gz','../outside.json'])
def test_wrong_output_namespace_is_rejected(tmp_path,name):
    m=api();p=tmp_path/name
    with pytest.raises(ValueError):m.write_json(p,{'a':1})
    assert not p.exists()

def test_symlink_source_is_not_accepted_as_a_bound_regular_artifact(tmp_path):
    m=api();target=tmp_path/'real.json';target.write_bytes(b'{"a":1}');link=tmp_path/'link.json';link.symlink_to(target);out=tmp_path/'copy.json.gz'
    with pytest.raises(ValueError):m.pack_existing(link,out,expected_source_sha256=sha(target.read_bytes()))
    assert not out.exists()


def test_archive_changed_during_decode_fails_even_if_raw_payload_is_identical(tmp_path,monkeypatch):
    m=api();p=tmp_path/'result.json.gz';r=m.write_json(p,{'a':1});original=m.gzip.GzipFile.read;changed=[]
    def read(stream,*args):
        result=original(stream,*args)
        if not changed:
            raw=bytearray(p.read_bytes());raw[4]^=1;p.write_bytes(raw);changed.append(True)
        return result
    monkeypatch.setattr(m.gzip.GzipFile,'read',read)
    with pytest.raises(ValueError,match='compressed'):load(m,p,r)
