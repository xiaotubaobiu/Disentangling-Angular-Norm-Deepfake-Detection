"""Large immutable files may reuse a digest only inside a sealed transaction."""
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/reproducibility/access_20261103/verified_digest.py'
    assert path.is_file(), 'verified digest transaction is not implemented'
    spec = importlib.util.spec_from_file_location('verified_digest', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stock(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_reused_hashes_are_fully_recomputed_before_sealing(tmp_path):
    m = module()
    p = tmp_path / 'weights'; p.write_bytes(b'original model bytes')
    calls = []
    def reader(path):
        calls.append(str(path)); return stock(path)
    memo = m.DigestTransaction(reader, min_bytes=0)
    assert memo.digest(p) == memo.digest(p) == stock(p)
    assert len(calls) == 1
    memo.seal()
    assert len(calls) == 2 and memo.sealed and memo.hits == 1


def test_cached_read_rejects_a_changed_file(tmp_path):
    m = module(); p = tmp_path / 'weights'; p.write_bytes(b'original')
    memo = m.DigestTransaction(stock, min_bytes=0); memo.digest(p)
    p.write_bytes(b'replaced with a different size')
    with pytest.raises(ValueError, match='changed'):
        memo.digest(p)


def test_seal_rejects_mutation_even_without_another_cached_read(tmp_path):
    m = module(); p = tmp_path / 'weights'; p.write_bytes(b'original')
    memo = m.DigestTransaction(stock, min_bytes=0); memo.digest(p)
    p.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='changed'):
        memo.seal()
    assert not memo.sealed


def test_restoring_mtime_does_not_hide_mutation(tmp_path):
    m = module(); p = tmp_path / 'weights'; p.write_bytes(b'original')
    previous = p.stat(); memo = m.DigestTransaction(stock, min_bytes=0); memo.digest(p)
    p.write_bytes(b'replaced'); os.utime(p, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match='changed'):
        memo.seal()


def test_small_files_are_never_memoized(tmp_path):
    m = module(); p = tmp_path / 'config'; p.write_bytes(b'first')
    memo = m.DigestTransaction(stock, min_bytes=1024)
    assert memo.digest(p) == stock(p)
    p.write_bytes(b'second')
    assert memo.digest(p) == stock(p) and memo.hits == 0


def test_final_full_read_catches_content_change_under_coarse_timestamps(tmp_path, monkeypatch):
    m = module(); p = tmp_path / 'weights'; p.write_bytes(b'original')
    fixed = m.signature(p)
    monkeypatch.setattr(m, 'signature', lambda path: fixed)
    memo = m.DigestTransaction(stock, min_bytes=0); memo.digest(p)
    p.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='changed'):
        memo.seal()
    assert not memo.sealed


def test_binding_is_restored_after_a_failed_transaction(tmp_path):
    m = module(); owner = SimpleNamespace(digest=stock)
    p = tmp_path / 'weights'; p.write_bytes(b'original')
    with pytest.raises(RuntimeError, match='failure'):
        with m.guarded_digests([(owner, 'digest')], min_bytes=0):
            owner.digest(p)
            raise RuntimeError('failure')
    assert owner.digest is stock
