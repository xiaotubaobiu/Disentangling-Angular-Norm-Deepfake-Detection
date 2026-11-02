"""Parallel scheduling must retain the complete frozen-input integrity contract."""
from collections import Counter
import importlib.util
import os
from pathlib import Path
from threading import Lock

import pandas as pd
import pytest

from journal_detector import cache_d, trainer


@pytest.fixture
def validators(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/run_parallel_d_validation.py'
    spec=importlib.util.spec_from_file_location('runtime_validation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(cache_d,'_validate_inputs',cache_d._validate_inputs)
    return module.install_validation_threads(32)


@pytest.fixture
def inputs(tmp_path):
    rows=[]
    # 65 makes a conflicting duplicate at the end cross a naive row-index shard.
    for index in range(65):
        path=tmp_path/f'{index:03d}.bin';path.write_bytes(f'original-{index:03d}'.encode())
        stat=path.stat();rows.append(dict(image_path=str(path),content_sha256=trainer.digest(path),
            bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return pd.DataFrame(rows)


def test_parallel_validator_checks_every_unique_path_once(inputs,validators,monkeypatch):
    _,parallel=validators;seen=Counter();lock=Lock()
    class RecordingPath(type(Path())):
        def stat(self,*args,**kwargs):
            with lock:seen[str(self)]+=1
            return super().stat(*args,**kwargs)
    monkeypatch.setattr(cache_d,'Path',RecordingPath)
    duplicated=pd.concat([inputs,inputs.iloc[[0]]],ignore_index=True)
    parallel(duplicated)
    assert seen==Counter({name:1 for name in inputs.image_path})


def test_conflicting_duplicate_is_rejected_across_potential_row_shards(inputs,validators):
    original,parallel=validators
    frame=pd.concat([inputs,inputs.iloc[[0]]],ignore_index=True)
    frame.loc[len(frame)-1,'content_sha256']='0'*64
    for validator in [original,parallel]:
        with pytest.raises(ValueError,match='conflicting frozen hash identity'):validator(frame)


def test_changed_stat_with_same_bytes_is_rehashed(inputs,validators,monkeypatch):
    _,parallel=validators;path=Path(inputs.iloc[0].image_path);stat=path.stat()
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
    digest=trainer.digest;seen=[]
    def record(name):
        seen.append(str(name));return digest(name)
    monkeypatch.setattr(trainer,'digest',record)
    parallel(inputs)
    assert seen==[str(path)]


def test_changed_bytes_are_rejected(inputs,validators):
    _,parallel=validators;path=Path(inputs.iloc[0].image_path);path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='byte hash identity mismatch'):parallel(inputs)


def test_missing_file_is_rejected(inputs,validators):
    _,parallel=validators;Path(inputs.iloc[0].image_path).unlink()
    with pytest.raises(ValueError,match='input image identity is unavailable'):parallel(inputs)
