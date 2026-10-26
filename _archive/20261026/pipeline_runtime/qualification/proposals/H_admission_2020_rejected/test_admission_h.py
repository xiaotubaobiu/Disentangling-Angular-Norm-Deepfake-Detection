"""CPU-only qualification admission: actual kernel locks, no CUDA workloads."""
import contextlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import pytest
from reliability_v2 import trainer, queue_d, qualification_h


def admission():
    spec=importlib.util.find_spec('reliability_v2.admission_h')
    assert spec is not None, 'H-only post-lock drain admission is missing'
    return importlib.import_module('reliability_v2.admission_h')


def test_post_lock_drain_admission_exists():
    assert callable(admission().owned_drain_qualification)


def lock_child(path, ready, release):
    with trainer.exclusive_file(path):
        ready.set();release.wait(20)


def test_same_kernel_lock_is_visible_to_unchanged_D_reserved(tmp_path):
    a=admission();uuid='GPU-test2'
    with trainer.exclusive_file(tmp_path/'gpu_locks'/f'{uuid}.lock'):
        assert queue_d.reserved(tmp_path,uuid)
        assert a.kernel_lock_owner(tmp_path/'gpu_locks'/f'{uuid}.lock',os.getpid())
    assert not queue_d.reserved(tmp_path,uuid)


def test_crash_releases_kernel_lease_and_duplicate_claim_fails(tmp_path):
    a=admission();path=tmp_path/'gpu_locks/GPU-test2.lock'
    ctx=mp.get_context('fork');ready=ctx.Event();release=ctx.Event()
    child=ctx.Process(target=lock_child,args=(path,ready,release));child.start()
    try:
        assert ready.wait(10)
        assert a.kernel_lock_owner(path,child.pid)
        with pytest.raises(BlockingIOError):
            with trainer.exclusive_file(path):pytest.fail('duplicate lease admitted')
        child.kill();child.join(10)
        assert not child.is_alive()
        with trainer.exclusive_file(path):assert a.kernel_lock_owner(path,os.getpid())
    finally:
        if child.is_alive():child.kill();child.join()


def test_competing_real_processes_never_share_lease(tmp_path):
    admission();ctx=mp.get_context('fork');ready=ctx.Event();release=ctx.Event()
    path=tmp_path/'gpu_locks/GPU-test3.lock'
    child=ctx.Process(target=lock_child,args=(path,ready,release));child.start()
    try:
        assert ready.wait(10)
        for _ in range(100):assert queue_d.reserved(tmp_path,'GPU-test3')
        release.set();child.join(10)
        assert child.exitcode==0
        assert not queue_d.reserved(tmp_path,'GPU-test3')
    finally:
        if child.is_alive():child.kill();child.join()


def test_no_prelease_or_transition_only_witness_is_accepted(tmp_path,monkeypatch):
    a=admission();config={'artifact_root':str(tmp_path),'allowed_gpus':[0,1,2,3],'allowed_gpu_uuids':['u0','u1','u2','u3']}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','u2')
    with pytest.raises(ValueError,match='lease'):
        a.drain_witness(tmp_path,config,{'schema':'invalid'})


def test_acceptance_uses_frozen_data_bindings_but_full_GPU_proofs(tmp_path,monkeypatch):
    calls=[]
    def context(root,*,deep=False):
        calls.append(('data',deep));assert deep is False,'accepted H data was fully scanned again'
        return {},{},{}
    monkeypatch.setattr(qualification_h,'_context',context)
    monkeypatch.setattr(qualification_h,'_probe_receipt',lambda *a:{})
    def proofs(*args,deep=True,**kwargs):
        calls.append(('proofs',deep));assert deep is True
        raise ValueError('missing genuine six GPU proofs')
    monkeypatch.setattr(qualification_h,'_qualified_backbones',proofs)
    with pytest.raises(ValueError,match='six GPU'):
        qualification_h.accept_qualification(tmp_path)
    assert calls==[('data',False),('proofs',True)]
    assert not (tmp_path/'qualification/H_gate.json').exists()
