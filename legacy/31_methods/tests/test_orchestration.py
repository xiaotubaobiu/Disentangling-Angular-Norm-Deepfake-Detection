from pathlib import Path
from deepfake_detector import orchestration as o
import json
import pytest


def test_physical_devices_reject_jev_cards_and_duplicate_indices():
    assert hasattr(o,'validate_devices'), 'physical device gate not implemented'
    assert o.validate_devices([4,5,6,7])==[4,5,6,7]
    for devices in [[0],[3,4],[4,4],[8],[]]:
        try:o.validate_devices(devices)
        except ValueError:pass
        else:raise AssertionError(f'Unsafe device set accepted: {devices}')


def test_lock_contention_rejects_duplicate_executor(tmp_path):
    assert hasattr(o,'exclusive_lock'), 'exclusive experiment lock not implemented'
    with o.exclusive_lock(tmp_path/'queue.lock'):
        try:
            with o.exclusive_lock(tmp_path/'queue.lock'):pass
        except BlockingIOError:pass
        else:raise AssertionError('Second executor must not acquire active lock')


def test_gpu_launch_requires_frozen_data_and_cpu_gates(tmp_path):
    assert hasattr(o,'require_gpu_gates'), 'GPU prerequisite gate not implemented'
    try:o.require_gpu_gates(tmp_path)
    except ValueError:pass
    else:raise AssertionError('Missing data/CPU gates must prohibit service stop and GPU launch')


def test_pause_failure_after_stop_still_restores_owned_original_service(tmp_path,monkeypatch):
    assert hasattr(o,'llm_session'), 'exception-safe service lifecycle missing'
    restored=[]
    record={'owner_pid':o.os.getpid(),'state':'PREPARED','original':{'state':{'Running':True}}}
    def fail_pause(root,devices):
        (root/'service_state.json').write_text(json.dumps(record))
        raise ValueError('idle verification failed after stop')
    monkeypatch.setattr(o,'pause_llm',fail_pause)
    monkeypatch.setattr(o,'restore_llm',lambda root,r:restored.append(r))
    with pytest.raises(ValueError,match='idle verification failed'):
        with o.llm_session(tmp_path,[4,5,6,7]):raise AssertionError('must not launch')
    assert restored==[record]


def test_session_does_not_restore_stale_journal_if_gates_reject(tmp_path,monkeypatch):
    assert hasattr(o,'llm_session'), 'exception-safe service lifecycle missing'
    (tmp_path/'service_state.json').write_text(json.dumps({'owner_pid':-1,'state':'PREPARED'}))
    monkeypatch.setattr(o,'pause_llm',lambda *args:(_ for _ in ()).throw(ValueError('missing gates')))
    monkeypatch.setattr(o,'restore_llm',lambda *args:pytest.fail('unowned stale journal must not restore'))
    with pytest.raises(ValueError,match='missing gates'):
        with o.llm_session(tmp_path,[4,5,6,7]):pass


def test_frozen_manifest_gate_rejects_mutation_and_missing_required_roles(tmp_path):
    from deepfake_detector.manifest import digest
    assert hasattr(o,'validate_frozen_manifests'), 'frozen manifest enforcement missing'
    data=tmp_path/'data';data.mkdir()
    files=['train.parquet','ff_test.parquet','cdf_test.parquet']
    for name in files:(data/name).write_bytes(b'frozen')
    marker={'status':'FROZEN','manifest_sha256':{name:digest(data/name) for name in files},'calibration_available':False}
    (data/'freeze_complete.json').write_text(json.dumps(marker))
    assert o.validate_frozen_manifests(tmp_path)==marker
    (data/'train.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):o.validate_frozen_manifests(tmp_path)
    marker['manifest_sha256'].pop('train.parquet')
    (data/'freeze_complete.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='mandatory'):o.validate_frozen_manifests(tmp_path)
