import os
from pathlib import Path
import pytest


def test_idle_gpu_allocation_is_confined_to_physical_zero_through_three():
    from journal_detector.execution import eligible_gpus
    config={'allowed_gpus':[0,1,2,3],'allowed_gpu_uuids':['a','b','c','d']}
    hardware=[{'index':i,'uuid':u,'memory':0} for i,u in enumerate('abcdefgh')]
    assert eligible_gpus(config,hardware,{'b'},{'c'})==['a','d']
    hardware[0]['memory']=200
    assert eligible_gpus(config,hardware,{'b'},{'c'})==['d']
    config['allowed_gpus']=[0,1,2,4]
    with pytest.raises(ValueError):eligible_gpus(config,hardware,set(),set())


def test_queue_waits_for_both_data_freeze_and_code_gate(tmp_path):
    from journal_detector.execution import ready
    a=tmp_path/'freeze.json';b=tmp_path/'code.json'
    job={'depends_on_files':[str(a),str(b)]}
    assert not ready(job)
    a.write_text('{}');assert not ready(job)
    b.write_text('{}');assert ready(job)


def test_process_identity_verifies_actual_live_handle():
    from journal_detector.execution import process_identity,alive
    token=process_identity(os.getpid())
    assert token and alive(os.getpid(),token)
    assert not alive(os.getpid(),str(int(token)+1))
    assert not alive(999999999,'0')
