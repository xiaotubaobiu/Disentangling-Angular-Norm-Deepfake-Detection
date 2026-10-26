import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'experiments/2026-09-23_e3_smolvlm_candidate/scripts/train_candidate.py'
spec = importlib.util.spec_from_file_location('e3_train_contract', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_formal_training_rejects_smoke_batch_limits():
    with pytest.raises(SystemExit):
        module.parse_args(['--seed', '42', '--max-train-batches', '1'])
    args = module.parse_args(['--seed', '42', '--smoke', '--max-train-batches', '3'])
    assert args.max_train_batches == 3


def test_runtime_rejects_other_gpus_and_incomplete_formal_world():
    with pytest.raises(ValueError, match='0,1,2,3'):
        module.validate_runtime('4,5,6,7', 4)
    with pytest.raises(ValueError, match='four'):
        module.validate_runtime('0,1,2,3', 1)
    module.validate_runtime('0,1,2,3', 4)


def test_protocol_rejects_modified_file_even_when_config_is_unchanged(tmp_path):
    code = tmp_path / 'module.py'
    code.write_text('original')
    manifest = {'files': {'module.py': module.sha256(code)}}
    module.verify_protocol(manifest, tmp_path)
    code.write_text('changed')
    with pytest.raises(ValueError, match='module.py'):
        module.verify_protocol(manifest, tmp_path)
