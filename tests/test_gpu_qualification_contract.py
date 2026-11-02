"""Guard against falsely accepting reduced or unsuccessful GPU probes."""
import importlib.util
from pathlib import Path

import pytest


def qualification():
    path = Path(__file__).resolve().parents[1] / 'scripts/qualify_journal_gpu.py'
    assert path.is_file(), 'GPU qualification implementation is required'
    spec = importlib.util.spec_from_file_location('qualify_journal_gpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def successful_result():
    return dict(
        batch_size=128, microbatch=128, precision='AMP',
        target_used=False, source_batch_checked=True,
        initial_raw_equals_ema=True, parameter_count=10,
        trainable_parameter_count=10, raw_changed=True, ema_changed=True,
        peak_memory_allocated_bytes=100, total_device_memory_bytes=200,
        events=[dict(success=True, skip=False, loss=.7, gradient_finite=True),
                dict(success=True, skip=False, loss=.6, gradient_finite=True)],
    )


def test_rejects_success_claim_after_only_one_actual_optimizer_update():
    result = successful_result()
    result['events'][0].update(success=False, skip=True, gradient_finite=False)
    with pytest.raises(ValueError, match='two successful'):
        qualification().validate_result(result)


@pytest.mark.parametrize('field,value', [
    ('microbatch', 64), ('batch_size', 64), ('target_used', True),
    ('source_batch_checked', False), ('initial_raw_equals_ema', False),
    ('raw_changed', False), ('ema_changed', False),
    ('trainable_parameter_count', 9),
])
def test_rejects_changed_recipe_or_missing_evidence(field, value):
    result = successful_result()
    result[field] = value
    with pytest.raises(ValueError):
        qualification().validate_result(result)


def test_rejects_nonfinite_successful_gradient():
    result = successful_result()
    result['events'][1]['gradient_finite'] = False
    with pytest.raises(ValueError, match='gradient'):
        qualification().validate_result(result)


def test_retains_visible_amp_skip_before_two_successful_updates():
    result = successful_result()
    result['events'].insert(0, dict(success=False, skip=True, loss=.8,
                                    gradient_finite=False))
    qualification().validate_result(result)
    assert len(result['events']) == 3


def test_device_binding_rejects_otherwise_approved_gpu_one():
    config = {'allowed_gpus': [0, 1, 2, 3],
              'allowed_gpu_uuids': ['GPU-zero', 'GPU-one', 'GPU-two', 'GPU-three']}
    inventory = [{'index': 0, 'uuid': 'GPU-zero'}, {'index': 1, 'uuid': 'GPU-one'}]
    with pytest.raises(ValueError, match='GPU 0'):
        qualification().validate_device_binding('GPU-one', config, inventory)


def test_device_binding_requires_uuid_to_match_physical_gpu_zero():
    config = {'allowed_gpus': [0, 1, 2, 3], 'allowed_gpu_uuids': ['GPU-zero']}
    inventory = [{'index': 0, 'uuid': 'GPU-actual'}]
    with pytest.raises(ValueError):
        qualification().validate_device_binding('GPU-zero', config, inventory)
