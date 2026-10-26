from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]


def record(method, video, frame):
    from data.index_ffpp import FrameRecord
    return FrameRecord(method, video, str(frame), f'/data/{method}/{video}/{frame}.png', None, int(method != 'real'))


def test_record_selection_deduplicates_and_is_input_order_independent():
    from experiment_data import select_records
    records = [record(m, v, f) for m in ('real', 'simswap') for v in ('a', 'b', 'c') for f in range(7)]
    one = select_records(records + [records[0]], max_fake_videos=2, max_real_videos=None, frames=4)
    two = select_records(list(reversed(records)), max_fake_videos=2, max_real_videos=None, frames=4)
    assert [r.frame_path for r in one] == [r.frame_path for r in two]
    assert len(one) == 20
    assert len({r.video_id for r in one if r.label == 0}) == 3


def test_disjointness_checks_videos_even_with_different_frames():
    from experiment_data import assert_disjoint
    with pytest.raises(ValueError, match='video overlap'):
        assert_disjoint({'train': [record('real', 'a', 0)], 'val': [record('real', 'a', 7)]})
    assert_disjoint({'train': [record('real', 'a', 0)], 'val': [record('real', 'b', 0)]})


@pytest.mark.parametrize('visible', ['', '4', '0,4', 'GPU-abc'])
def test_gpu_guard_rejects_unapproved_devices(visible):
    from experiment_data import validate_gpu
    with pytest.raises(ValueError):
        validate_gpu(visible)


def test_epoch_draws_equal_across_arms_and_balanced_across_tasks_families():
    from experiment_data import epoch_draws
    records = [record(m, str(v), 0) for m in ('real', 'simswap', 'fomm') for v in range(9)]
    left = epoch_draws(records, seed=42, epoch=1, count=4096)
    assert left == epoch_draws(records, seed=42, epoch=1, count=4096)
    assert left != epoch_draws(records, seed=42, epoch=2, count=4096)
    from collections import Counter
    tasks = Counter(t for _, t in left)
    families = Counter(records[i].method for i, _ in left)
    assert max(tasks.values()) - min(tasks.values()) <= 1
    assert max(families.values()) - min(families.values()) <= 1
