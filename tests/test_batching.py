"""Every scheduling count must fit its captured graph and allocated slot arena."""
import pytest
from qwen3_tts_engine.batching import decode_batch_size


def test_all_active_counts_fit_buckets_without_exceeding_arena():
    for slots in [1, 2, 4, 8, 16, 32, 64]:
        for step in [0] + [x for x in [1, 2, 4, 8, 16, 32, 64] if x <= slots]:
            for active in range(1, slots + 1):
                batch = decode_batch_size(active, slots, step)
                assert active <= batch <= slots
                if step and active > step:
                    assert batch % step == 0 and batch - active < step
                else:
                    assert batch & (batch - 1) == 0
                    assert batch == active or batch // 2 < active
    assert decode_batch_size(33, 64) == 64
    assert decode_batch_size(33, 64, 8) == 40
    assert decode_batch_size(17, 64, 8) == 24


@pytest.mark.parametrize('active,slots,step', [(0, 64, 8), (65, 64, 8), (1, 3, 0),
                                                (1, 64, -1), (1, 64, 3), (1, 8, 16)])
def test_invalid_scheduling_geometry_is_rejected(active, slots, step):
    with pytest.raises(ValueError):
        decode_batch_size(active, slots, step)
