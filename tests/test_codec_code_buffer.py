"""Code history must survive ring wrap, ragged progress and slot compaction."""
import pytest
import torch
from qwen3_tts_engine.codec_code_buffer import CodecCodeBuffer


def test_interleaved_history_wrap_and_compaction():
    buffer = CodecCodeBuffer(4, 16, 'cpu')
    history = [[], [], []]
    # Slots have different prior progress, as after staggered admissions.
    for tick in range(19):
        positions = torch.tensor([tick, tick + 3, tick + 7])
        codes = torch.arange(48).view(3, 16) + tick * 100
        buffer.write(codes, positions)
        for i in range(3): history[i].append(codes[i].clone())
        if tick >= 7:
            actual = buffer.read([2, 0], [tick, tick - 7], 8)
            expected = torch.stack([torch.stack(history[i][-8:]).T for i in [2, 0]])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    buffer.move(2, 0)
    torch.testing.assert_close(buffer.read([0], [18], 8),
                               torch.stack(history[2][-8:]).T.unsqueeze(0), atol=0, rtol=0)
    # Reused slot starts afresh; stale values must not leak into its valid prefix.
    for step in range(2):
        buffer.write(torch.full((1, 16), 900 + step), torch.tensor([step]))
    expected = torch.stack([torch.full((16,), 900), torch.full((16,), 901)], -1).unsqueeze(0)
    torch.testing.assert_close(buffer.read([0], [0], 2), expected, atol=0, rtol=0)


@pytest.mark.parametrize('slots,starts,length', [([], [], 2), ([0], [0], 9), ([4], [0], 2), ([0], [-1], 2)])
def test_invalid_intervals_rejected(slots, starts, length):
    with pytest.raises(ValueError): CodecCodeBuffer(4, 16, 'cpu').read(slots, starts, length)
