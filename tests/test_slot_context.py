"""Trimming masked KV tails must preserve live history and causal attention."""
from types import SimpleNamespace
import pytest
import torch

pytest.importorskip('torch_npu')
pytest.importorskip('qwen_tts')
from qwen3_tts_engine.continuous import SlotCache


@pytest.mark.parametrize('valid_length', [False, True])
@pytest.mark.parametrize('context,positions', [(64, [63, 17]), (128, [64, 127])])
def test_bucket_cache_preserves_history_and_attention(context, positions, valid_length):
    torch.manual_seed(42)
    keys, values = torch.randn(2, 2, 256, 8), torch.randn(2, 2, 256, 8)
    arena = SimpleNamespace(capacity=256, native_kv_update=False, triton_kv_update=False, keys=[keys.clone()], values=[values.clone()])
    position = torch.tensor(positions)
    cache = SlotCache(arena, position, context, valid_length)
    new_keys, new_values = torch.randn(2, 2, 1, 8), torch.randn(2, 2, 1, 8)
    actual_k, actual_v = cache.update(new_keys, new_values, 0, {})
    for row, offset in enumerate(positions):
        keys[row, :, offset] = new_keys[row, :, 0]
        values[row, :, offset] = new_values[row, :, 0]
    torch.testing.assert_close(arena.keys[0], keys, atol=0, rtol=0)
    torch.testing.assert_close(arena.values[0], values, atol=0, rtol=0)
    returned_length = 256 if valid_length else context
    assert actual_k.shape == (2, 2, returned_length, 8)
    if valid_length:
        assert actual_k.data_ptr() == arena.keys[0].data_ptr()
        assert actual_k.is_contiguous() and actual_v.is_contiguous()
        assert cache.actual_seq_lengths_kv == [context, context]
    query = torch.randn(2, 2, 1, 8)
    mask = torch.arange(256)[None, :] <= position[:, None]
    full = torch.nn.functional.scaled_dot_product_attention(query, keys, values, mask[:, None, None])
    trimmed = torch.nn.functional.scaled_dot_product_attention(query, actual_k, actual_v, mask[:, None, None, :returned_length])
    torch.testing.assert_close(trimmed, full, atol=1e-6, rtol=1e-6)
