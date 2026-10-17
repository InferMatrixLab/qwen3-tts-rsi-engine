"""Trimming masked KV tails must preserve live history and causal attention."""
from types import SimpleNamespace
import pytest
import torch

pytest.importorskip('torch_npu')
pytest.importorskip('qwen_tts')
from qwen3_tts_engine.continuous import SlotCache


@pytest.mark.parametrize('context,positions', [(64, [63, 17]), (128, [64, 127])])
def test_bucket_cache_updates_full_arena_but_returns_only_valid_bucket(context, positions):
    torch.manual_seed(42)
    keys, values = torch.randn(2, 2, 256, 8), torch.randn(2, 2, 256, 8)
    arena = SimpleNamespace(capacity=256, native_kv_update=False, triton_kv_update=False, keys=[keys.clone()], values=[values.clone()])
    position = torch.tensor(positions)
    cache = SlotCache(arena, position, context)
    new_keys, new_values = torch.randn(2, 2, 1, 8), torch.randn(2, 2, 1, 8)
    actual_k, actual_v = cache.update(new_keys, new_values, 0, {})
    for row, offset in enumerate(positions):
        keys[row, :, offset] = new_keys[row, :, 0]
        values[row, :, offset] = new_values[row, :, 0]
    torch.testing.assert_close(arena.keys[0], keys, atol=0, rtol=0)
    torch.testing.assert_close(arena.values[0], values, atol=0, rtol=0)
    assert actual_k.shape == (2, 2, context, 8)
    query = torch.randn(2, 2, 1, 8)
    mask = torch.arange(256)[None, :] <= position[:, None]
    full = torch.nn.functional.scaled_dot_product_attention(query, keys, values, mask[:, None, None])
    trimmed = torch.nn.functional.scaled_dot_product_attention(query, actual_k, actual_v, mask[:, None, None, :context])
    torch.testing.assert_close(trimmed, full, atol=1e-6, rtol=1e-6)
