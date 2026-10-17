"""Check residual lifetime and attention/cache contract through the optional fusion."""
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip('torch_npu')
import torch_npu
from qwen3_tts_engine.fused_decoder import FusedResidualDecoder


class Attention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        self.arguments = kwargs
        return hidden_states * 2, 'attention-weights'


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = Attention()
        self.post_attention_layernorm = SimpleNamespace(mode='native', weight=torch.ones(8), variance_epsilon=1e-6)
        self.mlp = torch.nn.Identity()
        self.attention_type = 'full_attention'

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + 7,)


def test_fusion_preserves_residual_and_forwards_cache(monkeypatch):
    layer = Layer()
    candidate = FusedResidualDecoder(layer)
    def fused(x, residual, weight, epsilon):
        total = x + residual
        return total / 2, None, total
    monkeypatch.setattr(torch_npu, 'npu_add_rms_norm', fused)
    inputs = torch.randn(3, 2, 8)
    cache, positions, mask = object(), object(), object()
    result = candidate(inputs, past_key_values=cache, cache_position=positions,
                       attention_mask=mask, output_attentions=True)
    torch.testing.assert_close(result[0], inputs * 4.5)
    assert result[1] == 'attention-weights'
    assert layer.self_attn.arguments['past_key_values'] is cache
    assert layer.self_attn.arguments['cache_position'] is positions
    assert layer.self_attn.arguments['attention_mask'] is mask
    assert candidate.attention_type == 'full_attention'


def test_single_reference_bypasses_fusion(monkeypatch):
    candidate = FusedResidualDecoder(Layer(), reference_single=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Single reference row must not use fusion')
    monkeypatch.setattr(torch_npu, 'npu_add_rms_norm', forbidden)
    inputs = torch.randn(1, 2, 8)
    torch.testing.assert_close(candidate(inputs)[0], inputs + 7)


def test_non_native_normalization_rejected():
    layer = Layer()
    layer.post_attention_layernorm.mode = 'reference-rounding'
    with pytest.raises(ValueError, match='native RMSNorm'):
        FusedResidualDecoder(layer)
