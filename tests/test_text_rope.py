"""CustomVoice axis selection must preserve upstream MRoPE arithmetic exactly."""
import pytest
import torch

pytest.importorskip('qwen_tts')
from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb, apply_rotary_pos_emb


@pytest.mark.parametrize('interleaved', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_equal_text_axes_match_multimodal_rope(interleaved, dtype):
    torch.manual_seed(42)
    for batch, length in [(1, 1), (4, 17)]:
        q = torch.randn(batch, 16, length, 128, dtype=dtype)
        k = torch.randn(batch, 8, length, 128, dtype=dtype)
        phase = torch.randn(batch, length, 64).repeat(1, 1, 2)
        cos = phase.cos().to(dtype).unsqueeze(0).expand(3, -1, -1, -1)
        sin = phase.sin().to(dtype).unsqueeze(0).expand(3, -1, -1, -1)
        expected = apply_multimodal_rotary_pos_emb(q, k, cos, sin, [24, 20, 20], interleaved)
        actual = apply_rotary_pos_emb(q, k, cos[0], sin[0])
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize('scope,expected', [('all', [True, True]), ('talker', [True, False]), ('predictor', [False, True])])
def test_native_rotary_scope_selects_model_component(scope, expected):
    from qwen3_tts_engine.packed_attention import PackedAttention
    selected = []
    for name in ['Qwen3TTSTalkerAttention', 'Qwen3TTSAttention']:
        reference = type(name, (torch.nn.Module,), {})()
        reference.q_proj = torch.nn.Linear(8, 8, bias=False)
        reference.k_proj = torch.nn.Linear(8, 4, bias=False)
        reference.v_proj = torch.nn.Linear(8, 4, bias=False)
        candidate = PackedAttention(reference, native_rope=True, text_only_rope=True, native_rope_scope=scope)
        selected.append(candidate.native_rope)
    assert selected == expected
