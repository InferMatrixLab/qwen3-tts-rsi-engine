"""Projection packing must preserve the upstream gate order and MLP contract."""
from types import SimpleNamespace
import pytest
import torch

pytest.importorskip('qwen_tts')
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerTextMLP
from qwen3_tts_engine.packed_mlp import PackedMLP


def test_projection_packing_matches_upstream_over_batch_and_sequence_shapes():
    torch.manual_seed(42)
    original = Qwen3TTSTalkerTextMLP(SimpleNamespace(hidden_size=32, intermediate_size=64, hidden_act='silu')).eval()
    candidate = PackedMLP(original).eval()
    with torch.inference_mode():
        for shape in [(1, 1, 32), (4, 2, 32), (8, 17, 32)]:
            value = torch.randn(shape)
            torch.testing.assert_close(candidate(value), original(value), atol=1e-6, rtol=1e-6)
