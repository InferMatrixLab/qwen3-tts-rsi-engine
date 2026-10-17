"""Single-row reference dispatch must not silently disable fusion for larger batches."""
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip('torch_npu')
pytest.importorskip('qwen_tts')
import torch_npu
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSRMSNorm, Qwen3TTSTalkerTextMLP
from qwen3_tts_engine.operators import AscendRMSNorm
from qwen3_tts_engine.packed_mlp import PackedMLP


def test_single_rms_uses_reference_and_larger_batch_uses_requested_native_operator(monkeypatch):
    reference = Qwen3TTSRMSNorm(32).eval()
    candidate = AscendRMSNorm(reference, 'native', reference_single=True)
    calls = []
    def native(inputs, weight, epsilon):
        calls.append(tuple(inputs.shape))
        return (torch.zeros_like(inputs),)
    monkeypatch.setattr(torch_npu, 'npu_rms_norm', native)
    inputs = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(candidate(inputs), reference(inputs), atol=0, rtol=0)
    assert calls == []
    assert not candidate(inputs.expand(2, -1, -1)).count_nonzero()
    assert calls == [(2, 3, 32)]


def test_single_mlp_keeps_reference_rounding_but_larger_batch_uses_swiglu(monkeypatch):
    reference = Qwen3TTSTalkerTextMLP(SimpleNamespace(hidden_size=32, intermediate_size=64, hidden_act='silu')).eval()
    candidate = PackedMLP(reference, fused=True, reference_single=True)
    control = PackedMLP(reference)
    calls = []
    def fused(value):
        calls.append(value.shape[0])
        gate, up = value.chunk(2, -1)
        return torch.nn.functional.silu(gate) * up
    monkeypatch.setattr(torch_npu, 'npu_swiglu', fused)
    inputs = torch.randn(1, 3, 32)
    torch.testing.assert_close(candidate(inputs), control(inputs), atol=0, rtol=0)
    assert calls == []
    candidate(inputs.expand(2, -1, -1))
    assert calls == [2]


@pytest.mark.parametrize('already_active,expected', [(False, []), (True, [False, True])])
def test_prefill_override_restores_reference_dispatch_after_failure(monkeypatch, already_active, expected):
    from qwen3_tts_engine.continuous import ContinuousModel
    monkeypatch.setattr(torch.npu, 'set_device', lambda device: None)
    model = object.__new__(ContinuousModel)
    model.device = 'npu:0'
    model.states = [SimpleNamespace(identifier=0)] if already_active else []
    model.reference_single = True
    model.batch_prefill = model.graph_prefill = True
    calls = []
    model.model = SimpleNamespace(set_single_reference=calls.append)
    def fail(submissions):
        raise RuntimeError('prefill failure')
    model.admit_batch = fail
    with pytest.raises(RuntimeError, match='prefill failure'):
        model.tick([(1, {})])
    assert calls == expected
