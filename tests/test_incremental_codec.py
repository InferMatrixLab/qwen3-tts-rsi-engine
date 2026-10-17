"""Causal state invariants independently checked against complete convolutions."""
import pytest
import torch

pytest.importorskip('qwen_tts')
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
)
from qwen3_tts_engine.incremental_codec import CodecState, IncrementalCodec


@pytest.mark.parametrize('kind', ['dilated', 'transpose', 'pointwise'])
def test_chunked_convolution_matches_complete_and_state_is_independent(kind):
    torch.manual_seed(123)
    if kind == 'transpose':
        module = Qwen3TTSTokenizerV2CausalTransConvNet(3, 5, 16, 8)
    else:
        module = Qwen3TTSTokenizerV2CausalConvNet(3, 5, 7 if kind == 'dilated' else 1,
                                               dilation=9 if kind == 'dilated' else 1)
    module.eval()
    engine = IncrementalCodec(None)
    inputs = [torch.randn(2, 3, 19), torch.randn(2, 3, 19)]
    states = [CodecState(), CodecState()]
    pieces = [[], []]
    with torch.inference_mode():
        for start, end in [(0, 1), (1, 3), (3, 11), (11, 19)]:
            for i in range(2):
                pieces[i].append(engine.run_module(module, inputs[i][..., start:end], states[i]))
        for i in range(2):
            torch.testing.assert_close(torch.cat(pieces[i], dim=-1), module(inputs[i]), atol=1e-6, rtol=1e-6)
        # A fresh stream must not inherit convolution tails from either stream.
        torch.testing.assert_close(engine.run_module(module, inputs[0], CodecState()), module(inputs[0]))
