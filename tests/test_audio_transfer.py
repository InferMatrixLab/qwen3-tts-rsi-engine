"""Batched host transfer must retain per-request PCM, order and frame accounting."""
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip('torch_npu')
pytest.importorskip('qwen_tts')
from qwen3_tts_engine.continuous import ContinuousModel


class Codec:
    def decode(self, indices, codes):
        # Distinct deterministic content and unequal lengths expose row/offset mixups.
        samples = codes.shape[-1] * 1920
        return torch.stack([torch.linspace(-1.2, 1.2, samples) + index / 100
                            for index in indices]).unsqueeze(1).to(torch.bfloat16)


def exercise(batched):
    model = object.__new__(ContinuousModel)
    model.batch_audio_transfer = batched
    model.codec_codes = None
    model.incremental_codec = True
    model.codec_batch_size = 2
    model.codec = Codec()
    model.states = [SimpleNamespace(identifier=i, frames=length, emitted=0,
                                   codes=[torch.full((16,), i) for _ in range(length)], audio=[])
                    for i, length in enumerate([2, 4, 2, 8, 4, 2])]
    emitted = []
    model.emit_audio(model.states, lambda *args: emitted.append(args))
    assert all(state.emitted == state.frames for state in model.states)
    assert all(len(state.audio[0]) == state.frames * 1920 * 2 for state in model.states)
    return emitted, [state.audio for state in model.states]


def test_grouped_transfer_preserves_ragged_batch_pcm_and_callback_order():
    assert exercise(True) == exercise(False)
