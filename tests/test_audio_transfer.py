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

    def decode_groups(self, groups, on_output=None):
        outputs = []
        for index, (indices, codes) in enumerate(groups):
            outputs.append(self.decode(indices, codes))
            if on_output is not None:
                on_output(index, outputs[-1])
        return outputs


def exercise(batched, streams=1, lengths=None, forbid_parallel=False):
    model = object.__new__(ContinuousModel)
    model.batch_audio_transfer = batched
    model.codec_streams = streams
    model.codec_codes = None
    model.incremental_codec = True
    model.codec_batch_size = 2
    model.codec = Codec()
    if forbid_parallel:
        def unexpected(*_args, **_kwargs):
            raise AssertionError('single group must use direct owner delivery')
        model.codec.decode_groups = unexpected
    model.states = [SimpleNamespace(identifier=i, frames=length, emitted=0,
                                   codes=[torch.full((16,), i) for _ in range(length)], audio=[])
                    for i, length in enumerate(lengths or [2, 4, 2, 8, 4, 2])]
    emitted = []
    model.emit_audio(model.states, lambda *args: emitted.append(args))
    assert all(state.emitted == state.frames for state in model.states)
    assert all(len(state.audio[0]) == state.frames * 1920 * 2 for state in model.states)
    return emitted, [state.audio for state in model.states]


def test_grouped_transfer_preserves_ragged_batch_pcm_and_callback_order():
    assert exercise(True) == exercise(False)


@pytest.mark.parametrize('batched', [False, True])
def test_parallel_group_submission_preserves_ragged_pcm_and_callback_order(batched):
    assert exercise(batched, streams=2) == exercise(batched, streams=1)


@pytest.mark.parametrize('batched', [False, True])
def test_single_group_uses_direct_delivery_with_identical_pcm(batched):
    assert exercise(batched, streams=2, lengths=[2, 2], forbid_parallel=True) == exercise(
        batched, streams=1, lengths=[2, 2])


def test_shutdown_logs_mixed_native_and_lane_shapes_and_closes_all_resources():
    from unittest.mock import Mock

    model = object.__new__(ContinuousModel)
    model.samplers = {}
    model.prefills = {}
    model.graphs = {}
    model.incremental_codec = True
    model.codec = Mock(graphs={(1, 2, -1): None, (1, 2, 0): None, (1, 2, 1): None})
    model.predictor = Mock()
    model.model = Mock()
    model.close()
    model.codec.close.assert_called_once_with()
    model.predictor.close.assert_called_once_with()
    model.model.close.assert_called_once_with()
