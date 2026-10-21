import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from qwen3_tts_engine.request_views import single_request_inputs


class SingleRequestInputTests(unittest.TestCase):
    def state(self, dtype=torch.bfloat16, strided=False, seed=42):
        factor = 2 if strided else 1
        generator = torch.Generator().manual_seed(seed)
        storage = torch.zeros(129, 16 * factor)
        storage[:, ::factor].copy_(torch.rand(129, 16, generator=generator))
        return SimpleNamespace(
            token=torch.tensor([[seed]]),
            hidden=torch.randn(1, 1, 12 * factor, dtype=dtype)[..., ::factor],
            uniforms=storage[:, ::factor],
            trailing=torch.randn(1, 3, 12 * factor, dtype=dtype)[..., ::factor],
            pad=torch.randn(1, 1, 12 * factor, dtype=dtype)[..., ::factor],
            seen=torch.zeros(1, 31 * factor, dtype=torch.bool)[:, ::factor],
            frames=0,
            position=123 + seed,
            rope_delta={17: -13, 42: 0, 43: 17}[seed],
        )

    def original(self, state):
        frame = state.frames
        text = (
            state.trailing[:, frame : frame + 1]
            if frame < state.trailing.shape[1]
            else state.pad
        )
        return (
            torch.cat([state.token]),
            torch.cat([state.hidden]),
            torch.stack([state.uniforms[frame]]),
            torch.cat([text]),
            torch.stack([state.uniforms[frame + 1, 0]]),
            torch.cat([state.seen]),
            torch.tensor([state.position]),
            torch.tensor([state.position + state.rope_delta])
            .view(1, 1, 1)
            .expand(3, -1, -1),
            torch.tensor([frame + 1]),
        )

    def test_boundary_frames_match_original_with_changed_strided_sources(self):
        for dtype in (torch.bfloat16, torch.float32):
            for strided in (False, True):
                for seed in (17, 42, 43):
                    state = self.state(dtype, strided, seed)
                    originals = {
                        name: getattr(state, name).clone()
                        for name in (
                            "token",
                            "hidden",
                            "uniforms",
                            "trailing",
                            "pad",
                            "seen",
                        )
                    }
                    for frame in (0, 1, 2, 7, 63, 127):
                        with self.subTest(
                            dtype=dtype, strided=strided, seed=seed, frame=frame
                        ):
                            state.frames = frame
                            expected = self.original(state)
                            actual = single_request_inputs(state)
                            self.assertEqual(len(expected), len(actual))
                            for left, right in zip(expected, actual):
                                self.assertEqual(left.shape, right.shape)
                                self.assertEqual(left.dtype, right.dtype)
                                self.assertTrue(torch.equal(left, right))
                            self.assertTrue(
                                all(
                                    torch.equal(getattr(state, name), value)
                                    for name, value in originals.items()
                                )
                            )

    def test_consumed_outputs_are_owned_while_borrowed_sources_stay_live(self):
        state = self.state(strided=True)
        inputs = single_request_inputs(state)
        self.assertIs(inputs.tokens, state.token)
        self.assertIs(inputs.hidden, state.hidden)
        self.assertIs(inputs.seen, state.seen)
        owned = [torch.empty(x.shape, dtype=x.dtype).copy_(x) for x in inputs]
        expected = [x.clone() for x in owned]
        state.hidden.add_(3)
        state.uniforms.add_(0.01)
        state.seen.fill_(True)
        state.frames = 7
        state.position += 7
        later = single_request_inputs(state)
        for source in later:
            torch.empty(source.shape, dtype=source.dtype).copy_(source)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(owned, expected)))
        self.assertTrue(torch.equal(inputs.hidden, state.hidden))
        self.assertTrue(torch.equal(inputs.draws, state.uniforms[:1]))

    def test_metadata_has_one_integer_storage_and_original_broadcast_values(self):
        state = self.state()
        state.position, state.rope_delta = 1 << 40, -(1 << 35)
        inputs = single_request_inputs(state)
        integers = (inputs.positions, inputs.rotary, inputs.steps)
        self.assertEqual(len({x.untyped_storage().data_ptr() for x in integers}), 1)
        self.assertTrue(all(x.dtype == torch.long for x in integers))
        self.assertEqual(inputs.rotary.shape, (3, 1, 1))
        self.assertEqual(
            inputs.rotary.flatten().tolist(), [state.position + state.rope_delta] * 3
        )
        self.assertEqual(inputs.positions.item(), state.position)
        self.assertEqual(inputs.steps.item(), 1)

    def test_invalid_frame_is_rejected_before_any_metadata_upload(self):
        state = self.state()
        with patch("torch.tensor") as upload:
            for frame in (-1, 128, True, 1.0):
                state.frames = frame
                with self.subTest(frame=frame), self.assertRaises(ValueError):
                    single_request_inputs(state)
            upload.assert_not_called()

    def test_module_import_needs_no_inference_or_accelerator_packages(self):
        code = """import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'qwen_tts', 'triton'}:
        raise ImportError('runtime unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import qwen3_tts_engine.request_views
"""
        process = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
