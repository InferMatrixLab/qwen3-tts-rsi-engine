import unittest

import torch

from qwen3_tts_engine.fixed_predictor_rotary import FixedPredictorRotary


class Rotary(torch.nn.Module):
    rope_type = "default"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, positions):
        self.calls += 1
        angles = positions.float()[..., None] * torch.tensor([0.2, 0.4])
        return angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)


class FixedPredictorRotaryTests(unittest.TestCase):
    def setUp(self):
        self.original = Rotary().eval()
        self.hidden = torch.zeros(4, 2, 8, dtype=torch.bfloat16)
        self.positions = [torch.tensor([[0, 1]]), torch.tensor([[2]])]
        self.fixed = FixedPredictorRotary(self.original, self.hidden, self.positions)

    def test_coefficients_are_exact_and_reused_for_changing_hidden(self):
        for position in self.positions:
            expected = self.original(self.hidden, position)
            actual = self.fixed(self.hidden + 1, position)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
            self.assertIs(actual, self.fixed(self.hidden * 3, position))
        calls = self.original.calls
        self.fixed(self.hidden, self.positions[0])
        self.assertEqual(calls, self.original.calls)

    def test_rejects_equal_but_unowned_position(self):
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.fixed(self.hidden, self.positions[0].clone())

    def test_rejects_changed_dtype_and_device(self):
        for hidden in (self.hidden.float(), torch.empty(4, 2, 8, device="meta", dtype=self.hidden.dtype)):
            with self.subTest(device=hidden.device, dtype=hidden.dtype):
                with self.assertRaisesRegex(ValueError, "dtype/device"):
                    self.fixed(hidden, self.positions[0])

    def test_restores_original_after_success_and_exception(self):
        model = torch.nn.Module()
        model.rotary_emb = self.original
        with self.fixed.installed(model):
            self.assertIs(model.rotary_emb, self.fixed)
        self.assertIs(model.rotary_emb, self.original)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with self.fixed.installed(model):
                raise RuntimeError("synthetic capture failure")
        self.assertIs(model.rotary_emb, self.original)

    def test_rejects_training_or_non_default_rope(self):
        for training, rope_type in ((True, "default"), (False, "dynamic"), (False, "linear")):
            self.original.train(training)
            self.original.rope_type = rope_type
            with self.subTest(training=training, rope_type=rope_type):
                with self.assertRaisesRegex(ValueError, "evaluation with default"):
                    FixedPredictorRotary(self.original, self.hidden, self.positions)

    def test_independent_capture_lifetimes(self):
        other_positions = [position.clone() for position in self.positions]
        other = FixedPredictorRotary(self.original, self.hidden, other_positions)
        self.fixed.entries.clear()
        self.assertEqual(len(other(self.hidden, other_positions[0])), 2)


if __name__ == "__main__":
    unittest.main()
