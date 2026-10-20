"""Constructor/startup regressions; no accelerator allocations or model weights."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


class PredictorStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qwen3_tts_engine.continuous import UniformPredictor
        cls.predictor_type = UniformPredictor

    def setUp(self):
        self.rotary = torch.nn.Module().eval()
        self.rotary.rope_type = "default"
        self.reference = SimpleNamespace(model=SimpleNamespace(rotary_emb=self.rotary))

    def test_opted_out_does_not_inspect_model(self):
        class Wrapped:
            @property
            def model(self):
                raise AssertionError("Disabled option must not inspect wrappers")

        predictor = self.predictor_type(Wrapped())
        self.assertFalse(predictor.fixed_rope)
        self.assertEqual(predictor.captures, {})

    def test_valid_constructor_does_not_compute_or_allocate(self):
        # A bare module has no forward implementation: even a CPU coefficient
        # computation would fail. Construction must inspect only metadata.
        stream = Mock(side_effect=AssertionError("No capture at startup"))
        with patch.object(torch, "npu", SimpleNamespace(Stream=stream), create=True):
            predictor = self.predictor_type(self.reference, fixed_rope=True)
        stream.assert_not_called()
        self.assertEqual(predictor.captures, {})
        self.assertIs(self.reference.model.rotary_emb, self.rotary)

    def test_missing_structure_rejected_in_constructor(self):
        for reference in (object(), SimpleNamespace(model=None),
                          SimpleNamespace(model=SimpleNamespace()),
                          SimpleNamespace(model=SimpleNamespace(rotary_emb=None)),
                          SimpleNamespace(model=SimpleNamespace(rotary_emb=object()))):
            with self.subTest(reference_type=type(reference).__name__):
                with self.assertRaisesRegex(ValueError, "reference.model.rotary_emb"):
                    self.predictor_type(reference, fixed_rope=True)

    def test_training_and_non_default_rejected_in_constructor(self):
        for training, rope_type in ((True, "default"), (False, "dynamic"), (False, "linear")):
            self.rotary.train(training)
            self.rotary.rope_type = rope_type
            with self.subTest(training=training, rope_type=rope_type):
                with self.assertRaisesRegex(ValueError, "evaluation with default"):
                    self.predictor_type(self.reference, fixed_rope=True)

    def test_capture_rechecks_after_constructor(self):
        predictor = self.predictor_type(self.reference, fixed_rope=True)
        self.rotary.rope_type = "dynamic"
        stream = Mock(side_effect=AssertionError("Guard must precede capture"))
        with patch.object(torch, "npu", SimpleNamespace(Stream=stream), create=True):
            with self.assertRaisesRegex(ValueError, "evaluation with default"):
                predictor.run(torch.zeros(1, 2, 8), torch.zeros(1, 16))
        stream.assert_not_called()
        self.assertIs(self.reference.model.rotary_emb, self.rotary)

    def test_service_start_fails_before_ready_or_worker(self):
        from qwen3_tts_engine.server import SpeechService

        self.rotary.rope_type = "dynamic"

        async def exercise():
            options = SimpleNamespace(queue_size=1, model=Path("unused"), device="unused")
            service = SpeechService(options, lambda *_: self.predictor_type(self.reference, fixed_rope=True))
            try:
                with self.assertRaisesRegex(ValueError, "evaluation with default"):
                    await service.start(None)
                self.assertIsNone(service.model)
                self.assertIsNone(service.worker)
            finally:
                await service.stop(None)

        asyncio.run(exercise())
