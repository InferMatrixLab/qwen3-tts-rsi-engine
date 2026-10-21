"""Reject conflicting Codec work before graph capture or device submission."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from qwen3_tts_engine.codec_slots import CodecSlots


class CodecGroupValidationTests(unittest.TestCase):
    def setUp(self):
        self.codec = object.__new__(CodecSlots)
        self.codec.slots = 8
        self.codec.decoder = SimpleNamespace(config=SimpleNamespace(num_quantizers=16))
        self.codec.max_chunk_frames = 8
        self.codec.frames = torch.zeros(8, dtype=torch.long)
        self.codec.stream_count = 2
        self.codec._graph = Mock(
            side_effect=AssertionError("must not capture invalid work")
        )

    def codes(self, batch=2, length=2):
        return torch.zeros(batch, 16, length, dtype=torch.long)

    def test_range_reset_matches_per_row_reference_and_preserves_other_slots(self):
        self.codec.buffers = [
            torch.arange(64).view(8, 2, 4).to(torch.bfloat16) + 1,
            torch.arange(8, dtype=torch.long) + 1,
        ]
        expected = [buffer.clone() for buffer in self.codec.buffers]
        for buffer in expected:
            for slot in (2, 3, 4):
                buffer[slot].zero_()
        self.codec.reset_range(2, 3)
        self.assertTrue(
            all(torch.equal(x, y) for x, y in zip(expected, self.codec.buffers))
        )

    def test_invalid_or_empty_range_cannot_mutate_any_buffer(self):
        self.codec.buffers = [torch.ones(8, 3, 5), torch.ones(8, dtype=torch.long)]
        saved = [buffer.clone() for buffer in self.codec.buffers]
        self.codec.reset_range(8, 0)
        for start, count in ((-1, 2), (7, 2), (0, -1), (9, 0), (True, 1), (0, False)):
            with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                self.codec.reset_range(start, count)
            self.assertTrue(
                all(torch.equal(x, y) for x, y in zip(saved, self.codec.buffers))
            )

    def test_cross_group_overlap_rejected_before_capture(self):
        callback = Mock()
        with self.assertRaises(ValueError):
            self.codec.decode_groups(
                [([0, 1], self.codes()), ([2, 0], self.codes())], on_output=callback
            )
        self.codec._graph.assert_not_called()
        callback.assert_not_called()

    def test_serial_callback_preserves_group_order_and_owned_outputs(self):
        self.codec.stream_count = 1
        outputs = [torch.tensor([17]), torch.tensor([23])]
        self.codec.decode = Mock(side_effect=outputs)
        callback = Mock()
        actual = self.codec.decode_groups(
            [([0, 1], self.codes()), ([4, 5], self.codes())], on_output=callback
        )
        self.assertEqual(len(actual), 2)
        self.assertIs(actual[0], outputs[0])
        self.assertIs(actual[1], outputs[1])
        self.assertEqual([call.args[0] for call in callback.call_args_list], [0, 1])
        self.assertIs(callback.call_args_list[0].args[1], outputs[0])
        self.assertIs(callback.call_args_list[1].args[1], outputs[1])

    def test_later_invalid_group_cannot_partially_submit_first_group(self):
        for indices in ([], [1, 1], [-1, 2], [2, 8]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.codec.decode_groups(
                    [([0, 3], self.codes()), (indices, self.codes(len(indices)))]
                )
        self.codec._graph.assert_not_called()

    def test_single_group_runs_on_owner_and_propagates_callback_failure(self):
        graph = Mock()
        output = torch.tensor([17])
        graph.run.return_value = output
        self.codec._graph = Mock(return_value=graph)
        codes = self.codes()
        callback = Mock()
        self.assertIs(self.codec.decode_groups([([1, 6], codes)], callback)[0], output)
        self.codec._graph.assert_called_once_with(2, 2)
        graph.run.assert_called_once_with([1, 6], codes)
        callback.assert_called_once_with(0, output)
        callback.side_effect = RuntimeError("consumer stopped")
        with self.assertRaisesRegex(RuntimeError, "consumer stopped"):
            self.codec.decode_groups([([1, 6], codes)], callback)

    def test_lane_cache_keys_can_be_sorted_at_service_shutdown(self):
        del self.codec._graph
        self.codec.graphs = {}
        self.codec.streams = [Mock(), Mock()]
        self.codec.scatter = Mock()
        with patch("qwen3_tts_engine.codec_slots.CodecSlotGraph") as graph:
            for lane in (0, 1):
                self.codec._graph(2, 8, lane)
            self.assertEqual(graph.call_count, 2)
        self.assertEqual(sorted(self.codec.graphs), [(2, 8, 0), (2, 8, 1)])

    def test_invalid_code_shape_or_type_rejected_before_capture(self):
        for codes in (
            self.codes(1),
            self.codes(length=9),
            self.codes().float(),
            torch.zeros(2, 15, 2, dtype=torch.long),
        ):
            with (
                self.subTest(shape=tuple(codes.shape), dtype=codes.dtype),
                self.assertRaises(ValueError),
            ):
                self.codec.decode_groups([([0, 1], codes)])
        self.codec._graph.assert_not_called()

    def test_empty_work_has_no_stream_or_capture_dependency(self):
        for count in (1, 2):
            self.codec.stream_count = count
            self.assertEqual(self.codec.decode_groups([]), [])
        self.codec._graph.assert_not_called()


if __name__ == "__main__":
    unittest.main()
