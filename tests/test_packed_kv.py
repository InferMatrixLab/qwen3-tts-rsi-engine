import unittest

import torch

from qwen3_tts_engine.packed_kv import copy_packed_prefix


class PackedPrefixTests(unittest.TestCase):
    def tensors(self, dtype=torch.float32):
        target = torch.full((4, 5, 2, 7, 3), -7, dtype=dtype)
        source = torch.arange(4 * 3 * 2 * 5 * 6).view(4, 3, 2, 5, 6).to(dtype)[..., ::2]
        return target, source

    def test_changed_strided_prefix_matches_native_and_preserves_other_state(self):
        for dtype in (torch.bfloat16, torch.float32, torch.int64):
            target, source = self.tensors(dtype)
            for row, slot, length in ((2, 3, 1), (0, 1, 5), (1, 4, 3)):
                original = source.clone()
                expected = target.clone()
                for layer in range(target.shape[0]):
                    expected[layer, slot, :, :length].copy_(
                        source[layer, row, :, :length]
                    )
                copy_packed_prefix(target, source, slot, row=row, length=length)
                self.assertTrue(torch.equal(target, expected))
                self.assertTrue(torch.equal(source, original))
                source.add_(1)

    def test_default_length_preserves_padded_tail(self):
        target, source = self.tensors()
        expected = target.clone()
        expected[:, 0, :, :5].copy_(source[:, 0])
        copy_packed_prefix(target, source, 0)
        self.assertTrue(torch.equal(target, expected))

    def test_strided_target_preserves_interleaved_storage(self):
        base = torch.full((4, 5, 2, 7, 6), -9.0)
        target = base[..., ::2]
        _, source = self.tensors()
        expected = base.clone()
        expected[..., ::2][:, 2, :, :4].copy_(source[:, 1, :, :4])
        copy_packed_prefix(target, source, 2, row=1, length=4)
        self.assertTrue(torch.equal(base, expected))

    def test_invalid_bounds_cannot_partially_write(self):
        target, source = self.tensors()
        saved = target.clone()
        for slot, row, length in (
            (-1, 0, 1),
            (5, 0, 1),
            (0, 3, 1),
            (0, -1, 1),
            (0, 0, 0),
            (0, 0, 6),
            (True, 0, 1),
            (0, True, 1),
            (0, 0, True),
        ):
            with (
                self.subTest(slot=slot, row=row, length=length),
                self.assertRaises(ValueError),
            ):
                copy_packed_prefix(target, source, slot, row=row, length=length)
            self.assertTrue(torch.equal(target, saved))

    def test_incompatible_or_aliasing_source_is_rejected_before_write(self):
        target, source = self.tensors()
        saved = target.clone()
        for bad in (
            source[:3],
            source[:, :, :1],
            source[..., :2],
            source.double(),
            source[0],
            source[:, :0],
            target,
            torch.empty_like(source, device="meta"),
        ):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                copy_packed_prefix(target, bad, 0, length=1)
            self.assertTrue(torch.equal(target, saved))


if __name__ == "__main__":
    unittest.main()
