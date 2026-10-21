"""Exact native copies of one request prefix across packed KV layer buffers."""

import torch


def copy_packed_prefix(target, source, slot, row=0, length=None):
    """Preserve every other slot and each selected row's padded tail."""
    if (
        not isinstance(target, torch.Tensor)
        or not isinstance(source, torch.Tensor)
        or target.ndim != 5
        or source.ndim != 5
        or any(size <= 0 for size in target.shape + source.shape)
        or (target.shape[0], target.shape[2], target.shape[4])
        != (source.shape[0], source.shape[2], source.shape[4])
        or target.device != source.device
        or target.dtype != source.dtype
        or target.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    ):
        raise ValueError("Expected independent compatible packed KV tensors")
    length = source.shape[3] if length is None else length
    if (
        type(slot) is not int
        or not 0 <= slot < target.shape[1]
        or type(row) is not int
        or not 0 <= row < source.shape[1]
        or type(length) is not int
        or not 0 < length <= min(target.shape[3], source.shape[3])
    ):
        raise ValueError("Expected a valid packed KV row and prefix length")
    target[:, slot, :, :length].copy_(source[:, row, :, :length])
