"""Pure selected-row stores for disjoint concurrent Codec graph state updates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter(
    SOURCE,
    INDICES,
    TARGET,
    N: tl.constexpr,
    ROW: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    SEGMENT: tl.constexpr,
    ELEMENT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles_per_segment = tl.cdiv(SEGMENT, BLOCK)
    segments_per_row = ROW // SEGMENT
    for tile in range(
        tl.program_id(0), (N // SEGMENT) * tiles_per_segment, tl.num_programs(0)
    ):
        subrow = tile // tiles_per_segment
        row = subrow // segments_per_row
        start = (subrow % segments_per_row) * SEGMENT
        inner = (tile % tiles_per_segment) * BLOCK + tl.arange(0, BLOCK)
        source_base = (
            row * S0
            + (start // (D2 * D3)) * S1
            + (start // D3 % D2) * S2
            + (start % D3) * S3
        )
        destination = tl.load(INDICES + row)
        value = tl.load(
            SOURCE + source_base + inner * ELEMENT_STRIDE, inner < SEGMENT, other=0
        )
        tl.store(TARGET + destination * ROW + start + inner, value, inner < SEGMENT)


def scatter_rows(target, indices, source):
    """Copy rows only; caller validates unique, in-range indices before capture."""
    if (
        target.device.type != "npu"
        or source.device != target.device
        or indices.device != target.device
        or indices.dtype != torch.long
        or indices.ndim != 1
        or not indices.is_contiguous()
        or not target.is_contiguous()
        or not 1 <= target.ndim <= 4
        or source.ndim != target.ndim
        or source.dtype != target.dtype
        or source.shape[1:] != target.shape[1:]
        or source.shape[0] != indices.numel()
        or source.numel() == 0
        or source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()
    ):
        raise ValueError(
            "Expected independent source and contiguous selected-row target"
        )
    shape = tuple(source.shape) + (1,) * (4 - source.ndim)
    strides = tuple(source.stride()) + (0,) * (4 - source.ndim)
    row = source.numel() // source.shape[0]
    segment = 1
    element_stride = source.stride(-1)
    for dimension in range(source.ndim - 1, 0, -1):
        if source.shape[dimension] == 1:
            continue
        if source.stride(dimension) != segment * element_stride:
            break
        segment *= source.shape[dimension]
    block = min(1024, max(32, triton.next_power_of_2(segment)))
    cores = torch.npu.get_device_properties(target.device).vector_core_num
    _scatter[(min(cores, source.numel() // segment * triton.cdiv(segment, block)),)](
        source,
        indices,
        target,
        source.numel(),
        row,
        *shape[1:],
        *strides,
        segment,
        element_stride,
        block,
    )
