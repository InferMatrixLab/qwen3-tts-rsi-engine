"""Experimental vector-only SnakeBeta for contiguous NCT tensors on Ascend.

Preserves explicit BF16 rounding between operations, following the numerical
contract documented by SGLang-Omni utils/snake_beta.py. This implementation uses
cached coefficients and a one-dimensional Ascend grid; no CUDA libdevice.
"""
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _snake(X, A, R, Y, N: tl.constexpr, C: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = (offsets // T) % C
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    alpha = tl.load(A + channel).to(tl.float32)
    inverse = tl.load(R + channel).to(tl.float32)
    phase = (x * alpha).to(tl.bfloat16).to(tl.float32)
    sine = tl.sin(phase).to(tl.bfloat16).to(tl.float32)
    square = (sine * sine).to(tl.bfloat16).to(tl.float32)
    scaled = (inverse * square).to(tl.bfloat16).to(tl.float32)
    result = (x + scaled).to(tl.bfloat16)
    tl.store(Y + offsets, result, offsets < N)


def snake(hidden, alpha, inverse_beta, tiled=False):
    if hidden.device.type != 'npu' or hidden.dtype != torch.bfloat16 or hidden.ndim != 3 or not hidden.is_contiguous():
        raise ValueError('Experimental Snake kernel requires contiguous NCT BF16 NPU input')
    if alpha.numel() != hidden.shape[1] or inverse_beta.numel() != hidden.shape[1]:
        raise ValueError('Snake coefficients must match the channel count')
    block = 1024
    if not 0 < hidden.numel() <= block * 65535:
        raise ValueError('Snake probe shape exceeds the validated grid range')
    output = torch.empty_like(hidden)
    if tiled:
        rows, channels, length = hidden.shape[0] * hidden.shape[1], hidden.shape[1], hidden.shape[2]
        time_block = min(256, max(32, triton.next_power_of_2(length)))
        row_block = 1024 // time_block
        tiles = triton.cdiv(rows, row_block) * triton.cdiv(length, time_block)
        cores = torch.npu.get_device_properties(hidden.device).vector_core_num
        _snake_tiled[(min(cores, tiles),)](hidden, alpha, inverse_beta, output, rows, channels, length,
                                         row_block, time_block, enable_fp_fusion=False)
        return output
    _snake[(triton.cdiv(hidden.numel(), block),)](
        hidden, alpha, inverse_beta, output, hidden.numel(), hidden.shape[1], hidden.shape[2], block,
        enable_fp_fusion=False)
    return output


@triton.jit
def _snake_tiled(X, A, R, Y, ROWS: tl.constexpr, C: tl.constexpr, T: tl.constexpr,
                 RC: tl.constexpr, TC: tl.constexpr):
    time_tiles = tl.cdiv(T, TC)
    tiles = tl.cdiv(ROWS, RC) * time_tiles
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        row = (tile // time_tiles) * RC + tl.arange(0, RC)
        time_index = (tile % time_tiles) * TC + tl.arange(0, TC)
        alpha = tl.load(A + row % C, row < ROWS, other=0).to(tl.float32)
        inverse = tl.load(R + row % C, row < ROWS, other=0).to(tl.float32)
        offsets = row[:, None] * T + time_index[None, :]
        mask = (row[:, None] < ROWS) & (time_index[None, :] < T)
        x = tl.load(X + offsets, mask, other=0).to(tl.float32)
        phase = (x * alpha[:, None]).to(tl.bfloat16).to(tl.float32)
        sine = tl.sin(phase).to(tl.bfloat16).to(tl.float32)
        square = (sine * sine).to(tl.bfloat16).to(tl.float32)
        scaled = (inverse[:, None] * square).to(tl.bfloat16).to(tl.float32)
        result = (x + scaled).to(tl.bfloat16)
        tl.store(Y + offsets, result, mask)
