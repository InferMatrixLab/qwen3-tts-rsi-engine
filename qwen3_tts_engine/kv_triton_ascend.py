"""Experimental paired KV write for one-token BNSD decode on Ascend."""
import torch
import triton
import triton.language as tl


@triton.jit
def _write(K, V, TK, TV, P, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
           S: tl.constexpr, KB: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
           VB: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < B * H * D
    row = x // (H * D)
    head = (x // D) % H
    dim = x % D
    pos = tl.load(P + row, valid, other=0)
    k = tl.load(K + row * KB + head * KH + dim * KD, valid, other=0)
    v = tl.load(V + row * VB + head * VH + dim * VD, valid, other=0)
    target = ((row * H + head) * S + pos) * D + dim
    tl.store(TK + target, k, valid)
    tl.store(TV + target, v, valid)


@triton.jit
def _write_row(K, V, TK, TV, P, H: tl.constexpr, D: tl.constexpr,
               S: tl.constexpr, KB: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
               VB: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr):
    row = tl.program_id(0)
    head = tl.arange(0, H)[:, None]
    dim = tl.arange(0, D)[None, :]
    pos = tl.load(P + row)
    k = tl.load(K + row * KB + head * KH + dim * KD)
    v = tl.load(V + row * VB + head * VH + dim * VD)
    target = ((row * H + head) * S + pos) * D + dim
    tl.store(TK + target, k)
    tl.store(TV + target, v)


def write(keys, values, target_keys, target_values, positions, row_tiled=False):
    """Caller must guarantee valid positions; no host synchronization in graphs."""
    if keys.ndim != 4 or keys.shape[2] != 1 or values.shape != keys.shape:
        raise ValueError('Expected matching single-token BNSD K/V')
    batch, heads, _, width = keys.shape
    if (target_keys.shape != target_values.shape or target_keys.shape[:2] != (batch, heads)
            or target_keys.shape[-1] != width or not target_keys.is_contiguous()
            or not target_values.is_contiguous() or positions.shape != (batch,)
            or not positions.is_contiguous() or positions.dtype != torch.int64):
        raise ValueError('Invalid contiguous KV arena or position vector')
    tensors = (keys, values, target_keys, target_values)
    if any(t.device != keys.device or t.dtype != keys.dtype for t in tensors) or positions.device != keys.device:
        raise ValueError('KV tensors must share dtype and device')
    if keys.device.type != 'npu' or keys.dtype != torch.bfloat16:
        raise ValueError('Paired KV write supports BF16 NPU tensors only')
    block = 1024
    if row_tiled:
        _write_row[(batch,)](
            keys, values, target_keys, target_values, positions, heads, width,
            target_keys.shape[2], keys.stride(0), keys.stride(1), keys.stride(3),
            values.stride(0), values.stride(1), values.stride(3))
        return
    _write[(triton.cdiv(batch * heads * width, block),)](
        keys, values, target_keys, target_values, positions, batch, heads, width,
        target_keys.shape[2], keys.stride(0), keys.stride(1), keys.stride(3),
        values.stride(0), values.stride(1), values.stride(3), block)
