"""Experimental BSND rotary preserving separate BF16 multiply/add rounding."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotary(X, C, S, Y, ROWS: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    heads = tl.arange(0, H)[:, None]
    dims = tl.arange(0, D // 2)[None, :]
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        base = (row * H + heads) * D + dims
        first = tl.load(X + base).to(tl.float32)
        second = tl.load(X + base + D // 2).to(tl.float32)
        cos1 = tl.load(C + row * D + dims).to(tl.float32)
        cos2 = tl.load(C + row * D + dims + D // 2).to(tl.float32)
        sin1 = tl.load(S + row * D + dims).to(tl.float32)
        sin2 = tl.load(S + row * D + dims + D // 2).to(tl.float32)
        left = (first * cos1).to(tl.bfloat16).to(tl.float32)
        right = (-second * sin1).to(tl.bfloat16).to(tl.float32)
        out1 = (left + right).to(tl.bfloat16)
        left2 = (second * cos2).to(tl.bfloat16).to(tl.float32)
        right2 = (first * sin2).to(tl.bfloat16).to(tl.float32)
        out2 = (left2 + right2).to(tl.bfloat16)
        tl.store(Y + base, out1)
        tl.store(Y + base + D // 2, out2)


def rotary(hidden, cos, sin):
    if hidden.ndim != 4 or hidden.device.type != 'npu' or hidden.dtype != torch.bfloat16:
        raise ValueError('Expected BF16 BSND NPU input')
    batch, length, heads, width = hidden.shape
    if heads not in (8, 16) or width not in (64, 128) or batch * length < 1:
        raise ValueError('Unsupported rotary shape')
    if cos.shape != (batch, length, width) or sin.shape != cos.shape:
        raise ValueError('Rotary coefficients must match batch, length and width')
    if any(t.device != hidden.device or t.dtype != hidden.dtype or not t.is_contiguous()
           for t in (hidden, cos, sin)):
        raise ValueError('Rotary tensors must be contiguous with matching dtype/device')
    output = torch.empty_like(hidden)
    cores = torch.npu.get_device_properties(hidden.device).vector_core_num
    _rotary[(min(cores, batch * length),)](hidden, cos, sin, output, batch * length, heads, width,
                                          enable_fp_fusion=False)
    return output
