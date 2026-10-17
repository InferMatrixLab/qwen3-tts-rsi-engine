"""Experimental inverse-CDF tail; input CDF retains the reference accumulation."""
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _select(CDF, U, OUT, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        values = tl.load(CDF + row * WIDTH + offsets, offsets < WIDTH, other=0)
        total = tl.load(CDF + row * WIDTH + WIDTH - 1)
        uniform = tl.load(U + row)
        normalized = tl.div_rn(values, total)
        token = tl.sum(((normalized <= uniform) & (offsets < WIDTH)).to(tl.int32), 0)
        tl.store(OUT + row, token.to(tl.int64))


def select(cdf, uniforms):
    if (cdf.device.type != 'npu' or cdf.dtype != torch.float32 or cdf.ndim != 2
            or not cdf.is_contiguous() or not 1 <= cdf.shape[1] <= 4096):
        raise ValueError('CDF must be contiguous FP32 NPU rows of width 1..4096')
    if (uniforms.device != cdf.device or uniforms.dtype != torch.float32
            or uniforms.shape != (cdf.shape[0],) or not uniforms.is_contiguous()):
        raise ValueError('Uniforms must be one contiguous FP32 value per CDF row')
    if cdf.shape[0] < 1:
        raise ValueError('Empty CDF batch')
    # The caller supplies a finite monotone CDF with positive total and uniforms
    # in [0, 1); do not add host synchronization to the graph-captured path.
    output = torch.empty((cdf.shape[0], 1), device=cdf.device, dtype=torch.long)
    cores = torch.npu.get_device_properties(cdf.device).vector_core_num
    _select[(min(cores, cdf.shape[0]),)](
        cdf, uniforms, output, cdf.shape[0], cdf.shape[1],
        triton.next_power_of_2(cdf.shape[1]), enable_fp_fusion=False)
    return output
