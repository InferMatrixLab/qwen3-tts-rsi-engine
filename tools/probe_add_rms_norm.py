"""Measure residual-add RMSNorm fusion against rounded BF16 and CPU FP32 references."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
result = {'status': 'DIAGNOSTIC_ONLY', 'scope': 'Operator probe only, not speech-quality acceptance', 'cases': []}
torch.npu.set_device('npu:0')
graphs = {}
try:
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(42)
        for width in (1024, 2048):
            for batch, length in ((1, 1), (8, 2), (32, 1), (64, 1), (4, 64)):
                cpu_x = torch.randn(batch, length, width, generator=generator).bfloat16()
                cpu_r = torch.randn(batch, length, width, generator=generator).bfloat16()
                cpu_w = (torch.randn(width, generator=generator) * .1 + 1).bfloat16()
                x, residual, weight = [t.to('npu:0') for t in (cpu_x, cpu_r, cpu_w)]
                def separate():
                    total = x + residual
                    return torch_npu.npu_rms_norm(total, weight, epsilon=1e-6)[0], total
                def fused():
                    output, _, total = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon=1e-6)
                    return output, total
                for name, operation in [('separate', separate), ('fused', fused)]:
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3): operation()
                    torch.npu.current_stream().wait_stream(stream)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream): output = operation()
                    graphs[name] = graph, output
                for graph, _ in graphs.values(): graph.replay()
                torch.npu.synchronize()
                ref, actual = [graphs[k][1] for k in ('separate', 'fused')]
                total = cpu_x.float() + cpu_r.float()
                golden = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) * cpu_w.float()
                row = {'width': width, 'batch': batch, 'length': length,
                       'output_max_abs': (ref[0].float() - actual[0].float()).abs().max().item(),
                       'residual_equal': torch.equal(ref[1], actual[1]),
                       'cpu_fp32_max_abs': (golden - actual[0].float().cpu()).abs().max().item(),
                       'finite': bool(torch.isfinite(actual[0]).all()), 'samples': []}
                if not row['finite'] or not row['residual_equal']: raise AssertionError(row)
                for name in ('separate', 'fused', 'fused', 'separate'):
                    graph = graphs[name][0]
                    for _ in range(10): graph.replay()
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(1000): graph.replay()
                    torch.npu.synchronize()
                    row['samples'].append({'arm': name, 'mean_ms': (time.perf_counter() - start)})
                result['cases'].append(row)
                print(row, flush=True)
                for graph, _ in graphs.values(): graph.reset()
                graphs.clear()
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    for graph, _ in graphs.values(): graph.reset()
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
