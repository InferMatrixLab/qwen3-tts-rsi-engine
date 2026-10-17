"""Validate Ascend fused Snake arithmetic and NPUGraph replay; no service claim."""
import argparse
import json
import time
import traceback
from pathlib import Path
import torch
import torch_npu
from qwen3_tts_engine.snake_triton_ascend import snake

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tiled', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
result = {'status': 'DIAGNOSTIC_ONLY', 'cases': [], 'tiled': a.tiled}
graphs = {}
try:
    torch.npu.set_device('npu:0')
    with torch.inference_mode():
        g = torch.Generator().manual_seed(42)
        for shape in [(1, 7, 37), (4, 1536, 32), (4, 768, 256), (4, 192, 5120), (4, 96, 15360)]:
            for scale in [1., 16.]:
                x = (torch.randn(shape, generator=g) * scale).bfloat16().to('npu:0')
                alpha = torch.randn(1, shape[1], 1, generator=g).bfloat16().to('npu:0').exp()
                inverse = 1.0 / (torch.randn(1, shape[1], 1, generator=g).bfloat16().to('npu:0').exp() + 1e-9)
                def reference(): return x + inverse * torch.pow(torch.sin(x * alpha), 2)
                for name, operation in [('reference', reference), ('triton', lambda: snake(x, alpha, inverse, tiled=a.tiled))]:
                    operation()
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
                expected, actual = graphs['reference'][1], graphs['triton'][1]
                error = (expected.float() - actual.float()).abs()
                row = {'shape': shape, 'scale': scale, 'max_abs': error.max().item(),
                       'rms': error.square().mean().sqrt().item(), 'exact': torch.equal(expected, actual),
                       'finite': bool(torch.isfinite(actual).all()), 'timings': []}
                if not row['finite']: raise ValueError('Nonfinite fused Snake output')
                for name in ['reference', 'triton', 'triton', 'reference']:
                    graph = graphs[name][0]
                    for _ in range(10): graph.replay()
                    torch.npu.synchronize(); start = time.perf_counter()
                    for _ in range(100): graph.replay()
                    torch.npu.synchronize()
                    row['timings'].append({'arm': name, 'mean_ms': (time.perf_counter() - start) * 10})
                result['cases'].append(row)
                print(row, flush=True)
                for graph, _ in graphs.values(): graph.reset()
                graphs.clear()
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    (a.output / 'error.txt').write_text(traceback.format_exc())
    raise
finally:
    for graph, _ in graphs.values(): graph.reset()
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
