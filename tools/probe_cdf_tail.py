"""Check fused CDF selection including exact boundaries, then time graph replays."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch_npu
from qwen3_tts_engine.cdf_triton_ascend import select

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
torch.manual_seed(42)
result = {'status': 'INCONCLUSIVE', 'scope': 'Synthetic operator semantics and timings, not speech quality or service performance', 'cases': []}
graphs = []
try:
    with torch.inference_mode():
        for batch in [1, 17, 32, 64]:
            for width in [2048, 3072]:
                cdf = torch.ones(batch, width, device='npu:0').cumsum(-1)
                uniforms = torch.full((batch,), .5, device='npu:0')
                def reference():
                    normalized = cdf / cdf[:, -1:]
                    return (normalized <= uniforms[:, None]).sum(-1, keepdim=True)
                outputs = []
                for operation in [reference, lambda: select(cdf, uniforms)]:
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3):
                            operation()
                    torch.npu.current_stream().wait_stream(stream)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        output = operation()
                    graphs.append(graph)
                    outputs.append(output)
                for replay in range(12):
                    scores = torch.randn(batch, width, device='npu:0')
                    cutoff = scores.topk(50, dim=-1).values[:, -1:]
                    cdf.copy_(torch.softmax(scores.masked_fill(scores < cutoff, -float('inf')), -1).cumsum(-1))
                    if replay < 4:
                        uniforms.copy_(torch.rand(batch, device='npu:0'))
                    else:
                        # Include exact CDF plateaus and one-ULP neighbors. CPU
                        # nextafter avoids relying on an NPU nextafter operator.
                        normalized = (cdf / cdf[:, -1:]).cpu()
                        boundary = normalized[:, (width // 3 + replay * 17) % width]
                        if replay % 3:
                            target = torch.full_like(boundary, float('inf') if replay % 3 == 1 else -float('inf'))
                            boundary = torch.nextafter(boundary, target)
                        uniforms.copy_(boundary.clamp(0, 1 - torch.finfo(torch.float32).eps))
                    for graph in graphs:
                        graph.replay()
                    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
                timings = []
                for arm in [0, 1, 1, 0]:
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(200):
                        graphs[arm].replay()
                    torch.npu.synchronize()
                    timings.append({'arm': 'fused' if arm else 'reference', 'mean_ms': (time.perf_counter() - start) * 5})
                result['cases'].append({'batch': batch, 'width': width, 'exact_replays': 12, 'measurements': timings})
                for graph in graphs:
                    graph.reset()
                graphs = []
        result['status'] = 'DIAGNOSTIC_ONLY'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = repr(error)
    raise
finally:
    for graph in graphs:
        graph.reset()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
