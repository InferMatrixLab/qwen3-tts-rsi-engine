"""Exact cache/replay checks and ABBA timing for paired Ascend KV writes."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from qwen3_tts_engine.kv_triton_ascend import write


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--row-tiled', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
torch.manual_seed(42)
result = {'status': 'INCONCLUSIVE', 'scope': 'Synthetic cache operator, not model or service validation',
          'row_tiled': args.row_tiled, 'timing_caveat': 'Native arm includes index preparation, amortized across layers in serving.', 'cases': []}
graphs = []
try:
    with torch.inference_mode():
        for batch in (1, 8, 32, 64):
            for capacity in (64, 512):
                # Packed projection views have a larger row stride than plain KV.
                packed = torch.randn(batch, 1, 32 * 128, device='npu:0', dtype=torch.bfloat16)
                k = packed[..., 16 * 128:24 * 128].view(batch, 1, 8, 128).transpose(1, 2)
                v = packed[..., 24 * 128:].view(batch, 1, 8, 128).transpose(1, 2)
                baseline = [torch.randn(batch, 8, capacity, 128, device='npu:0', dtype=k.dtype) for _ in range(2)]
                candidate = [t.clone() for t in baseline]
                positions = torch.arange(batch, device='npu:0') % capacity
                rows = torch.arange(batch, device='npu:0')[:, None].expand(batch, 8)
                heads = torch.arange(8, device='npu:0')[None].expand(batch, 8)
                def operation(fused):
                    if fused:
                        write(k, v, *candidate, positions, row_tiled=args.row_tiled)
                    else:
                        indices = torch.stack((rows, heads, positions[:, None].expand_as(rows)), -1).reshape(-1, 3)
                        torch_npu.npu_scatter_nd_update_(baseline[0], indices, k.reshape(-1, 128))
                        torch_npu.npu_scatter_nd_update_(baseline[1], indices, v.reshape(-1, 128))
                for fused in (False, True):
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3): operation(fused)
                    torch.npu.current_stream().wait_stream(stream)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream): operation(fused)
                    graphs.append(graph)
                for step in range(4):
                    packed.copy_(torch.randn_like(packed))
                    positions.copy_((torch.arange(batch, device='npu:0') + capacity - 1 + step) % capacity)
                    for graph in graphs: graph.replay()
                    for a, b in zip(baseline, candidate):
                        assert torch.equal(a, b), 'Cache changed outside target positions or replay used stale inputs'
                row = {'batch': batch, 'capacity': capacity, 'full_arena_exact': True, 'measurements': []}
                for arm in (0, 1, 1, 0):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(200): graphs[arm].replay()
                    torch.npu.synchronize()
                    row['measurements'].append({'arm': 'paired' if arm else 'native', 'mean_ms': (time.perf_counter() - start) * 5})
                result['cases'].append(row)
                for graph in graphs: graph.reset()
                graphs.clear()
        result['status'] = 'PASS'
except Exception as error:
    result.update(status='FAIL', error=repr(error))
    raise
finally:
    for graph in graphs: graph.reset()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
