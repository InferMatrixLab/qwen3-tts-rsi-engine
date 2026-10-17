"""Compare graph-captured KV writes; timings are operator diagnostics only."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch_npu

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
result = {'status': 'INCONCLUSIVE', 'scope': 'KV write correctness and isolated graph replay, not service performance', 'cases': []}
try:
    with torch.inference_mode():
        for batch in [1, 4, 32, 64]:
            heads, capacity, width = 8, 512, 128
            positions = torch.zeros(batch, device='npu:0', dtype=torch.long)
            rows = torch.arange(batch, device='npu:0')[:, None].expand(batch, heads)
            columns = torch.arange(heads, device='npu:0')[None, :].expand(batch, heads)
            updates = torch.randn(batch, heads, 1, width, device='npu:0', dtype=torch.bfloat16)
            buffers = {name: torch.zeros(batch, heads, capacity, width, device='npu:0', dtype=updates.dtype)
                       for name in ['torch', 'native']}
            graphs = {}
            for name in buffers:
                target = buffers[name]
                def compute():
                    if name == 'torch':
                        target.scatter_(2, positions.view(batch, 1, 1, 1).expand(batch, heads, 1, width), updates)
                    else:
                        indices = torch.stack((rows, columns, positions[:, None].expand(batch, heads)), -1).reshape(-1, 3)
                        torch_npu.npu_scatter_nd_update_(target, indices, updates.reshape(-1, width))
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(3):
                        compute()
                torch.npu.current_stream().wait_stream(stream)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    compute()
                graphs[name] = graph
            for buffer in buffers.values():
                buffer.zero_()
            for step in [0, 1, 127, 511]:
                positions.copy_((torch.arange(batch, device='npu:0') + step) % capacity)
                updates.copy_(torch.randn_like(updates))
                for graph in graphs.values():
                    graph.replay()
                torch.testing.assert_close(buffers['native'], buffers['torch'], atol=0, rtol=0)
            samples = []
            for name in ['torch', 'native', 'native', 'torch']:
                for _ in range(10):
                    graphs[name].replay()
                torch.npu.synchronize()
                started = time.perf_counter()
                for _ in range(100):
                    graphs[name].replay()
                torch.npu.synchronize()
                samples.append({'backend': name, 'mean_replay_ms': (time.perf_counter() - started) * 10})
            result['cases'].append({'batch': batch, 'capacity': capacity, 'exact': True, 'samples': samples})
            print(result['cases'][-1], flush=True)
            for graph in graphs.values():
                graph.reset()
        result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
