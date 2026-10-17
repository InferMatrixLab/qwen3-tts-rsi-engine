"""Measure ND versus FRACTAL_NZ linear weights on fixed TTS projection shapes."""
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
torch.npu.config.allow_internal_format = True
result = {'status': 'INCONCLUSIVE', 'scope': 'isolated linear graph layout diagnostic, not service performance', 'cases': []}
try:
    with torch.inference_mode():
        for inputs, outputs in [(2048, 4096), (2048, 12288), (6144, 2048), (1024, 6144)]:
            weight = torch.randn(outputs, inputs, device='npu:0', dtype=torch.bfloat16) * 0.02
            weights = {'ND': weight, 'NZ': torch_npu.npu_format_cast(weight, 29)}
            for batch in [1, 16, 64]:
                value = torch.randn(batch, 1, inputs, device='npu:0', dtype=torch.bfloat16)
                captures = {}
                for name, matrix in weights.items():
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3):
                            torch.nn.functional.linear(value, matrix)
                    torch.npu.current_stream().wait_stream(stream)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        output = torch.nn.functional.linear(value, matrix)
                    captures[name] = (graph, output)
                for graph, _ in captures.values():
                    graph.replay()
                error = (captures['ND'][1].float() - captures['NZ'][1].float()).abs().max().item()
                samples = []
                for name in ['ND', 'NZ', 'NZ', 'ND']:
                    graph = captures[name][0]
                    for _ in range(10):
                        graph.replay()
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(100):
                        graph.replay()
                    torch.npu.synchronize()
                    samples.append({'layout': name, 'mean_replay_ms': (time.perf_counter() - start) * 10})
                row = {'batch': batch, 'input_width': inputs, 'output_width': outputs,
                       'max_abs_error': error, 'exact': error == 0, 'samples': samples}
                result['cases'].append(row)
                print(row, flush=True)
                for graph, _ in captures.values():
                    graph.reset()
        result['status'] = 'DIAGNOSTIC_ONLY'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
