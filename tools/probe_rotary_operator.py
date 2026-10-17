"""Compare native rotary graph kernels; numerical diagnostics are not speech quality."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
from qwen_tts.core.models.modeling_qwen3_tts import apply_rotary_pos_emb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--layout', choices=['bnsd', 'bsnd'], default='bnsd')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
result = {'status': 'INCONCLUSIVE', 'scope': 'isolated rotary graph diagnostics; no service or quality claim', 'layout': args.layout, 'cases': []}
captures = {}
try:
    with torch.inference_mode():
        for batch in [1, 17, 64]:
            for length in [1, 2, 16, 64]:
                generator = torch.Generator().manual_seed(42)
                query = torch.randn(batch, 16, length, 128, generator=generator, dtype=torch.bfloat16).to('npu:0')
                key = torch.randn(batch, 8, length, 128, generator=generator, dtype=torch.bfloat16).to('npu:0')
                if args.layout == 'bsnd':
                    query = query.transpose(1, 2).contiguous()
                    key = key.transpose(1, 2).contiguous()
                phase = torch.randn(batch, length, 64, generator=generator).repeat(1, 1, 2).to('npu:0')
                cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()
                r1, r2 = cos.unsqueeze(1), sin.unsqueeze(1)
                operations = {'reference': lambda: apply_rotary_pos_emb(query, key, cos, sin),
                              'native': lambda: (torch_npu.npu_rotary_mul(query, r1, r2),
                                                 torch_npu.npu_rotary_mul(key, r1, r2))}
                if args.layout == 'bsnd':
                    r1, r2 = cos.unsqueeze(2), sin.unsqueeze(2)
                    operations = {'reference': lambda: apply_rotary_pos_emb(query.transpose(1, 2), key.transpose(1, 2), cos, sin),
                                  'native': lambda: (torch_npu.npu_rotary_mul(query, r1, r2).transpose(1, 2),
                                                     torch_npu.npu_rotary_mul(key, r1, r2).transpose(1, 2))}
                for name, operation in operations.items():
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
                    captures[name] = (graph, output)
                for graph, _ in captures.values(): graph.replay()
                torch.npu.synchronize()
                errors = [(a.float() - b.float()).abs() for a, b in zip(captures['reference'][1], captures['native'][1])]
                assert all(bool(torch.isfinite(value).all()) for value in captures['native'][1])
                row = {'batch': batch, 'length': length, 'max_abs_error': max(e.max().item() for e in errors), 'samples': []}
                for name in ['reference', 'native', 'native', 'reference']:
                    graph = captures[name][0]
                    for _ in range(10): graph.replay()
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(100): graph.replay()
                    torch.npu.synchronize()
                    row['samples'].append({'arm': name, 'mean_ms': (time.perf_counter() - start) * 10})
                result['cases'].append(row)
                print(row, flush=True)
                for graph, _ in captures.values(): graph.reset()
                captures.clear()
        result['status'] = 'DIAGNOSTIC_ONLY'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    for graph, _ in captures.values(): graph.reset()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
