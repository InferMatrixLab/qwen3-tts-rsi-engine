"""Check exact BF16 Triton rotary; isolated diagnostics are not speech quality."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
from qwen_tts.core.models.modeling_qwen3_tts import apply_rotary_pos_emb
from qwen3_tts_engine.rotary_triton_ascend import rotary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--layout', choices=['bnsd', 'bsnd'], default='bsnd')
args = parser.parse_args()
if args.layout != 'bsnd':
    parser.error('Exact Triton probe requires BSND')
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
                              'triton_exact': lambda: (rotary(query, cos, sin),
                                                 rotary(key, cos, sin))}
                if args.layout == 'bsnd':
                    r1, r2 = cos.unsqueeze(2), sin.unsqueeze(2)
                    operations = {'reference': lambda: apply_rotary_pos_emb(query.transpose(1, 2), key.transpose(1, 2), cos, sin),
                                  'triton_exact': lambda: (rotary(query, cos, sin).transpose(1, 2),
                                                     rotary(key, cos, sin).transpose(1, 2))}
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
                for replay in range(3):
                    query.copy_(torch.randn_like(query))
                    key.copy_(torch.randn_like(key))
                    cos.copy_(torch.randn_like(cos))
                    sin.copy_(torch.randn_like(sin))
                    for graph, _ in captures.values(): graph.replay()
                    torch.npu.synchronize()
                    assert all(torch.equal(a, b) for a, b in zip(captures['reference'][1], captures['triton_exact'][1])), 'Rotary replay changed BF16 output'

                errors = [(a.float() - b.float()).abs() for a, b in zip(captures['reference'][1], captures['triton_exact'][1])]
                assert all(bool(torch.isfinite(value).all()) for value in captures['triton_exact'][1])
                assert all(torch.equal(a, b) for a, b in zip(captures['reference'][1], captures['triton_exact'][1])), 'Rotary changed BF16 output'
                row = {'batch': batch, 'length': length, 'max_abs_error': max(e.max().item() for e in errors), 'samples': []}
                for name in ['reference', 'triton_exact', 'triton_exact', 'reference']:
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
        result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    for graph, _ in captures.values(): graph.reset()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
