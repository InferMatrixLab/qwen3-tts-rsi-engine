"""Check fixed-capacity FIA KV with explicit valid lengths before model integration."""
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
torch.manual_seed(42)
result = {'status': 'INCONCLUSIVE', 'scope': 'synthetic operator diagnostic only', 'cases': []}
graph = None
try:
    with torch.inference_mode():
        for batch in [1, 8, 32, 64]:
            for length in [2, 3, 8, 16]:
                qlen = 2 if length == 2 else 1
                q = torch.randn(batch, 16, qlen, 64, device='npu:0', dtype=torch.bfloat16)
                k = torch.randn(batch, 8, 16, 64, device=q.device, dtype=q.dtype)
                v = torch.randn_like(k)
                mask = torch.zeros(1, 1, qlen, 16, device=q.device, dtype=torch.bool)
                if qlen == 2:
                    mask[..., 0, 1:] = True

                def operation(padded):
                    return torch_npu.npu_fused_infer_attention_score(
                        q, k if padded else k[:, :, :length].contiguous(),
                        v if padded else v[:, :, :length].contiguous(),
                        atten_mask=mask if padded else mask[..., :length].contiguous(),
                        actual_seq_lengths_kv=[length] * batch if padded else None,
                        num_heads=16, num_key_value_heads=8, input_layout='BNSD',
                        scale=64 ** -0.5, sparse_mode=0)[0]

                timings = {}
                errors = []
                for padded in [False, True]:
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3):
                            operation(padded)
                    torch.npu.current_stream().wait_stream(stream)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        output = operation(padded)
                    for replay in range(3):
                        q.copy_(torch.randn_like(q))
                        # Poison unused slots to check masking across graph replays.
                        k[:, :, length:].fill_(100 * (replay + 1))
                        v[:, :, length:].fill_(-100 * (replay + 1))
                        graph.replay()
                        expected = operation(False)
                        errors.append((output.float() - expected.float()).abs().max().item())
                        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(100):
                        graph.replay()
                    torch.npu.synchronize()
                    timings['padded' if padded else 'sliced'] = (time.perf_counter() - start) * 10
                    graph.reset()
                    graph = None
                result['cases'].append({'batch': batch, 'length': length,
                                        'max_abs_error': max(errors), 'milliseconds': timings})
    result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = repr(error)
    raise
finally:
    if graph is not None:
        graph.reset()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
