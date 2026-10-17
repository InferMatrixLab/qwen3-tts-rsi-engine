"""Isolate Ascend grouped-attention correctness and graph replay compatibility."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=['fusion', 'infer'], required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
torch.manual_seed(42)
result = {'status': 'INCONCLUSIVE', 'backend': args.backend,
          'scope': 'operator correctness and capture only; not full-model quality or performance', 'cases': []}


def operation(q, k, v, mask):
    if args.backend == 'fusion':
        return torch_npu.npu_fusion_attention(q, k, v, 16, 'BNSD', atten_mask=mask,
                                             scale=128 ** -0.5, keep_prob=1.0, sparse_mode=1)[0]
    return torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=16, num_key_value_heads=8, input_layout='BNSD',
        atten_mask=mask, scale=128 ** -0.5, sparse_mode=0)[0]


graph = None
try:
    with torch.inference_mode():
        for batch, qlen, kvlen in [(1, 2, 2), (1, 1, 16), (4, 1, 512), (32, 1, 512)]:
            q = torch.randn(batch, 16, qlen, 128, device='npu:0', dtype=torch.bfloat16)
            k = torch.randn(batch, 8, kvlen, 128, device='npu:0', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            mask = torch.zeros(batch, 1, qlen, kvlen, device='npu:0', dtype=torch.bool)
            mask[..., 0, -1] = True
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    operation(q, k, v, mask)
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                output = operation(q, k, v, mask)
            for replay in range(3):
                # Reuse addresses but change data and mask to detect frozen arguments.
                q.copy_(torch.randn_like(q))
                mask.zero_()
                mask[..., 0, -(replay % kvlen + 1)] = True
                graph.replay()
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1),
                    attn_mask=~mask)
                error = (output.float() - expected.float()).abs().max().item()
                result['cases'].append({'batch': batch, 'query_length': qlen, 'kv_length': kvlen,
                                        'replay': replay, 'max_abs_error': error})
                torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
            graph.reset()
            graph = None
    torch.npu.synchronize()
    result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
