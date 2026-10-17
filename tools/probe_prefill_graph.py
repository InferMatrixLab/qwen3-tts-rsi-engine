"""Validate prefill graph replay against eager execution at identical padded shapes."""
import argparse
import json
from pathlib import Path

import torch
from qwen3_tts_engine.continuous import ContinuousModel
from qwen3_tts_engine.prefill_graph import PrefillGraph, pad_prefill_batch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--dummy-rows', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.matmul.allow_hf32 = False
model = ContinuousModel(args.model, 'npu:0', max_slots=4, initial_capacity=64)
model.talker.float()
payloads = [dict(json.loads(line), instructions='') for line in args.workload.read_text().splitlines()]
result = {'status': 'INCONCLUSIVE', 'scope': 'strict FP32 prefill replay and optional dummy-row isolation, not BF16 quality or speed', 'dummy_rows': args.dummy_rows, 'rounds': []}
graph = None
try:
    with torch.inference_mode():
        prompts = [model.prepare_prompt(payloads[i]) for i in [0, 1, 64, 65]]
        longest = max(p['inputs_embeds'].shape[1] for p in prompts)
        length = 1 << (longest - 1).bit_length()
        graph = PrefillGraph(model.talker, 4, length)
        graph.capture()
        orders = [[0, 1, 2, 3], [3, 0, 1, 2]] + ([[3, 1, 0], [2]] if args.dummy_rows else [])
        for order in orders:
            batch = 1 << (len(order) - 1).bit_length()
            if graph.inputs.shape[0] != batch:
                graph.close()
                graph = PrefillGraph(model.talker, batch, length)
                graph.capture()
            selected = [prompts[i] for i in order]
            embeds = torch.cat([torch.nn.functional.pad(p['inputs_embeds'], (0, 0, 0, length - p['inputs_embeds'].shape[1])) for p in selected])
            mask = torch.cat([torch.nn.functional.pad(p['attention_mask'], (0, length - p['attention_mask'].shape[1])) for p in selected])
            ids, _ = model.talker.get_rope_index(mask)
            reference = model.talker.model(inputs_embeds=embeds, attention_mask=mask, position_ids=ids,
                                            use_cache=True, output_hidden_states=False, return_dict=True)
            hidden = reference.last_hidden_state[torch.arange(len(order), device='npu:0'), mask.sum(-1) - 1].unsqueeze(1)
            logits = model.talker.codec_head(hidden)[:, -1]
            padded_embeds, padded_mask = pad_prefill_batch(embeds, mask, batch)
            actual_hidden, actual_logits, actual_cache = graph.run(padded_embeds, padded_mask)
            actual_hidden, actual_logits = actual_hidden[:len(order)], actual_logits[:len(order)]
            torch.testing.assert_close(actual_hidden, hidden, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual_logits, logits, atol=1e-5, rtol=1e-5)
            max_kv_error = 0.0
            for actual, expected in zip(actual_cache.layers, reference.past_key_values.layers):
                for value, target in [(actual.keys[:len(order)], expected.keys), (actual.values[:len(order)], expected.values)]:
                    max_kv_error = max(max_kv_error, (value - target).abs().max().item())
                    torch.testing.assert_close(value, target, atol=1e-5, rtol=1e-5)
            result['rounds'].append({'order': order, 'graph_batch': batch, 'max_kv_error': max_kv_error,
                                     'max_hidden_error': (actual_hidden - hidden).abs().max().item(),
                                     'max_logit_error': (actual_logits - logits).abs().max().item()})
        result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if graph is not None:
        graph.close()
    model.close()
