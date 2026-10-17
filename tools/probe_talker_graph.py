"""Teacher-forced multi-step decode parity with padded KV and graph replay."""
import argparse
import json
from pathlib import Path
import time
import torch
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.talker_graph import TalkerDecodeGraph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--capacity', type=int, default=256)
args = parser.parse_args()
model = CustomVoiceModel(args.model, 'npu:0')
reference = model.wrapper.model.talker.model
graph = None
result = {'status': 'INCONCLUSIVE', 'scope': 'talker decoder component only',
          'batch_size': args.batch_size, 'capacity': args.capacity, 'rows': []}
try:
    with torch.inference_mode():
        torch.npu.manual_seed_all(1234)
        batch = args.batch_size
        prefill = torch.randn(batch, 20, 2048, device='npu', dtype=torch.bfloat16)
        mask = torch.ones(batch, 20, device='npu', dtype=torch.long)
        output = reference(inputs_embeds=prefill, attention_mask=mask, use_cache=True, return_dict=True)
        cache = output.past_key_values
        inputs = [torch.randn(batch, 1, 2048, device='npu', dtype=torch.bfloat16) for _ in range(12)]
        graph = TalkerDecodeGraph(reference, inputs[0], args.capacity)
        graph.capture()
        graph.cache.load(cache)
        for index, embeddings in enumerate(inputs):
            position = torch.tensor([20 + index], device='npu')
            positions = position.view(1, 1, 1).expand(3, batch, 1)
            mask = torch.ones(batch, 21 + index, device='npu', dtype=torch.long)
            torch.npu.synchronize(); started = time.perf_counter()
            expected = reference(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                                 cache_position=position, past_key_values=cache, use_cache=True,
                                 output_hidden_states=True, return_dict=True).last_hidden_state.clone()
            torch.npu.synchronize(); eager_s = time.perf_counter() - started
            started = time.perf_counter()
            actual = graph.decode(embeddings, positions, position, mask).last_hidden_state
            torch.npu.synchronize(); graph_s = time.perf_counter() - started
            row = {'step': index, 'exact': torch.equal(expected, actual),
                   'max_abs_error': float((expected - actual).abs().max()),
                   'eager_s': eager_s, 'graph_s': graph_s}
            result['rows'].append(row)
            print(row, flush=True)
        result['exact'] = all(row['exact'] for row in result['rows'])
        result['status'] = 'PASS' if result['exact'] else 'FAIL'
        if not result['exact']:
            raise ValueError('Padded graph output differs from reference')
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if graph is not None:
        graph.close()
    model.close()
