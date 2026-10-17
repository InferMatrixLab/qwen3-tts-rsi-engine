"""Compare padded batch prefill against individual prefill, including every KV layer."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from qwen3_tts_engine.continuous import ContinuousModel, SlotArena

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.matmul.allow_hf32 = False
data = args.workload.read_bytes()
payloads = [dict(json.loads(line), instructions='') for line in data.splitlines()]
indices = [0, 1, 64, 65]
model = ContinuousModel(args.model, 'npu:0', max_slots=4, initial_capacity=64)
model.talker.float()
model.arena = SlotArena(model.talker.model, 4, 64)
result = {'status': 'INCONCLUSIVE', 'scope': 'strict FP32 prefill hidden/KV invariants, not service performance',
          'workload_sha256': hashlib.sha256(data).hexdigest(), 'rows': []}
try:
    with torch.inference_mode():
        model.admit_batch([(i, payloads[i]) for i in indices])
        references = []
        for slot, state in enumerate(model.states):
            references.append({'hidden': state.hidden.clone(), 'token': state.token.clone(),
                               'position': state.position, 'rope_delta': state.rope_delta,
                               'kv': [buffer[slot, :, :state.position].clone()
                                      for buffer in model.arena.keys + model.arena.values]})
        model.states.clear()
        for slot, index in enumerate(indices):
            model.admit(index, payloads[index])
            state, reference = model.states[-1], references[slot]
            assert state.position == reference['position'] and state.rope_delta == reference['rope_delta']
            torch.testing.assert_close(state.token, reference['token'], atol=0, rtol=0)
            torch.testing.assert_close(state.hidden, reference['hidden'], atol=1e-4, rtol=1e-4)
            kv_error = 0.0
            for buffer, expected in zip(model.arena.keys + model.arena.values, reference['kv']):
                value = buffer[slot, :, :state.position]
                kv_error = max(kv_error, (value - expected).abs().max().item())
                torch.testing.assert_close(value, expected, atol=1e-4, rtol=1e-4)
            result['rows'].append({'index': index, 'prompt_length': state.position, 'max_kv_error': kv_error,
                                   'max_hidden_error': (state.hidden - reference['hidden']).abs().max().item()})
        result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    model.close()
