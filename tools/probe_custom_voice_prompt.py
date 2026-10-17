"""Compare direct CustomVoice prompts with upstream tensors and request settings."""
import argparse
import json
from pathlib import Path
import time

import torch
from qwen3_tts_engine.continuous import ContinuousModel
from qwen3_tts_engine.custom_voice_prompt import CustomVoicePrompt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
model = ContinuousModel(args.model, 'npu:0', max_slots=2, initial_capacity=64)
builder = CustomVoicePrompt(model.model.wrapper)
payloads = [dict(json.loads(line), instructions='') for line in args.workload.read_text().splitlines() if line.strip()]
extra = [dict(payloads[0], instructions='Speak with a calm and reassuring tone.'),
         dict(payloads[64], language='auto'),
         dict(payloads[64], voice='eric', language='chinese')]
result = {'status': 'INCONCLUSIVE', 'scope': 'Prepared BF16 tensors and host/device prompt timing, not complete speech or service throughput',
          'cases': [], 'measurements': []}
try:
    with torch.inference_mode():
        for index, payload in enumerate(payloads + extra):
            expected = model.prepare_prompt(payload)
            actual = builder.build(payload)
            errors = {}
            for key in ['inputs_embeds', 'attention_mask', 'trailing_text_hidden', 'tts_pad_embed']:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                errors[key] = (actual[key].float() - expected[key].float()).abs().max().item()
            for key in ['do_sample', 'top_k', 'top_p', 'temperature', 'repetition_penalty',
                        'subtalker_dosample', 'subtalker_top_k', 'subtalker_top_p', 'subtalker_temperature', 'max_new_tokens']:
                assert actual[key] == expected[key], key
            result['cases'].append({'index': index, 'length': actual['inputs_embeds'].shape[1], 'max_abs_errors': errors})
        for arm in ['upstream', 'direct', 'direct', 'upstream']:
            operation = model.prepare_prompt if arm == 'upstream' else builder.build
            torch.npu.synchronize()
            start = time.perf_counter()
            for payload in payloads:
                operation(payload)
            torch.npu.synchronize()
            result['measurements'].append({'arm': arm, 'requests': len(payloads),
                                           'mean_ms': (time.perf_counter() - start) * 1000 / len(payloads)})
        result['status'] = 'DIAGNOSTIC_ONLY'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = repr(error)
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    model.close()
