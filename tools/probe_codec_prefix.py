"""Check whether prefix codec decoding can stream unchanged PCM samples."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from qwen3_tts_engine.model import CustomVoiceModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
model = CustomVoiceModel(args.model, 'npu:0', predictor_graph='whole', talker_graph=True)
rows = []
try:
    with torch.inference_mode():
        payloads = [json.loads(line) for line in args.workload.read_text().splitlines()]
        for index in [0, 1, 64, 65]:
            codes = []
            def observe(module, inputs, output):
                value = output.hidden_states[-1]
                if value is not None:
                    codes.append(value.clone())
            hook = model.wrapper.model.talker.register_forward_hook(observe)
            try:
                result = model.generate([dict(payloads[index], instructions='')])[0]
            finally:
                hook.remove()
            codes = torch.cat(codes, dim=0)[:result['codec_frames']]
            expected = np.frombuffer(result['pcm'], dtype='<i2')
            for length in [1, 2, 4, 8, 16]:
                started = time.perf_counter()
                waves, rate = model.wrapper.model.speech_tokenizer.decode([{'audio_codes': codes[:length]}])
                pcm = (np.clip(waves[0], -1, 1) * 32767).astype('<i2')
                row = {'index': index, 'frames': length, 'elapsed_s': time.perf_counter() - started,
                       'exact': np.array_equal(pcm, expected[:len(pcm)]),
                       'max_pcm_error': int(np.abs(pcm.astype(np.int32) - expected[:len(pcm)]).max()),
                       'pcm_sha256': hashlib.sha256(pcm.tobytes()).hexdigest()}
                rows.append(row)
                print(row, flush=True)
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'prefix codec PCM comparison', 'rows': rows,
                                      'status': 'PASS' if len(rows) == 20 and all(r['exact'] for r in rows) else 'FAIL'}, indent=2) + '\n')
    model.close()
