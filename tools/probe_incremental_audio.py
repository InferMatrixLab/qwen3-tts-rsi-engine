"""Retain real-prompt incremental codec audio for independent ASR/speaker checks."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import torch
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.incremental_codec import CodecState, IncrementalCodec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 64, 65])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
data = args.workload.read_bytes()
payloads = [dict(json.loads(line), instructions='') for line in data.splitlines() if line.strip()]
(args.output / 'runspec.json').write_text(json.dumps({'workload': str(args.workload),
    'workload_sha256': hashlib.sha256(data).hexdigest(), 'sample_rate': 24000, 'frame_rate': 12.5}, indent=2)+'\n')
model = CustomVoiceModel(args.model, 'npu:0', predictor_graph='whole', talker_graph=True)
codec = IncrementalCodec(model.wrapper.model.speech_tokenizer.model.decoder)
result = {'status': 'INCONCLUSIVE', 'scope': 'real-prompt incremental codec diagnostic, independent quality pending',
          'requests': []}


def save(path, pcm):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(pcm)


try:
    with torch.inference_mode():
        for index in args.indices:
            codes = []
            def observe(module, inputs, output):
                value = output.hidden_states[-1]
                if value is not None:
                    codes.append(value.clone())
            hook = model.wrapper.model.talker.register_forward_hook(observe)
            try:
                reference = model.generate([payloads[index]])[0]
            finally:
                hook.remove()
            if reference['finish_reason'] != 'stop':
                raise ValueError('Reference generation hit its length limit')
            tokens = torch.cat(codes, dim=0)[:reference['codec_frames']].T.unsqueeze(0)
            torch.save(tokens.cpu(), args.output / f'{index:05d}.codes.pt')
            state, chunks, start = CodecState(), [], 0
            while start < tokens.shape[-1]:
                size = 2 if start < 4 else 4 if start < 8 else 8
                chunks.append(codec.decode(tokens[..., start:start + size], state))
                start += size
            waveform = torch.cat(chunks, dim=-1)[0, 0].float().cpu().numpy()
            if not np.isfinite(waveform).all():
                raise ValueError('Nonfinite incremental waveform')
            pcm = (np.clip(waveform, -1, 1) * 32767).astype('<i2').tobytes()
            assert len(pcm) == len(reference['pcm'])
            save(args.output / f'{index:05d}.wav', pcm)
            save(args.output / f'{index:05d}.reference.wav', reference['pcm'])
            difference = np.frombuffer(pcm, dtype='<i2').astype(np.float64) - np.frombuffer(reference['pcm'], dtype='<i2')
            result['requests'].append({'index': index, 'success': True, 'codec_frames': reference['codec_frames'],
                'finish_reason': 'stop', 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                'pcm_bytes': len(pcm), 'audio_seconds': len(pcm)/48000,
                'reference_pcm_sha256': hashlib.sha256(reference['pcm']).hexdigest(),
                'max_pcm_error': float(np.abs(difference).max()),
                'rms_pcm_error': float(np.sqrt(np.mean(difference ** 2)))})
            print(result['requests'][-1], flush=True)
            (args.output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    result['status'] = 'DIAGNOSTIC_ONLY'
finally:
    (args.output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    model.close()
