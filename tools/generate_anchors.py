"""Create fixed same-preset speaker anchors using the unoptimized reference model."""
import argparse
import hashlib
import json
from pathlib import Path
import wave
from qwen3_tts_engine.model import CustomVoiceModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
data = args.inputs.read_bytes()
manifest = {'status': 'INCONCLUSIVE', 'backend': 'official qwen-tts reference; no graph or operator replacements',
            'inputs_sha256': hashlib.sha256(data).hexdigest(), 'records': []}
model = CustomVoiceModel(args.model, 'npu:0')
try:
    for index, line in enumerate(data.splitlines()):
        payload = json.loads(line)
        output = model.generate([payload])[0]
        if output['finish_reason'] != 'stop':
            raise ValueError('Speaker anchor did not finish normally')
        audio = args.output / f'{index:02d}-{payload["voice"].lower()}.wav'
        with wave.open(str(audio), 'wb') as handle:
            handle.setparams((1, 2, output['sample_rate'], 0, 'NONE', 'not compressed'))
            handle.writeframes(output['pcm'])
        row = dict(payload, audio=str(audio.resolve()), pcm_sha256=hashlib.sha256(output['pcm']).hexdigest(),
                   sample_rate=output['sample_rate'], codec_frames=output['codec_frames'])
        manifest['records'].append(row)
        print(payload['voice'], output['codec_frames'], flush=True)
    manifest['status'] = 'PASS'
finally:
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    model.close()
