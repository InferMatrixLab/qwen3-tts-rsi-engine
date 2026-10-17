"""Score preset speaker consistency with the pinned upstream WavLM verifier."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--anchors', type=Path, required=True)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--scorer-source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
torch.set_num_threads(4)
if args.device.startswith('npu:'):
    import torch_npu
    torch.npu.set_device(args.device)
run = json.loads((args.run / 'results.json').read_text())
specification = json.loads((args.run / 'runspec.json').read_text())
data = Path(specification['workload']).read_bytes()
if hashlib.sha256(data).hexdigest() != specification['workload_sha256']:
    raise ValueError('Workload identity changed')
payloads = [json.loads(line) for line in data.splitlines() if line.strip()]
anchors = json.loads(args.anchors.read_text())
if anchors['status'] != 'PASS':
    raise ValueError('Anchor generation did not pass')
assets = {row['filename']: row for row in json.loads(args.assets.read_text())}
for row in assets.values():
    digest = hashlib.sha256()
    with Path(row['path']).open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != row['sha256']:
        raise ValueError('Speaker verifier checkpoint identity changed')
module_spec = importlib.util.spec_from_file_location('upstream_speaker_similarity', args.scorer_source)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
scorer = module.WavLMSpeakerSimilarity(finetune_checkpoint=assets['wavlm_large_finetune.pth']['path'],
                                      wavlm_base=assets['wavlm_large.pt']['path'], device=args.device)
args.output.mkdir(parents=True, exist_ok=False)
reference_embeddings = {}
for row in anchors['records']:
    pcm, rate = sf.read(row['audio'], dtype='int16')
    if rate != row['sample_rate'] or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']:
        raise ValueError('Anchor audio identity changed')
    embedding = torch.nn.functional.normalize(scorer.embed([row['audio']]).float(), dim=-1)
    reference_embeddings.setdefault(row['voice'].lower(), []).append(embedding)
centroids = {voice: torch.nn.functional.normalize(torch.cat(values).mean(0, keepdim=True), dim=-1)
             for voice, values in reference_embeddings.items()}
rows = []
for record in run['requests']:
    index = record['index']
    audio = args.run / f'{index:05d}.wav'
    pcm, rate = sf.read(audio, dtype='int16')
    if not record['success'] or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != record['pcm_sha256']:
        raise ValueError('Request audio invalid or changed')
    voice = payloads[index]['voice'].lower()
    embedding = scorer.embed([str(audio)]).float()
    cosine = torch.nn.functional.cosine_similarity(centroids[voice], embedding, dim=-1).item()
    if not np.isfinite(cosine):
        raise ValueError('Nonfinite speaker similarity')
    row = {'index': index, 'voice': voice, 'cosine': cosine, 'pcm_sha256': record['pcm_sha256']}
    rows.append(row)
    with (args.output / 'samples.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print(f'Scored speaker {len(rows)}/{len(run["requests"])}', flush=True)
summary = {'status': 'INCONCLUSIVE', 'scored': len(rows), 'expected': len(run['requests']),
           'mean_cosine': float(np.mean([row['cosine'] for row in rows])),
           'by_voice': {voice: float(np.mean([r['cosine'] for r in rows if r['voice'] == voice]))
                        for voice in sorted({r['voice'] for r in rows})},
           'device': args.device, 'anchors_sha256': hashlib.sha256(args.anchors.read_bytes()).hexdigest(),
           'scorer_sha256': hashlib.sha256(args.scorer_source.read_bytes()).hexdigest(),
           'assets': assets, 'workload_sha256': specification['workload_sha256'],
           'definition': 'Cosine to normalized mean of two independently generated same-preset anchor embeddings'}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
