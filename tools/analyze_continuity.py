"""Estimate unbuffered playback stalls from retained PCM chunk arrival times."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
run = json.loads((args.run / 'results.json').read_text())
spec = json.loads((args.run / 'runspec.json').read_text())
bytes_per_second = spec['sample_rate'] * 2
rows = []
for request in run['requests']:
    if not request['success']:
        raise ValueError('Continuity scoring requires completed valid audio')
    chunks = request['chunks']
    if sum(chunk['bytes'] for chunk in chunks) != request['pcm_bytes']:
        raise ValueError('Chunk byte counts do not match retained waveform')
    play_until = chunks[0]['seconds']
    total_stall = 0.0
    max_stall = 0.0
    for chunk in chunks:
        stall = max(0.0, chunk['seconds'] - play_until)
        total_stall += stall
        max_stall = max(max_stall, stall)
        play_until = max(play_until, chunk['seconds']) + chunk['bytes'] / bytes_per_second
    rows.append({'index': request['index'], 'chunks': len(chunks),
                 'total_stall_seconds': total_stall, 'max_stall_seconds': max_stall})
result = {'scope': 'Diagnostic playback starts immediately at the first received PCM chunk, with no extra buffer',
          'requests': rows, 'mean_total_stall_seconds': sum(r['total_stall_seconds'] for r in rows) / len(rows),
          'max_stall_seconds': max(r['max_stall_seconds'] for r in rows)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
