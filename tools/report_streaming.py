"""Describe chunk arrival gaps and ideal no-extra-buffer playback stalls."""
import argparse
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    left = int(x)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (x - left)


def describe(record):
    if record.get('success') is not True or not record.get('chunks'):
        raise ValueError('Requires successful requests with retained chunks')
    previous = None
    playback_end = 0.0
    stalled = 0.0
    gaps = []
    for chunk in record['chunks']:
        arrival, size = chunk['seconds'], chunk['bytes']
        if not math.isfinite(arrival) or arrival < 0 or not isinstance(size, int) or size < 0 or size % 2:
            raise ValueError('Invalid PCM chunk timing or byte count')
        if size == 0:
            continue  # Terminal/non-audio events do not advance playback.
        if previous is None:
            playback_end = arrival
        else:
            if arrival < previous:
                raise ValueError('Chunk times must be monotonic')
            gaps.append(arrival - previous)
            stalled += max(0.0, arrival - playback_end)
        playback_end = max(playback_end, arrival) + size / 48000
        previous = arrival
    if previous is None:
        raise ValueError('No nonempty PCM chunks')
    return {'index': record['index'], 'chunks': sum(c['bytes'] > 0 for c in record['chunks']),
            'maximum_inter_chunk_seconds': max(gaps, default=0.0),
            'ideal_playback_stall_seconds': stalled}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((args.run / 'runspec.json').read_text())
    rows = [describe(json.loads(p.read_text())) for p in sorted(args.run.glob('[0-9][0-9][0-9][0-9][0-9].json'))]
    if not rows:
        raise ValueError('No request records')
    stalls = [r['ideal_playback_stall_seconds'] for r in rows]
    maxima = [r['maximum_inter_chunk_seconds'] for r in rows]
    report = {'status': 'DIAGNOSTIC_ONLY', 'run': str(args.run),
              'scope': '24 kHz mono PCM16, playback begins at first chunk with no extra buffering. Arrival-based simulation, not measured player behavior or a quality gate. First-audio waiting time excluded.',
              'workload_sha256': spec.get('workload_sha256'), 'requests': len(rows),
              'requests_with_stall': sum(x > 1e-6 for x in stalls),
              'mean_stall_seconds': statistics.mean(stalls), 'p95_stall_seconds': percentile(stalls, .95),
              'p95_request_maximum_inter_chunk_seconds': percentile(maxima, .95), 'samples': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
