"""Audit retained cohort audio, timings and quality provenance; not a 2x acceptance gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import wave


def load(path):
    return json.loads(Path(path).read_text())


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def distance(left, right):
    row = list(range(len(right) + 1))
    for i, first in enumerate(left, 1):
        following = [i]
        for j, second in enumerate(right, 1):
            following.append(min(row[j] + 1, following[-1] + 1, row[j - 1] + (first != second)))
        row = following
    return row[-1]


def audit(run, quality, speaker, contract, anchors=None):
    run, quality, speaker = map(Path, (run, quality, speaker))
    issues = []
    def check(condition, message):
        if not condition: issues.append(message)
    spec, result = load(run / 'runspec.json'), load(run / 'results.json')
    count = contract['requests_per_concurrency']
    records = result['requests']
    check(len(records) == count, 'request count differs from contract')
    check([r['index'] for r in sorted(records, key=lambda r: r['index'])] == list(range(count)), 'request indices are not complete/unique')
    check(spec['workload_sha256'] == contract['workload_sha256'], 'workload digest differs')
    check(spec['frame_rate'] == contract['frame_rate'], 'frame rate differs')
    check(spec['requests'] == count, 'runspec request count differs')
    pcm_hashes = {}
    for record in records:
        index = record['index']
        prefix = f'request {index}'
        check(record.get('success') is True, f'{prefix}: unsuccessful')
        terminal = record.get('terminal', {})
        check(terminal.get('type') == 'speech.audio.done', f'{prefix}: no successful terminal event')
        check(terminal.get('finish_reason') not in ('length', 'error', 'cancelled'), f'{prefix}: truncated/failed terminal')
        with wave.open(str(run / f'{index:05d}.wav'), 'rb') as audio:
            check((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, spec['sample_rate']), f'{prefix}: PCM format differs')
            frames, rate = audio.getnframes(), audio.getframerate()
            pcm = audio.readframes(frames)
        digest = hashlib.sha256(pcm).hexdigest()
        pcm_hashes[index] = digest
        check(len(pcm) > 0 and len(pcm) == frames * 2 == record['pcm_bytes'], f'{prefix}: PCM length differs')
        check(digest == record['pcm_sha256'], f'{prefix}: PCM digest differs')
        check(math.isclose(frames / rate, record['audio_seconds'], rel_tol=1e-9), f'{prefix}: audio duration differs')
        check(all(math.isfinite(record[k]) and record[k] > 0 for k in ('ttfa_seconds', 'elapsed_seconds', 'audio_seconds')), f'{prefix}: invalid timing')
        check(record['ttfa_seconds'] <= record['elapsed_seconds'], f'{prefix}: TTFA exceeds completion')
        chunks = record['chunks']
        check(sum(chunk['bytes'] for chunk in chunks) == len(pcm), f'{prefix}: chunk bytes differ')
        nonempty = [chunk for chunk in chunks if chunk['bytes'] > 0]
        check(bool(nonempty) and math.isclose(nonempty[0]['seconds'], record['ttfa_seconds'], rel_tol=1e-9), f'{prefix}: first audio timing differs')
    metrics = result['metrics']
    check(metrics['wall_seconds'] > 0, 'invalid wall time')
    if records and metrics['wall_seconds'] > 0:
        calculated = {
            'throughput_frames_s': sum(r['audio_seconds'] for r in records) * spec['frame_rate'] / metrics['wall_seconds'],
            'ttfa_p50_ms': percentile([r['ttfa_seconds'] * 1000 for r in records], .5),
            'ttfa_p95_ms': percentile([r['ttfa_seconds'] * 1000 for r in records], .95),
            'rtf_mean': statistics.mean(r['elapsed_seconds'] / r['audio_seconds'] for r in records)}
        for key, value in calculated.items():
            check(math.isclose(value, metrics[key], rel_tol=1e-8, abs_tol=1e-10), f'{key}: aggregate differs from requests')
    else:
        calculated = {}
    for path, name in [(quality, 'asr'), (speaker, 'speaker')]:
        summary = load(path / 'summary.json')
        samples = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines() if line.strip()]
        check(summary['scored'] == summary['expected'] == len(samples) == count, f'{name}: incomplete quality records')
        check(summary['workload_sha256'] == spec['workload_sha256'], f'{name}: workload mismatch')
        check(sorted(x['index'] for x in samples) == list(range(count)), f'{name}: duplicate/missing indices')
        for sample in samples:
            check(sample['pcm_sha256'] == pcm_hashes.get(sample['index']), f'{name}: request {sample["index"]} scores different PCM')
        if name == 'speaker':
            check(all(math.isfinite(x['cosine']) and -1.000001 <= x['cosine'] <= 1.000001 for x in samples), 'speaker: invalid cosine')
            check(bool(samples) and math.isclose(statistics.mean(x['cosine'] for x in samples), summary['mean_cosine'], rel_tol=1e-8), 'speaker: mean differs')
            if anchors is not None:
                check(hashlib.sha256(Path(anchors).read_bytes()).hexdigest() == summary['anchors_sha256'], 'speaker: anchors digest differs')
        else:
            for locale in ('en', 'zh'):
                selected = [s for s in samples if s['locale'] == locale]
                edits, denominator = 0, 0
                for sample in selected:
                    left, right = sample['normalized_reference'], sample['normalized_transcript']
                    if locale == 'en': left, right = left.split(), right.split()
                    observed = distance(left, right)
                    check(observed == sum(sample[k] for k in ('substitutions', 'deletions', 'insertions')), f'asr: request {sample["index"]} edit count differs')
                    edits += observed
                    denominator += len(left)
                check(len(selected) == summary['metrics'][locale]['samples'] == count // 2, f'asr: {locale} count differs')
                check(denominator > 0 and math.isclose(edits / denominator, summary['metrics'][locale]['error_rate'], rel_tol=1e-8, abs_tol=1e-10), f'asr: {locale} error rate differs')
    return {'status': 'FAIL' if issues else 'PASS', 'scope': 'Retained cohort integrity only; not quality noninferiority, hardware comparability, lifecycle, repeat independence or 2x acceptance',
            'requests': len(records), 'recomputed_metrics': calculated, 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'quality', 'speaker', 'contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--anchors', type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.run, args.quality, args.speaker, load(args.contract), args.anchors)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, wave.Error) as error:
        result = {'status': 'FAIL', 'scope': 'Cohort integrity only', 'issues': [f'{type(error).__name__}: {error}']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], len(result['issues']), 'integrity issues')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
