"""Export selected per-request performance and quality records, without model/WAV files.

Completeness and archive integrity only: this does not decide acceptance.
Run from the repository root after all selected quality jobs have finished.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('evidence/campaign-20261017/plan'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    selection = json.loads((args.plan / 'final-comparison-runs.json').read_text())
    contract = json.loads((args.plan / 'contract.json').read_text())
    required = [args.plan / name for name in (
        'final-comparison-runs.json', 'contract.json', 'quality-gates.json',
        'workload-128.jsonl', 'sources.md', 'environment.md')]
    required.append(Path('runs/quality-anchors/manifest.json'))
    optional_missing = []
    for arm, runs in selection['arms'].items():
        if len(runs) != contract['independent_repetitions']:
            raise ValueError(f'{arm}: wrong repetition count')
        for row in runs:
            run = Path(row['run'])
            required.append(run / 'status.json')
            for filename in ('provenance.json', 'source.patch', 'hardware.txt'):
                path = run / filename
                if path.is_file():
                    required.append(path)
                else:
                    optional_missing.append(str(path))
            for c in contract['concurrency']:
                required.extend(run / f'requests-c{c}' / name for name in ('results.json', 'runspec.json'))
                for directory in (f"quality-c{c}{row['asr_suffix']}", f'speaker-c{c}'):
                    required.extend(run / directory / name for name in ('summary.json', 'samples.jsonl'))
    required = sorted(set(required))
    for path in required:
        path.resolve().relative_to(root)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print(json.dumps({'status': 'PENDING', 'missing_count': len(missing), 'missing': missing}, indent=2))
        return 1
    for path in required:
        if path.name == 'summary.json':
            data = json.loads(path.read_text())
            if data.get('scored') != contract['requests_per_concurrency'] or data.get('expected') != contract['requests_per_concurrency']:
                raise ValueError(f'Incomplete quality summary: {path}')
    if args.check_only:
        print(json.dumps({'status': 'READY_TO_EXPORT', 'files': len(required), 'optional_missing': optional_missing}))
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [(str(p), p.read_bytes()) for p in required]
    manifest = {
        'scope': 'Selected raw request timings, PCM hashes, transcripts, speaker scores and provenance; not acceptance',
        'excluded': ['Model weights', 'Generated and anchor WAV files', 'Profiler traces and verbose server logs'],
        'optional_missing': optional_missing,
        'files': [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in payloads],
    }
    payloads.append(('archive-manifest.json', (json.dumps(manifest, indent=2) + '\n').encode()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as destination:
        with tarfile.open(fileobj=destination, mode='w:gz') as archive:
            for name, data in payloads:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    with tarfile.open(args.output, 'r:gz') as archive:
        if archive.getnames() != [name for name, _ in payloads]:
            raise ValueError('Archive membership differs')
        for name, data in payloads:
            member = archive.extractfile(name)
            if member is None or hashlib.sha256(member.read()).digest() != hashlib.sha256(data).digest():
                raise ValueError(f'Archive hash mismatch: {name}')
    result = {'status': 'EXPORT_VERIFIED', 'scope': 'Archive content integrity only',
              'path': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'files': len(payloads), 'bytes': args.output.stat().st_size}
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
