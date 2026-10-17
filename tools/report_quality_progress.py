"""Report available per-concurrency quality against frozen margins, without final acceptance."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--vllm', type=Path, required=True)
    parser.add_argument('--sglang', type=Path, required=True)
    parser.add_argument('--vllm-asr-suffix', default='-npu')
    parser.add_argument('--plan', type=Path, default=Path('evidence/campaign-20261017/plan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = read(args.plan / 'contract.json')
    gates = read(args.plan / 'quality-gates.json')
    report = {'status': 'INCONCLUSIVE',
              'scope': 'Summary-level quality progress for one specified trial; no integrity, lifecycle, hardware, repeated-trial or throughput acceptance',
              'candidate': str(args.candidate), 'baselines': [str(args.vllm), str(args.sglang)], 'cohorts': []}

    def valid(summary):
        return (summary is not None and summary.get('scored') == contract['requests_per_concurrency']
                and summary.get('expected') == contract['requests_per_concurrency']
                and summary.get('workload_sha256') == contract['workload_sha256'])

    for c in contract['concurrency']:
        paths = [args.candidate / f'quality-c{c}/summary.json',
                 args.vllm / f'quality-c{c}{args.vllm_asr_suffix}/summary.json',
                 args.sglang / f'quality-c{c}/summary.json']
        asr = [read(p) for p in paths]
        row = {'concurrency': c, 'asr_sources': list(map(str, paths)), 'asr': 'PENDING', 'speaker': 'PENDING'}
        if all(valid(s) for s in asr):
            row['asr'] = {}
            for language, margin in [('en', 'en_wer_absolute_margin'), ('zh', 'zh_cer_absolute_margin')]:
                values = [s['metrics'][language]['error_rate'] for s in asr]
                limit = min(values[1:]) + gates[margin]
                row['asr'][language] = {'candidate': values[0], 'maximum': limit,
                                        'status': 'PASS' if values[0] <= limit else 'FAIL'}
        paths = [root / f'speaker-c{c}/summary.json' for root in (args.candidate, args.vllm, args.sglang)]
        speaker = [read(p) for p in paths]
        row['speaker_sources'] = list(map(str, paths))
        if all(valid(s) for s in speaker):
            if len({s['anchors_sha256'] for s in speaker}) != 1 or len({s['scorer_sha256'] for s in speaker}) != 1:
                row['speaker'] = {'status': 'INCOMPARABLE'}
            else:
                limit = max(s['mean_cosine'] for s in speaker[1:]) - gates['speaker_cosine_absolute_margin']
                value = speaker[0]['mean_cosine']
                row['speaker'] = {'candidate': value, 'minimum': limit, 'status': 'PASS' if value >= limit else 'FAIL'}
        report['cohorts'].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
