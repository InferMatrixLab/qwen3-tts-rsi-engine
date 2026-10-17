"""Audit the numerical contract and retained cohorts; provenance/publication are separate."""
import argparse
import json
import math
from pathlib import Path

from audit_cohort import audit, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('evidence/campaign-20261017/plan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract = load(args.plan / 'contract.json')
    gates = load(args.plan / 'quality-gates.json')
    selection = load(args.plan / 'final-comparison-runs.json')
    issues, missing, cohorts = [], [], []
    collected = {a: {} for a in selection['arms']}
    def check(condition, message):
        if not condition:
            issues.append(message)
    for arm, runs in selection['arms'].items():
        check(len(runs) == contract['independent_repetitions'], f'{arm}: repetition count')
        for trial, row in enumerate(runs, 1):
            run = Path(row['run'])
            state = load(run / 'status.json')
            check(state.get('phase') == 'finished' and state.get('server_exit') == state.get('benchmark_exit') == 0, f'{run}: lifecycle')
            check(sorted(r['concurrency'] for r in state['cohorts']) == contract['concurrency'], f'{run}: concurrency coverage')
            for c in contract['concurrency']:
                quality = run / f"quality-c{c}{row['asr_suffix']}"
                speaker = run / f'speaker-c{c}'
                required = [p / name for p in (quality, speaker) for name in ('summary.json', 'samples.jsonl')]
                absent = [str(p) for p in required if not p.is_file()]
                if absent:
                    missing.extend(absent)
                    continue
                result = audit(run / f'requests-c{c}', quality, speaker, contract, Path('runs/quality-anchors/manifest.json'))
                check(result['status'] == 'PASS', f'{run} C{c}: cohort integrity {result["issues"]}')
                a, s = load(quality / 'summary.json'), load(speaker / 'summary.json')
                check(a.get('precision') == 'float32' and a.get('asr_model') == '/dataset/models/whisper-large-v3', f'{run} C{c}: ASR configuration')
                check(a.get('workload_sha256') == s.get('workload_sha256') == contract['workload_sha256'], f'{run} C{c}: quality workload')
                record = {'arm': arm, 'trial': trial, 'run': str(run), 'concurrency': c,
                          'integrity': result, 'en_wer': a['metrics']['en']['error_rate'],
                          'zh_cer': a['metrics']['zh']['error_rate'], 'speaker_mean': s['mean_cosine'],
                          'anchors_sha256': s['anchors_sha256'], 'scorer_sha256': s['scorer_sha256']}
                check(all(math.isfinite(record[k]) for k in ('en_wer', 'zh_cer', 'speaker_mean')), f'{run} C{c}: nonfinite quality')
                cohorts.append(record)
                collected[arm].setdefault(c, []).append(record)
    comparisons = []
    for c in contract['concurrency']:
        references = collected['vllm'].get(c, []) + collected['sglang'].get(c, [])
        candidates = collected['candidate'].get(c, [])
        if len(references) != 2 * contract['independent_repetitions']:
            continue
        limits = {'en_wer': min(r['en_wer'] for r in references) + gates['en_wer_absolute_margin'],
                  'zh_cer': min(r['zh_cer'] for r in references) + gates['zh_cer_absolute_margin'],
                  'speaker_mean': max(r['speaker_mean'] for r in references) - gates['speaker_cosine_absolute_margin']}
        highest = max(r['integrity']['recomputed_metrics']['throughput_frames_s'] for r in references)
        for r in candidates:
            checks = {k: r[k] <= limits[k] for k in ('en_wer', 'zh_cer')}
            checks['speaker_mean'] = r['speaker_mean'] >= limits['speaker_mean']
            checks['speaker_provenance'] = all(r['anchors_sha256'] == b['anchors_sha256'] and r['scorer_sha256'] == b['scorer_sha256'] for b in references)
            ratio = r['integrity']['recomputed_metrics']['throughput_frames_s'] / highest
            checks['throughput_2x'] = ratio >= 2
            check(all(checks.values()), f"candidate trial {r['trial']} C{c}: {checks}")
            comparisons.append({'trial': r['trial'], 'concurrency': c, 'checks': checks,
                                'quality_limits': limits, 'ratio_to_highest_baseline_observation': ratio})
    expected = len(contract['concurrency']) * contract['independent_repetitions']
    if not missing:
        check(len(cohorts) == expected * 3, 'total cohort count')
        check(len(comparisons) == expected, 'total candidate comparison count')
    report = {'status': 'FAIL' if issues else 'PENDING' if missing else 'PASS',
              'scope': __doc__, 'comparison': 'Each candidate repetition versus the strongest observed throughput and quality references across both baseline frameworks and all three repeats; descriptive, not statistical confidence.',
              'issues': issues, 'missing': missing, 'cohorts': cohorts, 'comparisons': comparisons}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], len(cohorts), 'cohorts audited,', len(comparisons), 'comparisons,', len(missing), 'missing files')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
