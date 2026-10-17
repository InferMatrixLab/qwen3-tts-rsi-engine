"""Integrity audit must reject altered audio, metrics and quality/audio pairing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import wave
import pytest

spec = importlib.util.spec_from_file_location('audit_cohort', Path(__file__).parents[1] / 'tools/audit_cohort.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


@pytest.fixture
def cohort(tmp_path):
    run, quality, speaker = [tmp_path / name for name in ('run', 'quality', 'speaker')]
    for folder in (run, quality, speaker): folder.mkdir()
    requests, asr, speakers = [], [], []
    for index in range(2):
        pcm = bytes([index + 1]) * 7680
        digest = hashlib.sha256(pcm).hexdigest()
        with wave.open(str(run / f'{index:05d}.wav'), 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(pcm)
        requests.append({'index': index, 'success': True, 'terminal': {'type': 'speech.audio.done', 'finish_reason': 'stop'},
                         'pcm_bytes': len(pcm), 'pcm_sha256': digest, 'audio_seconds': .16,
                         'ttfa_seconds': .01 * (index + 1), 'elapsed_seconds': .2 + index * .1,
                         'chunks': [{'bytes': len(pcm), 'seconds': .01 * (index + 1)}]})
        asr.append({'index': index, 'pcm_sha256': digest, 'locale': 'en' if index == 0 else 'zh',
                    'normalized_reference': 'a b' if index == 0 else '你好',
                    'normalized_transcript': 'a c' if index == 0 else '你好',
                    'substitutions': 1 if index == 0 else 0, 'deletions': 0, 'insertions': 0})
        speakers.append({'index': index, 'pcm_sha256': digest, 'cosine': .8 + index * .1})
    data = {'requests': requests, 'metrics': {'wall_seconds': 1., 'throughput_frames_s': 4., 'ttfa_p50_ms': 15., 'ttfa_p95_ms': 19.5, 'rtf_mean': 1.5625}}
    (run / 'results.json').write_text(json.dumps(data))
    (run / 'runspec.json').write_text(json.dumps({'workload_sha256': 'fixed', 'frame_rate': 12.5, 'sample_rate': 24000, 'requests': 2}))
    for folder, samples in [(quality, asr), (speaker, speakers)]:
        (folder / 'samples.jsonl').write_text('\n'.join(json.dumps(s) for s in samples))
    common = {'scored': 2, 'expected': 2, 'workload_sha256': 'fixed'}
    (quality / 'summary.json').write_text(json.dumps(dict(common, metrics={'en': {'samples': 1, 'error_rate': .5}, 'zh': {'samples': 1, 'error_rate': 0.}})))
    (speaker / 'summary.json').write_text(json.dumps(dict(common, mean_cosine=.85)))
    return run, quality, speaker, {'requests_per_concurrency': 2, 'workload_sha256': 'fixed', 'frame_rate': 12.5}


def test_complete_cohort_recomputes_metrics(cohort):
    result = audit_module.audit(*cohort)
    assert result['status'] == 'PASS'
    assert result['recomputed_metrics']['ttfa_p95_ms'] == 19.5


@pytest.mark.parametrize('change', ['pcm', 'metric', 'quality_hash', 'edit_count'])
def test_altered_evidence_is_rejected(cohort, change):
    run, quality, speaker, contract = cohort
    if change in ('pcm', 'metric'):
        path = run / 'results.json'
        data = json.loads(path.read_text())
        if change == 'pcm': data['requests'][0]['pcm_sha256'] = 'incorrect'
        else: data['metrics']['throughput_frames_s'] = 8.
        path.write_text(json.dumps(data))
    else:
        path = quality / 'samples.jsonl'
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if change == 'quality_hash': data[0]['pcm_sha256'] = 'different-audio'
        else: data[0]['substitutions'] = 0
        path.write_text('\n'.join(json.dumps(row) for row in data))
    result = audit_module.audit(*cohort)
    assert result['status'] == 'FAIL' and result['issues']
