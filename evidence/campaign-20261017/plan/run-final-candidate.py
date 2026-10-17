"""Run three frozen same-card trials after the owned baseline releases NPU 0."""
import json
import socket
import subprocess
import time
from pathlib import Path

root = Path('/home/why/qwen3-tts-engine')
frozen = Path('/home/why/qwen3-tts-research/engine-direct-prompt-frozen')
sha = 'e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a'
quality = json.loads((root / 'evidence/campaign-20261017/results/direct-prompt-round1-quality-progress.json').read_text())
assert len(quality['cohorts']) == 7
for row in quality['cohorts']:
    assert all(row['asr'][lang]['status'] == 'PASS' for lang in ('en', 'zh'))
    assert row['speaker']['status'] == 'PASS'
    audit = root / f"evidence/campaign-20261017/results/direct-prompt-round1-c{row['concurrency']}-integrity.json"
    assert json.loads(audit.read_text())['status'] == 'PASS'
print('Waiting for same-card vLLM baseline to finish', flush=True)
deadline = time.monotonic() + 14400
while time.monotonic() < deadline:
    state = json.loads((root / 'runs/vllm-sweep-round3-card0-full128/status.json').read_text())
    if state.get('phase') == 'finished':
        assert state['server_exit'] == state['benchmark_exit'] == 0
        assert len(state['cohorts']) == 7
        assert all(r['metrics']['successful_requests'] == 128 and r['metrics']['failed_requests'] == 0 for r in state['cohorts'])
        rows = subprocess.check_output(['ps', '-eo', 'pgid,stat'], text=True).splitlines()[1:]
        live = [r for r in rows if r.split()[0] == str(state['server_pid']) and not r.split()[1].startswith('Z')]
        with socket.socket() as sock:
            sock.settimeout(2)
            released = sock.connect_ex(('127.0.0.1', 8096)) != 0
        if not live and released:
            break
    time.sleep(10)
else:
    raise TimeoutError('Baseline has not finished and released resources')
git = ['git', '-c', f'safe.directory={frozen}', '-C', str(frozen)]
assert subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() == sha
assert not subprocess.check_output(git + ['status', '--porcelain'], text=True).strip()
for trial in (1, 2, 3):
    output = root / f'runs/continuous-direct-prompt-card0-round{trial}'
    assert not output.exists(), output
    print(f'Starting final candidate trial {trial}', flush=True)
    subprocess.run([
        '/home/why/qwen3-tts-research/venv-engine/bin/python', 'tools/qualify.py',
        '--command-json', 'evidence/campaign-20261017/plan/continuous-direct-prompt-command.json',
        '--output', str(output), '--workload', 'evidence/campaign-20261017/plan/workload-128.jsonl',
        '--model', '/home/why/qwen3-tts-research/model', '--engine', 'standalone-direct-prompt',
        '--source-revision', sha, '--source-directory', str(frozen), '--port', '8099',
        '--requests', '128', '--concurrency-list', '1', '2', '4', '8', '16', '32', '64',
        '--startup-timeout', '1800'], cwd=frozen, check=True)
    state = json.loads((output / 'status.json').read_text())
    assert state['phase'] == 'finished' and state['server_exit'] == state['benchmark_exit'] == 0
    assert len(state['cohorts']) == 7
    assert all(r['metrics']['successful_requests'] == 128 and r['metrics']['failed_requests'] == 0 for r in state['cohorts'])
