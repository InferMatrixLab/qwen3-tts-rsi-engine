# Reproduce the measured configuration

The measured engine source is `e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a`.
Later commits add evidence and documentation. The engine uses its own Python
scheduler, NPU graph execution, KV state and incremental audio codec; it does not
run vLLM-Omni or SGLang-Omni as its inference backend. Official Qwen model
components and weights are reused. Rust is not required.

## Environment

Use the recorded Ascend 910B1/CANN container environment, Python 3.12,
torch 2.10.0, torch_npu 2.10.0.post4, qwen-tts 0.1.1 and transformers 4.57.3.
The selected configuration also requires the installed Triton Ascend toolchain.
This repository does not install the driver, CANN, torch_npu or model package.
Preserve CANN's existing `PYTHONPATH`. The package's minimal dependencies alone
are not a complete NPU environment.

[Environment details and known upstream exceptions](evidence/campaign-20261017/plan/environment.md)
and [model hashes](evidence/campaign-20261017/results/model-audit.json) record the
actual setup. Pin the CustomVoice model revision to
`0c0e3051f131929182e2c023b9537f8b1c68adfe`, including its speech tokenizer.

Install this project into the prepared engine environment:

```bash
python -m pip install --no-deps -e .
```

## Start the selected engine

Run from this repository. Set the visible device only after checking that it is
available. The allocator preload below is the recorded environment workaround;
it is not a fix for the upstream allocator issue.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export QWEN_TTS_MODEL=/path/to/pinned/Qwen3-TTS-12Hz-1.7B-CustomVoice
python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

command = json.loads(Path(
    'evidence/campaign-20261017/plan/continuous-direct-prompt-command.json'
).read_text())
command[0] = sys.executable
command[command.index('--model') + 1] = os.environ['QWEN_TTS_MODEL']
subprocess.run(command, check=True)
PY
```

The frozen command selects port 8099 and all measured optimization flags.
A bare `python -m qwen3_tts_engine.server` invocation uses different defaults
and does not reproduce the reported configuration. Startup prewarms graphs and
compiles selected kernels; wait for readiness before sending timed requests.
The streaming speech endpoint is `/v1/audio/speech`.

## Measure and score

`tools/qualify.py` starts the server, waits for readiness, warms each concurrency,
runs the benchmark, and records request artifacts and actual server exit status.
Its `--port` must agree with the command JSON. Use a fresh output directory for
each independent repetition. The exact three-trial campaign launcher is
[run-final-candidate.py](evidence/campaign-20261017/plan/run-final-candidate.py);
it contains this host's paths and a predecessor guard, so it is an execution
record rather than a portable installation script.

The [frozen contract](evidence/campaign-20261017/plan/contract.json) specifies
C1/2/4/8/16/32/64, 128 requests each and three independent repetitions. Use
`workload-128.jsonl`, not the historical 84-record `workload.jsonl`.

- Throughput: total decoded audio seconds × 12.5 / cohort wall seconds.
- TTFA: client dispatch to first nonempty PCM, including queueing.
- Mean RTF: arithmetic mean of request completion seconds / audio seconds.
- Quality: independently score every waveform with `tools/evaluate_audio.py`
  and `tools/evaluate_speaker.py`, then run `tools/audit_cohort.py`.

[Quality margins](evidence/campaign-20261017/plan/quality-gates.json) are frozen.
Byte-identical WAVs are a separate diagnostic and do not replace each trial's
independent scoring. A successful request count alone is not final acceptance.

## Results and retained evidence

[Three-framework metric table](evidence/campaign-20261017/same-card-performance.md),
[raw trial values](evidence/campaign-20261017/results/final-three-arm-performance.json),
[charts](evidence/campaign-20261017/same-card-three-trial-charts/README.md), and
[run selection](evidence/campaign-20261017/plan/final-comparison-runs.json) identify
the final comparison. Performance is complete; full candidate quality validation
is still in progress. Shared-host interference and sequential trial order remain
limitations; sample SD is not a confidence interval.

Weights and generated audio are retained outside Git under the recorded paths.
The public repository contains source, compact results, hashes and audit reports.
