# Qwen3-TTS Ascend engine

Independent NPU inference for **Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice** on
Ascend 910B1. Python implements request scheduling, NPU graph execution and
streaming audio; inference does not depend on vLLM-Omni or SGLang-Omni runtime.
Official Qwen model components and weights are reused.

**Acceptance: PASS — all three performance repetitions and full independent
audio-quality validation satisfy the frozen contract.**

[Final acceptance and evidence](evidence/campaign-20261017/acceptance.md).

Comparison reports with launch and benchmark commands: [local 910B1](compare.md)
and [provided 910B3 report](compare-910b3.md). Their workloads and environments differ.

## Measured performance

The candidate and both pinned baseline frameworks each completed three runs on
physical NPU 0, with 128 requests at C1/2/4/8/16/32/64. All 8,064 timed requests
succeeded and all nine servers exited 0. Every one of the 21 candidate throughput
observations exceeded twice the highest selected baseline observation at the
same concurrency. C32 was the narrowest margin.

Values are **mean ± sample standard deviation across three runs**, on one
Ascend 910B1 with BF16 and the same pinned model and workload. Higher throughput
is better; lower TTFA and RTF are better. TTFA includes queueing, from dispatch
to first nonempty PCM; p50/p95 are averages of trial percentiles, not pooled
percentiles. Throughput is decoded audio seconds × 12.5 / cohort wall time;
RTF is the mean per-request completion time / generated audio duration.

| Concurrency | Framework | Throughput (frames/s) ↑ | TTFA p50 (ms) ↓ | TTFA p95 (ms) ↓ | RTF (mean) ↓ |
|---:|---|---:|---:|---:|---:|
| 1 | SGLang-Omni | 4.31 ± 0.04 | 439.72 ± 17.08 | 556.22 ± 9.41 | 2.922 ± 0.036 |
| 1 | vLLM-Omni | 12.71 ± 0.39 | 263.51 ± 6.94 | 273.21 ± 10.52 | 0.989 ± 0.030 |
| 1 | Independent engine | 61.12 ± 0.54 | 50.41 ± 0.39 | 51.97 ± 0.58 | 0.206 ± 0.002 |
| 2 | SGLang-Omni | 7.97 ± 0.05 | 599.49 ± 2.64 | 788.08 ± 4.73 | 3.146 ± 0.020 |
| 2 | vLLM-Omni | 24.32 ± 1.23 | 305.15 ± 16.96 | 349.79 ± 22.07 | 1.032 ± 0.053 |
| 2 | Independent engine | 95.72 ± 0.61 | 67.55 ± 0.31 | 79.76 ± 0.53 | 0.261 ± 0.002 |
| 4 | SGLang-Omni | 14.35 ± 0.31 | 632.13 ± 24.59 | 919.25 ± 59.49 | 3.443 ± 0.075 |
| 4 | vLLM-Omni | 46.43 ± 2.32 | 320.69 ± 21.08 | 402.43 ± 24.04 | 1.073 ± 0.056 |
| 4 | Independent engine | 154.23 ± 0.55 | 89.56 ± 0.70 | 107.59 ± 0.90 | 0.322 ± 0.001 |
| 8 | SGLang-Omni | 25.59 ± 0.50 | 662.13 ± 24.15 | 1123.95 ± 125.27 | 3.840 ± 0.073 |
| 8 | vLLM-Omni | 83.36 ± 4.32 | 375.63 ± 16.61 | 492.49 ± 25.92 | 1.169 ± 0.059 |
| 8 | Independent engine | 231.65 ± 1.87 | 113.23 ± 1.27 | 162.26 ± 3.51 | 0.424 ± 0.004 |
| 16 | SGLang-Omni | 40.20 ± 1.71 | 716.00 ± 45.37 | 1605.89 ± 115.90 | 4.649 ± 0.131 |
| 16 | vLLM-Omni | 134.52 ± 8.81 | 466.73 ± 29.24 | 643.21 ± 15.53 | 1.313 ± 0.059 |
| 16 | Independent engine | 348.02 ± 3.69 | 146.89 ± 0.96 | 267.99 ± 8.56 | 0.561 ± 0.006 |
| 32 | SGLang-Omni | 57.62 ± 0.68 | 808.37 ± 20.98 | 2435.48 ± 23.69 | 6.127 ± 0.155 |
| 32 | vLLM-Omni | 209.45 ± 7.96 | 669.10 ± 34.99 | 994.41 ± 40.55 | 1.630 ± 0.077 |
| 32 | Independent engine | 439.73 ± 1.50 | 214.90 ± 0.89 | 489.36 ± 7.81 | 0.813 ± 0.003 |
| 64 | SGLang-Omni | 71.60 ± 4.43 | 3410.41 ± 663.46 | 7436.21 ± 304.93 | 9.386 ± 0.706 |
| 64 | vLLM-Omni | 228.07 ± 16.73 | 1333.99 ± 53.72 | 1721.06 ± 62.77 | 2.477 ± 0.120 |
| 64 | Independent engine | 549.39 ± 8.01 | 582.98 ± 19.72 | 920.75 ± 39.28 | 1.212 ± 0.020 |

The weakest individual engine throughput observation relative to the highest
baseline observation at the same concurrency is **2.047683×** (C32). This is an
observed-run comparison, not a statistical confidence bound.

- [All three frameworks: throughput, TTFA p50/p95 and mean RTF](evidence/campaign-20261017/same-card-performance.md)
- [Four-metric charts with sample standard deviation](evidence/campaign-20261017/same-card-three-trial-charts/README.md)
- [Raw values for every trial](evidence/campaign-20261017/results/final-three-arm-performance.json)
- [Frozen workload and acceptance contract](evidence/campaign-20261017/plan/contract.json)
- [Frozen audio-quality margins](evidence/campaign-20261017/plan/quality-gates.json)

The host was shared and trials were sequential. Sample standard deviation is not
a confidence interval. SGLang uses a documented import-only compatibility patch
on pinned main; both baselines use the recorded allocator workaround. These
qualifications remain part of the results.

## Audio quality and accuracy

All 63 framework/trial/concurrency cohorts passed audio, timing and scoring
integrity checks. Each run uses the same 128 texts: 64 English (Ryan) and 64
Chinese (Vivian). English word error rate (WER) and Chinese character error rate
(CER) are measured with pinned Whisper-large-v3 in FP32 on NPU. Speaker cosine
uses the pinned SGLang-Omni WavLM-large verifier and the same independently
generated voice anchors for all frameworks.

Values are **mean ± sample standard deviation across three runs**. WER/CER are
percentages (lower is better); speaker similarity is raw cosine (higher is better).

| Concurrency | Framework | English WER (%) ↓ | Chinese CER (%) ↓ | Speaker cosine ↑ |
|---:|---|---:|---:|---:|
| 1 | SGLang-Omni | 1.14 ± 0.00 | 2.65 ± 0.00 | 0.7720 ± 0.0000 |
| 1 | vLLM-Omni | 0.85 ± 0.00 | 3.73 ± 0.00 | 0.7709 ± 0.0000 |
| 1 | Independent engine | 1.14 ± 0.00 | 3.15 ± 0.00 | 0.7751 ± 0.0000 |
| 2 | SGLang-Omni | 1.42 ± 0.00 | 3.89 ± 0.00 | 0.7655 ± 0.0000 |
| 2 | vLLM-Omni | 1.14 ± 0.00 | 3.48 ± 0.00 | 0.7682 ± 0.0004 |
| 2 | Independent engine | 1.14 ± 0.00 | 3.15 ± 0.00 | 0.7746 ± 0.0000 |
| 4 | SGLang-Omni | 1.28 ± 0.00 | 3.39 ± 0.00 | 0.7670 ± 0.0000 |
| 4 | vLLM-Omni | 0.57 ± 0.00 | 3.84 ± 0.67 | 0.7725 ± 0.0009 |
| 4 | Independent engine | 1.14 ± 0.00 | 2.98 ± 0.00 | 0.7728 ± 0.0000 |
| 8 | SGLang-Omni | 0.85 ± 0.00 | 3.39 ± 0.00 | 0.7669 ± 0.0000 |
| 8 | vLLM-Omni | 0.85 ± 0.00 | 2.90 ± 0.17 | 0.7741 ± 0.0012 |
| 8 | Independent engine | 1.14 ± 0.00 | 2.98 ± 0.00 | 0.7769 ± 0.0000 |
| 16 | SGLang-Omni | 1.09 ± 0.30 | 2.84 ± 0.59 | 0.7698 ± 0.0014 |
| 16 | vLLM-Omni | 2.75 ± 1.64 | 2.98 ± 0.36 | 0.7731 ± 0.0022 |
| 16 | Independent engine | 0.85 ± 0.00 | 2.73 ± 0.00 | 0.7620 ± 0.0000 |
| 32 | SGLang-Omni | 1.00 ± 0.00 | 3.17 ± 0.25 | 0.7763 ± 0.0021 |
| 32 | vLLM-Omni | 0.95 ± 0.08 | 3.73 ± 0.38 | 0.7725 ± 0.0038 |
| 32 | Independent engine | 0.85 ± 0.00 | 3.15 ± 0.00 | 0.7676 ± 0.0000 |
| 64 | SGLang-Omni | 0.71 ± 0.00 | 3.09 ± 0.33 | 0.7705 ± 0.0008 |
| 64 | vLLM-Omni | 1.14 ± 0.00 | 3.48 ± 0.30 | 0.7710 ± 0.0028 |
| 64 | Independent engine | 1.00 ± 0.00 | 2.73 ± 0.00 | 0.7713 ± 0.0000 |

All 21 engine trial/concurrency comparisons pass the frozen quality margins:
WER and CER may increase by at most **1 percentage point**, and speaker cosine
may decrease by at most **0.02**, relative to the best observed baseline value
for each metric at the same concurrency across both frameworks and all repeats.
All 2,688 engine waveforms were independently scored; there were zero request
failures or length truncations. These automatic metrics establish the specified
quality acceptance, not subjective naturalness or exact waveform equivalence.

- [Per-trial quality values and all acceptance checks](evidence/campaign-20261017/results/final-numerical-audit.json)
- [Frozen quality margins and scorer definitions](evidence/campaign-20261017/plan/quality-gates.json)
- [Request timings, transcripts and speaker scores](evidence/campaign-20261017/final-request-records.tar.gz)

## Run and reproduce

### 1. Prepare the environment and model

The validated setup is **one Ascend 910B1**, Python 3.12, torch 2.10.0,
torch_npu 2.10.0.post4, qwen-tts 0.1.1 and transformers 4.57.3, with CANN and
Triton Ascend already installed. Use a prepared Ascend environment with these
packages; preserve its CANN `PYTHONPATH`. Installing this repository alone does
not install the NPU driver or inference toolchain. Rust is not required.
See [environment details](evidence/campaign-20261017/plan/environment.md).

```bash
git clone https://github.com/amy-why-3459/qwen3-tts-engine.git
cd qwen3-tts-engine
python -m pip install --no-deps -e .
npu-smi info
python -c "import torch, torch_npu, triton, qwen_tts; print(torch.__version__, torch_npu.__version__); print('NPU available:', torch.npu.is_available())"
```

Download the pinned model **before** enabling offline mode (or point
`QWEN_TTS_MODEL` to an existing complete copy). Keep the speech tokenizer files
in the downloaded snapshot.

```bash
export QWEN_TTS_MODEL=/path/to/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
HF_HUB_OFFLINE=0 python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    revision="0c0e3051f131929182e2c023b9537f8b1c68adfe",
    local_dir=os.environ["QWEN_TTS_MODEL"],
)
PY
```

### 2. Start the measured configuration

Select an available NPU. Run the following in the same terminal, from the
repository root. The allocator path is the recorded ARM64 environment workaround;
use the appropriate installed library path for your environment.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

command = json.loads(Path(
    "evidence/campaign-20261017/plan/continuous-direct-prompt-command.json"
).read_text())
command[0] = sys.executable
command[command.index("--model") + 1] = os.environ["QWEN_TTS_MODEL"]
subprocess.run(command, check=True)
PY
```

This loads the recorded optimization flags, including continuous batching,
streaming, NPU graphs and Triton kernels, and listens on **127.0.0.1:8099**.
Startup compiles kernels and prewarms graphs. In another terminal, check readiness:

```bash
curl --fail http://127.0.0.1:8099/health
```

Wait for a successful health response before submitting requests. Stop the
foreground service with `Ctrl+C`. A bare server invocation has different
defaults and does not reproduce the performance table.

### 3. Generate speech and save a WAV

The endpoint accepts **PCM over SSE**, not a direct WAV response. This Python
standard-library client decodes the streamed base64 PCM into `output.wav`
(mono, signed 16-bit little-endian; sample rate supplied by the server):

```bash
python - <<'PY'
import base64
import json
import urllib.request
import wave

payload = {
    "input": "Hello! This speech is generated by the independent Ascend engine.",
    "voice": "Ryan",
    "language": "English",
    "response_format": "pcm",
    "stream_format": "sse",
    "seed": 42,
    "max_new_tokens": 2048,
}
request = urllib.request.Request(
    "http://127.0.0.1:8099/v1/audio/speech",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
finished = False
sample_rate = None
with urllib.request.urlopen(request, timeout=1800) as response, wave.open("output.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    for line in response:
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:])
        if event["type"] == "speech.audio.error":
            raise RuntimeError(event["error"])
        if event["type"] == "speech.audio.delta":
            if sample_rate is None:
                sample_rate = event["sample_rate"]
                wav.setframerate(sample_rate)
            elif event["sample_rate"] != sample_rate:
                raise RuntimeError("Sample rate changed during streaming")
            wav.writeframes(base64.b64decode(event["audio"]))
        elif event["type"] == "speech.audio.done":
            finished = True
            print("Finish reason:", event["finish_reason"])
if not finished:
    raise RuntimeError("Stream ended without completion")
print("Saved output.wav")
PY
```

For Chinese, replace `input` with Chinese text and use `"voice": "Vivian"`,
`"language": "Chinese"`. Optional `instructions` supplies a style instruction.

### 4. Run a concurrency benchmark

With the service running, execute this from the repository root in a second
terminal using the prepared Python environment. Use a fresh output directory;
the following measures one C4 cohort with 128 requests:

```bash
python -m qwen3_tts_engine.benchmark \
  --endpoint http://127.0.0.1:8099/v1/audio/speech \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --engine independent \
  --source-revision "$(git rev-parse HEAD)" \
  --workload evidence/campaign-20261017/plan/workload-128.jsonl \
  --output runs/local-c4 \
  --concurrency 4 --requests 128 --warmup 4 --frame-rate 12.5
```

For C1/2/4/8/16/32/64, change `--concurrency` and `--output`, and set
`--warmup` to **max(2, concurrency)**, matching the measured campaign (C64
uses `--warmup 64`). Startup prewarming covers only selected graph shapes;
new larger prefill/decode shapes can still trigger graph capture on the first
requests. Using only two warmup requests at high concurrency can include this
first-use cost in timed results. Warmup does not guarantee every later shape
is cached; inspect first-request timing and repeat the same cohort when
investigating cold versus warm behavior. A client benchmark
alone does not verify server shutdown or audio quality. For the full three-run
protocol, server lifecycle capture and independent scoring, follow
[REPRODUCE.md](REPRODUCE.md) and `tools/qualify.py`.
The measured engine revision is `e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a`;
[the frozen command](evidence/campaign-20261017/plan/continuous-direct-prompt-command.json)
records all selected flags.

The engine uses continuous batching, request-owned KV/codec state, NPU graphs,
batched prefill and incremental audio decoding. Selected optimizations include
packed projections, native GQA and normalization, fused rotary/KV/codec/sampling
kernels, and direct CustomVoice prompt preparation. Only fixed model embeddings
are cached by prompt preparation; request text and instruction projections are
not cached.

## Evidence and design sources

[Source provenance](evidence/campaign-20261017/plan/sources.md) records OmniRSI
PR #9, Strata, pinned vLLM-Omni/SGLang-Omni benchmarks and local Ascend guidance.
[Environment exceptions](evidence/campaign-20261017/plan/environment.md) record
the exact compatibility qualifications.

[Campaign history](evidence/campaign-20261017/round1.md) and
[historical experiment notes](EXPERIMENTS.md) retain rejected candidates,
operator probes and earlier results. Microbenchmark gains are not multiplied
into service gains. Full speech quality is evaluated separately from exact
operator or waveform identity.

Weights, generated WAV files and verbose logs remain at the recorded local
paths. The repository contains source, compact results, hashes and audit reports.
`tools/export_evidence.py` exports selected per-request timings, transcripts and
speaker scores after all files are complete, and verifies the archive contents.

## Opt-in fixed predictor RoPE (U002)

`--fixed-predictor-rope` computes the predictor's immutable rotary coefficients before NPU graph capture. On the qualified single-910B3 workload, three paired runs measured **1.73% geometric-mean C1 throughput improvement**, with exact PCM for all 2,688 timed request pairs. C4–C64 did not improve consistently across all three pairs; inspect the complete results before enabling at those loads. This option targets low-concurrency serving and is disabled by default. See the [complete cohort results, alternatives, validation and limits](evidence/u002-fixed-predictor-rope/README.md) and [RFC design](docs/iterations/u002.md).
