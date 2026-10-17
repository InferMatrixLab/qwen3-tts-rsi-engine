# Qwen3-TTS Ascend engine

Independent NPU inference for **Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice** on
Ascend 910B1. Python implements request scheduling, NPU graph execution and
streaming audio; inference does not depend on vLLM-Omni or SGLang-Omni runtime.
Official Qwen model components and weights are reused.

**Acceptance: INCONCLUSIVE — repeated performance is complete; full independent
candidate audio quality validation is still running.**

## Measured performance

The candidate and both pinned baseline frameworks each completed three runs on
physical NPU 0, with 128 requests at C1/2/4/8/16/32/64. All 8,064 timed requests
succeeded and all nine servers exited 0. Every one of the 21 candidate throughput
observations exceeded twice the highest selected baseline observation at the
same concurrency. C32 was the narrowest margin.

| Concurrency | Candidate trial 1 | Candidate trial 2 | Candidate trial 3 |
|---:|---:|---:|---:|
| 1 | 61.72 | 60.68 | 60.96 |
| 2 | 96.38 | 95.19 | 95.59 |
| 4 | 154.82 | 154.13 | 153.74 |
| 8 | 231.84 | 233.43 | 229.69 |
| 16 | 351.51 | 348.41 | 344.15 |
| 32 | 441.46 | 438.93 | 438.81 |
| 64 | 558.62 | 544.32 | 545.23 |

Values are throughput in frames/s. This table alone is not quality-qualified
acceptance. The first final candidate run has completed all independent quality
checks; runs 2 and 3 are still being scored. Each final run is scored separately.

- [All three frameworks: throughput, TTFA p50/p95 and mean RTF](evidence/campaign-20261017/same-card-performance.md)
- [Four-metric charts with sample standard deviation](evidence/campaign-20261017/same-card-three-trial-charts/README.md)
- [Raw values for every trial](evidence/campaign-20261017/results/final-three-arm-performance.json)
- [Frozen workload and acceptance contract](evidence/campaign-20261017/plan/contract.json)
- [Frozen audio-quality margins](evidence/campaign-20261017/plan/quality-gates.json)

The host was shared and trials were sequential. Sample standard deviation is not
a confidence interval. SGLang uses a documented import-only compatibility patch
on pinned main; both baselines use the recorded allocator workaround. These
qualifications remain part of the results.

## Run and reproduce

Start with [REPRODUCE.md](REPRODUCE.md). The measured engine revision is
`e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a`; the complete selected flags are in
[continuous-direct-prompt-command.json](evidence/campaign-20261017/plan/continuous-direct-prompt-command.json).
A bare server invocation uses different defaults and does not reproduce these
measurements. Model weights, CANN and the NPU Python environment must already be
installed. The speech endpoint is `/v1/audio/speech` on port 8099 by default in
the selected command.

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
