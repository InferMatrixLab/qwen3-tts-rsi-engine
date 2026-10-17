# Exploratory first baseline sweep

Status: INCONCLUSIVE. This is one repetition, with independent quality scoring still running.

The [baseline charts](baseline-charts/README.md) provide PNG/SVG comparisons and
machine-readable inputs for all four metrics. They compare the two upstream
baselines only; they are not a standalone-engine speedup claim.

vLLM-Omni main `b164fea797531c1fa10f8b87829632cad006b9fc`, jemalloc environment workaround, one physical NPU 0, 128 fixed requests per concurrency. All 896 timed requests completed; benchmark and server exited 0.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 12.297 | 270.93 | 284.76 | 1.0210 |
| 2 | 22.943 | 324.36 | 375.00 | 1.0921 |
| 4 | 43.831 | 344.96 | 430.13 | 1.1361 |
| 8 | 79.491 | 391.49 | 520.53 | 1.2217 |
| 16 | 124.596 | 498.53 | 661.14 | 1.3816 |
| 32 | 200.265 | 709.22 | 1016.53 | 1.7185 |
| 64 | 242.077 | 1395.92 | 1793.43 | 2.6148 |

The host also ran separate-device implementation probes and ASR qualification during this exploratory sweep. Final acceptance requires isolated matched repetitions and completed quality gates.

The independent combined graph service completed a separate 8-request C1 qualification on physical NPU 1 at 22.016 frames/s, with server exit 0. This different sample/device cohort does not establish a speedup ratio. It still delivers a complete waveform in one audio event.

SGLang main plus the documented torch 2.10 type compatibility patch is now being scanned on physical NPU 0. No best-baseline or 2x acceptance conclusion is available yet.

## Continuous scheduling candidate

Commit `d8cb066d3812986a8e7474b71a8f86a05c935f47`, physical NPU 1, one repetition of the same 128 requests at each concurrency. All 896 requests completed and the server exited 0. Native RMSNorm, packed QKV, 72-frame codec context and codec batch size 4 were enabled. These are exploratory results on a different physical card, with other host work running; they do not establish a matched speedup or acceptance.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 27.357 | 152.85 | 158.88 | 0.4627 |
| 2 | 45.679 | 191.27 | 226.92 | 0.5498 |
| 4 | 68.932 | 229.26 | 401.77 | 0.7175 |
| 8 | 97.609 | 296.84 | 786.76 | 0.9956 |
| 16 | 123.621 | 429.92 | 1488.35 | 1.5482 |
| 32 | 154.422 | 697.16 | 3203.42 | 2.3671 |
| 64 | 180.846 | 1215.15 | 6874.90 | 3.8451 |

The all-concurrency 2x objective remains unachieved. Full-corpus candidate quality is still being scored. SGLang C1 ASR completed with English WER 0.011380 and Chinese CER 0.026490; speaker similarity and the remaining cohorts are pending.

Candidate C1 ASR is complete: English WER 0.009957 and Chinese CER 0.049669. The Chinese result exceeds the frozen minimum-baseline-plus-0.01 gate (SGLang C1: 0.026490). This candidate therefore fails the current C1 ASR gate. Traditional/simplified transcription differences contribute to the discrepancy; raw transcripts and the original scoring are retained, and no normalization change or relaxed threshold is used to claim acceptance. Speaker scoring remains pending.

## SGLang full sweep

Pinned main plus the recorded torch type compatibility patch and jemalloc workaround, physical NPU 0. All 896 timed requests completed; server exit 0. This remains an exploratory single repetition; quality scoring for remaining cohorts is running.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 4.275 | 424.17 | 565.99 | 2.9568 |
| 2 | 7.917 | 601.96 | 793.14 | 3.1687 |
| 4 | 14.116 | 658.99 | 986.29 | 3.4984 |
| 8 | 25.026 | 689.92 | 1267.35 | 3.9222 |
| 16 | 38.304 | 762.42 | 1739.05 | 4.7895 |
| 32 | 57.102 | 811.75 | 2462.35 | 6.2362 |
| 64 | 66.607 | 3451.91 | 7305.23 | 10.1708 |
