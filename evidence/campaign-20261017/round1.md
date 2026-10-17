# Exploratory first baseline sweep

Status: INCONCLUSIVE. This is one repetition, with independent quality scoring still running.

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
