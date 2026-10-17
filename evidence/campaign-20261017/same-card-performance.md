# Three-framework same-card performance

Status: **INCONCLUSIVE — full independent candidate audio scoring is still running.**

Three independent server runs per framework, Ascend 910B1 physical NPU 0, one NPU, BF16, the same pinned CustomVoice model and 128-record workload at each concurrency. All 8,064 timed requests completed and all nine servers exited 0. SGLang uses the documented import-only compatibility patch on pinned main. Round-1 vLLM device assignment is recorded in the contemporaneous campaign report; its older runner did not capture the environment field.

Values below are the mean ± sample standard deviation of three trial statistics. TTFA is dispatch-to-first-nonempty-PCM, including queueing; its p50/p95 columns average trial percentiles, not pooled request percentiles. Throughput is decoded audio duration × 12.5 divided by cohort wall time. RTF is the arithmetic mean of per-request completion time / audio duration.

| Concurrency | Framework | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
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

## Throughput observations

Every candidate trial exceeded twice the highest individual baseline observation at the same concurrency. This descriptive check does not establish a confidence bound or replace full quality acceptance.

| Concurrency | Minimum candidate / highest baseline observation |
|---:|---:|
| 1 | 4.647× |
| 2 | 3.760× |
| 4 | 3.185× |
| 8 | 2.609× |
| 16 | 2.433× |
| 32 | 2.048× |
| 64 | 2.249× |

All trials were sequential on a shared host; sample SD describes observed variation, not a confidence interval. No significance claim is made.

[Raw three-arm trial values](results/final-three-arm-performance.json), [run selection](plan/final-comparison-runs.json), [frozen contract](plan/contract.json), [quality margins](plan/quality-gates.json), and [four-metric charts](same-card-three-trial-charts/README.md).
