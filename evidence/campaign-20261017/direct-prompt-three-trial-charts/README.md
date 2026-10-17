# Three exploratory candidate trials versus vLLM

**INCONCLUSIVE for final acceptance.** Three independent performance trials per arm are retained. Candidate trials used physical NPU 6; vLLM trials 1/2 used physical NPU 0 and trial 3 used NPU 1, on the shared host. The final same-card comparison remains pending. Candidate independent scoring is incomplete; complete WAV identity across its repeats is separately recorded.

- [Throughput](throughput_frames_s/comparison.png) · [report](throughput_frames_s/report.md)
- [TTFA p50](ttfa_p50_ms/comparison.png) · [report](ttfa_p50_ms/report.md)
- [TTFA p95](ttfa_p95_ms/comparison.png) · [report](ttfa_p95_ms/report.md)
- [Mean RTF](rtf_mean/comparison.png) · [report](rtf_mean/report.md)

Bars are trial means; error bars are sample standard deviations, not confidence intervals. TTFA percentile charts summarize per-trial percentiles, not pooled-request percentiles. All trial values and source paths are retained in the input JSON and report. SVG and machine-readable summaries accompany each chart. No statistical significance or final 2x acceptance is inferred.
