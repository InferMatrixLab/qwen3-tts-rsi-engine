# Exploratory baseline charts

These compare the two baseline frameworks, not the standalone candidate. In the
generator's before/after labels, **before is SGLang main plus the recorded
compatibility patch**, and **after is vLLM main**. Each bar is one cohort of 128
requests on physical NPU 0. There is one trial per concurrency, so no repeat
variance estimate is available. Both servers exited 0; comparability and complete
quality checks remain inconclusive.

- [Throughput](throughput_frames_s/report.md)
- [TTFA p50](ttfa_p50_ms/report.md)
- [TTFA p95](ttfa_p95_ms/report.md)
- [Mean RTF](rtf_mean/report.md)

Each directory retains input measurements, PNG/SVG charts, machine-readable
summary and report. The figures use the local `perf-compare` skill's
`scripts/compare.py`. Differences are descriptive and do not establish a
standalone-engine improvement, statistical significance or the 2x target.
