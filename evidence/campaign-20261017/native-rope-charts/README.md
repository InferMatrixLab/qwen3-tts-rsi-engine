# Native rotary candidate: exploratory comparison

These four charts compare the same candidate (3ef961a) with the pinned vLLM-Omni first-repeat baseline, which outperformed SGLang throughput at every concurrency. Each bar is one independent service run, so no repeat error bars are shown. Request completion and shutdown pass, but C16 Chinese CER fails the frozen quality gate. Shared-host temporal comparability and three-repeat acceptance remain incomplete. This candidate is **not accepted**.

- [Throughput](throughput_frames_s/comparison.png)
- [TTFA p50](ttfa_p50_ms/comparison.png)
- [TTFA p95](ttfa_p95_ms/comparison.png)
- [Mean RTF](rtf_mean/comparison.png)

Each directory contains SVG, Markdown report, machine-readable summary and original measurement inputs. Baseline selection is based on throughput; latency and RTF do not independently select a different baseline.
