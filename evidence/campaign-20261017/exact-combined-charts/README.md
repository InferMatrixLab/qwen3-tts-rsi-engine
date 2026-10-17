# Direct combination versus adjacent control

Status: **INCONCLUSIVE** for final acceptance. Both arms used physical NPU 6 with 128 requests at each of C32/C64. There is one run per arm, no error bars, and shared-host/temporal variation is not excluded. All 512 WAV files across both arms match the corresponding previously quality-scored candidate files byte-for-byte. This is explicit quality transfer, not independent scoring. Both server processes exited 0.

The control is `continuous-valid-length-control-c32-c64-retry`; the candidate is `continuous-exact-combined-c32-c64`. The candidate combines paired KV writes, exact talker rotary, and prefill prewarming. No isolated gains are compounded.

- [Throughput](throughput_frames_s/comparison.png) · [report](throughput_frames_s/report.md)
- [TTFA p50](ttfa_p50_ms/comparison.png) · [report](ttfa_p50_ms/report.md)
- [TTFA p95](ttfa_p95_ms/comparison.png) · [report](ttfa_p95_ms/report.md)
- [Mean RTF](rtf_mean/comparison.png) · [report](rtf_mean/report.md)

Each directory also contains SVG and machine-readable summary output. Input JSON files retain source revisions and run paths. These compare engine configurations, not final performance against both upstream baselines. Complete same-hardware repeated testing remains outstanding.
