# Exact combination versus adjacent control: throughput_frames_s

Evidence checks: **INCONCLUSIVE**

PASS means supplied evidence checks passed; it does not establish a statistically significant gain.
Means are equally weighted trial means. SD is sample standard deviation, not a confidence interval.

| Metric / scenario | Direction | Before mean ± SD (n) | After mean ± SD (n) | Improvement | Scope |
|---|---|---:|---:|---:|---|
| throughput_frames_s / C32 (frames/s) | higher | 369.561 ± N/A (n=1) | 418.047 ± N/A (n=1) | +13.12% | single trial |
| throughput_frames_s / C64 (frames/s) | higher | 490.855 ± N/A (n=1) | 526.225 ± N/A (n=1) | +7.21% | single trial |

![Before and after](comparison.png)

## Input and provenance

```json
{
  "title": "Exact combination versus adjacent control: throughput_frames_s",
  "synthetic": false,
  "context": {
    "hardware": "Ascend 910B1 physical NPU 6, shared host",
    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice BF16",
    "workload": "Frozen 128-record corpus per concurrency",
    "baseline_revision": "f8c3d0e1a871b6af6658dd00e50665a09f310200",
    "candidate_revision": "b3498718dc31a49993cce4e02300a2a87c75aeb7",
    "source_logs": [
      "runs/continuous-valid-length-control-c32-c64-retry/status.json",
      "runs/continuous-exact-combined-c32-c64/status.json"
    ],
    "wave_identity": {
      "continuous-valid-length-control-c32-c64-retry": {
        "32": 128,
        "64": 128
      },
      "continuous-exact-combined-c32-c64": {
        "32": 128,
        "64": 128
      }
    }
  },
  "checks": {
    "comparable": "unknown",
    "correctness": "pass",
    "lifecycle": "pass"
  },
  "runs": {
    "before": {
      "completed": 256,
      "failed": 0
    },
    "after": {
      "completed": 256,
      "failed": 0
    }
  },
  "notes": [
    "One independent run per arm; no error bars or statistical significance claim.",
    "Same physical NPU, shared-host interference and temporal variation not excluded.",
    "Quality transferred only by complete WAV byte identity to previously scored full candidate; not independent scoring reruns.",
    "Control then combination. This is a direct measured combination, not multiplied isolated gains.",
    "Not a comparison to both baseline frameworks or final three-repeat 2x acceptance."
  ],
  "metrics": [
    {
      "name": "throughput_frames_s",
      "unit": "frames/s",
      "direction": "higher",
      "scenario": "C32",
      "before": [
        369.5605350035445
      ],
      "after": [
        418.04745299545476
      ]
    },
    {
      "name": "throughput_frames_s",
      "unit": "frames/s",
      "direction": "higher",
      "scenario": "C64",
      "before": [
        490.854909446518
      ],
      "after": [
        526.2245949851465
      ]
    }
  ]
}
```
