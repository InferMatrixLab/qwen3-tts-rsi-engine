# Exact combination versus adjacent control: rtf_mean

Evidence checks: **INCONCLUSIVE**

PASS means supplied evidence checks passed; it does not establish a statistically significant gain.
Means are equally weighted trial means. SD is sample standard deviation, not a confidence interval.

| Metric / scenario | Direction | Before mean ± SD (n) | After mean ± SD (n) | Improvement | Scope |
|---|---|---:|---:|---:|---|
| rtf_mean / C32 (ratio) | lower | 0.996949 ± N/A (n=1) | 0.86803 ± N/A (n=1) | +12.93% | single trial |
| rtf_mean / C64 (ratio) | lower | 1.37216 ± N/A (n=1) | 1.28203 ± N/A (n=1) | +6.57% | single trial |

![Before and after](comparison.png)

## Input and provenance

```json
{
  "title": "Exact combination versus adjacent control: rtf_mean",
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
      "name": "rtf_mean",
      "unit": "ratio",
      "direction": "lower",
      "scenario": "C32",
      "before": [
        0.9969489647679338
      ],
      "after": [
        0.868030238326315
      ]
    },
    {
      "name": "rtf_mean",
      "unit": "ratio",
      "direction": "lower",
      "scenario": "C64",
      "before": [
        1.372160679251504
      ],
      "after": [
        1.2820338539856255
      ]
    }
  ]
}
```
