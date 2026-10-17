# Exploratory vLLM baseline vs standalone: ttfa_p95_ms

Evidence checks: **FAIL**

PASS means supplied evidence checks passed; it does not establish a statistically significant gain.
Means are equally weighted trial means. SD is sample standard deviation, not a confidence interval.

| Metric / scenario | Direction | Before mean ± SD (n) | After mean ± SD (n) | Improvement | Scope |
|---|---|---:|---:|---:|---|
| ttfa_p95_ms / C1 (ms) | lower | 284.757 ± N/A (n=1) | 63.0267 ± N/A (n=1) | +77.87% | single trial |
| ttfa_p95_ms / C2 (ms) | lower | 374.999 ± N/A (n=1) | 90.8308 ± N/A (n=1) | +75.78% | single trial |
| ttfa_p95_ms / C4 (ms) | lower | 430.125 ± N/A (n=1) | 140.369 ± N/A (n=1) | +67.37% | single trial |
| ttfa_p95_ms / C8 (ms) | lower | 520.529 ± N/A (n=1) | 199.618 ± N/A (n=1) | +61.65% | single trial |
| ttfa_p95_ms / C16 (ms) | lower | 661.137 ± N/A (n=1) | 360.882 ± N/A (n=1) | +45.41% | single trial |
| ttfa_p95_ms / C32 (ms) | lower | 1016.53 ± N/A (n=1) | 1021.94 ± N/A (n=1) | -0.53% | single trial |
| ttfa_p95_ms / C64 (ms) | lower | 1793.43 ± N/A (n=1) | 1259.49 ± N/A (n=1) | +29.77% | single trial |

![Before and after](comparison.png)

## Input and provenance

```json
{
  "title": "Exploratory vLLM baseline vs standalone: ttfa_p95_ms",
  "synthetic": false,
  "context": {
    "hardware": "One Ascend 910B1, physical device 0, shared host",
    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice BF16",
    "workload_sha256": "5007716c83b29d0247b2c6291b9704c652e01a94362c19e90ea3cb34987060c7",
    "baseline_revision": "b164fea797531c1fa10f8b87829632cad006b9fc",
    "candidate_revision": "3ef961a70b59ff653f1568d852121ca07370ccf6",
    "source_logs": [
      "runs/vllm-sweep-round1-full128/status.json",
      "runs/sglang-sweep-round1-full128/status.json",
      "runs/continuous-native-rope-sampling-full/status.json"
    ],
    "timing_scope": "Warm service, client dispatch through terminal PCM event; TTFA first nonempty PCM"
  },
  "checks": {
    "comparable": "unknown",
    "correctness": "fail",
    "lifecycle": "pass"
  },
  "runs": {
    "before": {
      "completed": 896,
      "failed": 0
    },
    "after": {
      "completed": 896,
      "failed": 0
    }
  },
  "notes": [
    "Single independent run per arm; no repeat error bars or statistical significance claim.",
    "vLLM has higher measured throughput than SGLang at every concurrency; all four metrics compare against that same selected framework.",
    "C16 candidate Chinese CER 0.03807947 exceeds frozen 0.03566225 limit; all other full ASR/speaker cohort gates pass.",
    "Both arms exited 0. Shared-host temporal variability is not controlled; full three-repeat acceptance remains pending.",
    "Do not combine gains from other configurations with this measured candidate."
  ],
  "metrics": [
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C1",
      "before": [
        284.75743043236434
      ],
      "after": [
        63.0266873165965
      ]
    },
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C2",
      "before": [
        374.9993046745658
      ],
      "after": [
        90.83082168363035
      ]
    },
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C4",
      "before": [
        430.1252225879579
      ],
      "after": [
        140.36910058930513
      ]
    },
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C8",
      "before": [
        520.5294966232032
      ],
      "after": [
        199.6178788598624
      ]
    },
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C16",
      "before": [
        661.1373523715883
      ],
      "after": [
        360.88213864713896
      ]
    },
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C32",
      "before": [
        1016.5279964916408
      ],
      "after": [
        1021.9438086729496
      ]
    },
    {
      "name": "ttfa_p95_ms",
      "unit": "ms",
      "direction": "lower",
      "scenario": "C64",
      "before": [
        1793.4304440394044
      ],
      "after": [
        1259.4859242904931
      ]
    }
  ]
}
```
