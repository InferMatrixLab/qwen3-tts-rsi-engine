# U002 actual single-910B3 service evidence

[RFC #2](https://github.com/InferMatrixLab/qwen3-tts-rsi-engine/issues/2) predates implementation and measurement. Status: **PASS** against its frozen, low-concurrency opt-in gates. This is a separate method from rejected U001; no paired Q/K kernel is included.

Base: `9f190577aa2a147c6840560a56c5996a6c4879a4`. Measured uncommitted candidate snapshot: `b00e46522eff0f40044fa4e51c474094ac68c9bcb4eed002cf047d38efa8cb62`. The later commit is bound to all measured Python blobs by `source-applicability.json`. Supplemental metadata tooling/tests are validated separately in `final-checks.json`.

## Review revision applicability

The campaign below measured commit `890e64ecd0ede47dd395d280886688ccc3dfcfad` (the initial uncommitted snapshot was later bound to that commit). Historical manifests, source-applicability statements and audit reports describe that measured revision; they are retained unchanged. The subsequent review fix adds metadata validation at predictor construction and retains capture-time validation. Its separate A2 checks are recorded in [review R1](../u002-review-r1/README.md); the original three-pair timings are not relabeled as fresh measurements of that fix.

Enable the option only for C1–C2. At C≥4, expect no useful throughput benefit: the small aggregate gains have two negative repeats per load, while conservative p95 TTFA regresses up to 4.30% at C16 and 3.96% at C32.

## Results

| C | Control frames/s | Fixed RoPE frames/s | Paired geometric gain | Individual paired gains | Conservative p95 change | Gate |
|---:|---:|---:|---:|---|---:|---|
| 1 | 63.20 ± 0.36 | 64.30 ± 0.35 | +1.73% | +1.78%, +0.63%, +2.80% | +0.64% | PASS |
| 2 | 97.37 ± 0.81 | 99.14 ± 0.77 | +1.81% | +1.62%, +0.34%, +3.50% | +0.40% | PASS |
| 4 | 159.38 ± 1.08 | 159.80 ± 1.13 | +0.26% | -0.18%, -0.84%, +1.83% | +1.05% | PASS |
| 8 | 248.16 ± 2.96 | 249.52 ± 1.70 | +0.55% | -0.24%, -0.81%, +2.72% | +1.21% | PASS |
| 16 | 359.54 ± 0.90 | 359.69 ± 3.80 | +0.04% | -0.41%, -0.91%, +1.44% | +4.30% | PASS |
| 32 | 460.24 ± 2.17 | 461.43 ± 3.95 | +0.26% | -0.76%, -0.16%, +1.72% | +3.96% | PASS |
| 64 | 582.64 ± 2.24 | 583.01 ± 4.84 | +0.06% | -0.39%, -0.85%, +1.44% | +1.40% | PASS |

Throughput is total emitted audio seconds × 12.5 / timed wall seconds. Absolute throughput entries show mean ± sample SD of three trials. The gain is the geometric mean of the three paired ratios, not the ratio of displayed means. Conservative tail change is max(candidate p95)/min(control p95)−1 across all repeats. Original per-request timings, output identities, cohort metrics and lifecycles are provided as redacted JSON.

The primary C1 gate requires geometric gain >1% with all three paired ratios >1. Every cohort requires throughput ratio ≥0.97 and conservative p95 ratio ≤1.05. Scope is an explicitly enabled small-concurrency option; C4–C64 do not improve consistently across all three pairs and the default remains unchanged. This does not meet or claim the historical >5%-at-every-cohort target.

## Actual execution and accuracy

Three fresh service pairs ran control/candidate, candidate/control, control/candidate. Each arm ran C1/2/4/8/16/32/64 with 128 timed requests and max(2,C) warmup requests. All 5,376 timed requests reached real stop/EOS; all 2,688 paired retained PCM identities matched byte-for-byte. The controller opened all original WAVs and checked mono/PCM16/24kHz format, PCM SHA256, duration and complete streamed-byte coverage. Six servers and 42 benchmark processes exited0; no forced kill. Warmup requests are excluded from these coverage/performance claims.

The model is `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`, BF16, on one Ascend 910B3. The frozen 128-request English/Chinese corpus has SHA256 `5007716c83b29d0247b2c6291b9704c652e01a94362c19e90ea3cb34987060c7`; seed42 and max_new_tokens2048 were unchanged. It is available from the [pinned upstream corpus](https://github.com/InferMatrixLab/qwen3-tts-rsi-engine/blob/9f190577aa2a147c6840560a56c5996a6c4879a4/evidence/campaign-20261017/plan/workload-128.jsonl). A2 source/model content was verified unchanged. Installed dependency versions and the container image digest are in `identity.json`. `model-upstream-binding.json` independently binds both actual weight hashes to authoritative public Hugging Face revision metadata.

The graph probe recorded 40 matched predictor cases across five batch sizes, changed embeddings/draws and both embedding modes. It also exercised returned-output ownership. Initial checks: 6 unittest cases and 63 pytest tests, all passed. Final checks include 10 unittest cases, 67 pytest tests, syntax, package import without loading inference dependencies and invalid CLI rejection. Fourteen existing pytest warnings and five passing subtests are retained in the allowlisted summaries. The Windows full unittest discovery was INCONCLUSIVE because accelerator dependencies were unavailable (actual exit1); ten focused CPU tests passed there. Full suites passed in the A2 environment.

PCM equivalence establishes relative output preservation only for this fixed model/configuration/corpus. No held-out speech, independent MOS/WER, other precision/hardware, long-running production or generalized concurrency improvement is claimed.

## Integrity, redaction and replay

Raw originals are access-controlled outside Git. Archive label `U002_PRIVATE_ORIGINALS`, 1462835200 bytes, SHA256 `1f744c0ccadeb4d587df9f1242740b0786131a03c115a046c847e04c48909bf6`. Actual connection identifiers, resolved paths, endpoints, input texts and raw exception/warning lines are deliberately omitted from shared artifacts. Retained numeric timings, source/output hashes, process exits and revision identities permit the metadata checks; original audio is needed for a new independent WAV-byte audit.

```bash
python tools/audit_fixed_rope_campaign.py --evidence evidence/u002-fixed-predictor-rope --output "<AUDIT_REPORT>"
```

`audit-a2.json` and `audit-local.json` are actual stdlib metadata recomputations, not fresh NPU runs or independent reads of private audio. Failure and missing evidence stay FAIL/INCONCLUSIVE. The full common CLI, alternatives, cache lifetime and supported scope are documented in [the iteration design](../../docs/iterations/u002.md). This task opens a code PR for review; it does not merge, publish packages or activate production systems.
