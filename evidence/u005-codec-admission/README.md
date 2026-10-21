# U005: dual-stream Codec and request admission search

The declared three-repeat service matrix, known-corpus regression, independent original-media collection and A2/local metadata audits passed for the measured snapshot.

## Qualified service results

| C | Control FPS range | Candidate FPS range | Conservative TP change | Conservative p95 change | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 62.26–62.70 | 64.46–65.09 | +2.80% | -3.25% | PASS |
| 2 | 97.15–97.82 | 96.31–97.29 | -1.55% | +0.33% | PASS |
| 4 | 157.87–158.18 | 160.94–161.56 | +1.74% | -4.17% | PASS |
| 8 | 246.21–246.38 | 255.08–256.97 | +3.53% | -18.22% | PASS |
| 16 | 357.81–359.73 | 389.91–393.09 | +8.39% | -32.47% | PASS |
| 32 | 456.54–458.41 | 525.81–527.78 | +14.70% | -42.21% | PASS |
| 64 | 576.80–579.77 | 706.21–712.41 | +21.81% | -47.52% | PASS |

FPS denotes generated Codec frames per wall-clock second, using 12.5 Codec frames per audio second. Each range spans three runs. Throughput change is min(candidate) / max(control) − 1; p95 change is max(candidate p95) / min(control p95) − 1. Lower p95 is better. A non-primary positive throughput delta inside noise is not a statistically established speedup.

The retained acceptance noise floor is 2.57945%. Primary spread is reported separately in assessment.json and both audit reports. Shared context-change samples: 0; maximum observed telemetry gap: 2.303 s. These observations do not prove exclusive compute or attribute slowdown to another process.

control readiness 25.52–26.04 s; driver HBM ready 28–28%, after cohorts 31–31%. candidate readiness 33.03–33.53 s; driver HBM ready 32–32%, after cohorts 38–38%. These are operational readiness and aggregate driver-memory observations, not a controlled cold-start comparison or peak task allocation.

Evidence: [full metrics](assessment.json), [A2 metadata replay](audit-a2.json), [local metadata replay](audit-local.json), [independent original-media collection](collection.json), [eight actual prerequisite exits](checks.json), [actual request-view tick checks](views-operator.json), [retained unsuccessful full matrix](rejected-admission.json), [rejected native owner alternative](rejected-owner-native.json), [input ablation](input-discriminator.json), and [private archive digest](private-archive.json).

[Source, model and runtime bindings](identity.json), [A/A calibration](calibration.json), [known-corpus regression](regression.json), [Codec state checks](operator.json) and [packed KV/initial sampling checks](packed-operator.json) record their separate verification scopes.

The public metadata auditor recomputes gates and byte/hash accounting. Original PCM equality was independently checked by the collector against access-controlled original WAVs: 2688 formal pairs, 256 A/A pairs and 64 known-regression pairs. Metadata replay alone does not independently access those private WAVs. Image equality comes from live before/after proofs and the collector; it is not an additional image inspection by the metadata auditor.

## Scope and mechanism

Baseline: `a29f8193b557c415175d70919bdfb9bd3dcd51fc` of
[qwen3-tts-rsi-npu-engine](https://github.com/InferMatrixLab/qwen3-tts-rsi-npu-engine).
Target: one Ascend 910B3, BF16 CustomVoice 1.7B, unchanged request scheduling,
Codec batch 4, maximum chunk 8 frames. The candidate CLI additions are
`--codec-streams=2 --packed-kv-load --single-request-views`. Codec streams
default to 1; packed loading and request views default off. This is an incremental engine comparison,
not evidence about the historical C++ implementation or unrelated frameworks.

Two lanes replay separate captured graphs and write only disjoint state rows in
one arena. Strided KV tails use pure copies without changing arithmetic. Events
allow callbacks in original group order; final stream fences protect subsequent
owner-stream slot reset/move, including callback failure. Multiple newly admitted
contiguous Codec slots are cleared once per buffer. Single actual groups use
the existing lane-0 pure-copy graph directly on the owner stream, without lane
events or a separate graph family. Graphs bind their write function at
construction; serving does not mutate a shared dispatcher.

At B1–4/L1–8, candidate prewarm captures 64 graphs versus the default 32. This
costs additional startup and graph storage. HBM/startup evidence and completion
of readiness remain required; component speedups alone do not establish value.
No extra slot-reset graphs are captured. Packed KV loading stacks the original
prefill cache after unchanged arithmetic, then uses one native strided copy for
each request's true prefix across all layers. Per-layer contiguous target views
share one allocation; padding and unrelated rows are preserved. Source packing
adds temporary storage and graph work. Existing graph sampling also covers the
first semantic token, reusing immutable zero inputs and returning owned token
and seen state; seeded random tables and draw selection are unchanged.

When both the actual and graph batch are one, the optional input preparation
path borrows six read-only request views and uploads position/rotary/step
integers once. Existing graph consumers copy into independent static storage
and retain owned outputs. Other batches and padding retain the original path.
The source views stay alive until owner-stream consumption; no in-place write
may use these borrowed values.

## Alternatives and negative evidence

| Search direction | Observation | Decision |
| --- | --- | --- |
| Native valid-prefix attention | One BF16 mismatch in strict graph replay | Reject; no timing after exactness failure |
| Codec regrouping to B8/B16 | Waveform mismatches | Reject; no precision tolerance introduced |
| Snake tile 2048 | Exact component result; C32 pilot improvement about 1.06%, within calibration noise | Do not include in this candidate |
| Independent arenas/two streams | Exact component upper bound, overlap benefit | Use only as discriminator; avoid multiple complete state arenas |
| Shared-copy v1 | Exact, but pair time roughly 57/64 ms versus baseline 16/25 ms | Reject vector-heavy address mapping |
| Destination row alignment only | Some L8 recovery, little margin | Improve source segments instead |
| Linear regular-stride source segments | L8 pair roughly 18.3 ms versus 25.3 ms | Retain pure-copy mechanism, require service validation |
| Wait for both lanes before any output | Throughput benefit; tail latency failed | Reject delivery policy |
| Progressive delivery | Exact service outputs; C32 tail guard still failed | Profile initial admission rather than loosen tail guard |
| Initial profiling v1 | Warmup drain mislabeled as timed phase | INCONCLUSIVE for attribution; retain failed profile |
| Corrected admission profile | Reset overhead visible in inclusive intervals | Batch only contiguous Codec resets; no exclusive-causal claim |
| Range-reset full formal, snapshot 002f9c8c95a0 | 5376 requests / 2688 literal PCM pairs exact; C32/C64 passed but C1/C2 throughput and tails failed | Overall FAIL; not PR-ready |
| Native singleton fallback discriminator | Four shapes × four changed-input replays exact in waveforms and every buffer, actual exit 0; native timings near baseline | Tried as an intermediate candidate, subsequently removed |
| Singleton-only five-cohort screen, snapshot ebac7fe9d38b | 640 literal PCM pairs exact in independent reread; C1 TTFA +5.50% fails; candidate service exit 1 from mixed None/integer shape-key sorting | Unsuccessful, preserve all evidence; fix key and add real close/cache-key regressions |
| Captured native per-slot clears discriminator | Twelve changed-state replays exact; native wall about 0.34 ms versus captured 0.063 ms; actual child exit 0 | Tried as an intermediate candidate, subsequently removed |
| Captured-reset screen, snapshot f2136e56a775 | 640 original PCM pairs exact; all services and benchmarks exited 0; C1/C2 TTFA p95 increased 8.08%/8.12% | Reject despite high-concurrency gains; remove reset graph family |
| Custom KV Triton kernels | Compiler failures; explicit-buffer variant exact in synthetic cases but about 1286 ms versus 1.64 ms native; smaller variants did not finish within probe budget | Reject slow variant; timeout/terminated variants INCONCLUSIVE, never qualified |
| Native foreach KV copies | Three variants exact across six synthetic cases and three changed replays; roughly 0.96 versus 1.61 ms at one row | Retain as alternative, choose packed layout for stronger discriminator |
| Native packed KV synthetic probe | Exact across two variants, six cases and three replays; including packing roughly 0.23 versus 1.63 ms at one row | Integrate behind an explicit flag, require real model and service checks |
| Early real packed timing | Exact model/cache/state results, but control timing reused its static input buffer | Invalid comparative timing; do not use for qualification; final probe uses independent inputs |
| Initial semantic graph sampling | Actual sampler discriminator exact over 96 cases including boundary uniforms and interleaved seen state | Integrate using existing graph-sampling flag; validate actual add_state API |
| Packed admission full formal, snapshot 646520ace810 | All 2688 original PCM pairs exact, all six services/42 benchmarks exit 0; conservative C32/C64 throughput +13.43%/+21.15%, but C1 p95 +5.4945% exceeds 5% | Overall FAIL; preserve all results and search for a lower-overhead input path |
| B1/B2 native owner graph recheck | Actual 64-slot arena; eight shapes, five changed replays each, full state exact before/after 48 component timings; native slower than pure stores in all shapes | Reject; never integrate the proposed extra 16 graphs |
| First input-view harness attempt | Incorrect expected configuration hash stopped before any operator or timing | Harness failure; retain it, correct identity binding in a new directory |
| Six input views | 72 exact input/source/consumer cases; about 0.34–0.37 to 0.17–0.18 ms | Retain as input-preparation discriminator; no service claim |
| Nine-input three-way ablation | 72 exact cases including signed rotary deltas; original 0.69–0.70 ms, views alone 0.48–0.50, views plus packed integers 0.36–0.38 | Select the packed variant for actual-tick and service validation |
| First new-candidate screen controller | Stale global prerequisite record referred to rejected 6465; hash guard rejected it before any service/benchmark launch | INCONCLUSIVE, zero timed requests; bind to campaign-specific checks and use a fresh output directory |

The current 106-file candidate snapshot is
`8a15df152059328216fe9f28a23950a9a788be47ee936f0c2a80847a074e6812`.
Before formal service qualification it passed A2 62 unittest, 124 pytest and
131 subtests, with no skipped cases. Real packed checks cover B1 and B4 prefill, four admission cases,
three changed replays, original/default/packed hidden/logits/cache equality,
state ownership, slot moves, actual arena growth, and 81 actual add_state eager
versus graph state pairs across seeds 17/42/43. Codec checks cover all 32
singleton shapes, sixteen native reset slots, mixed groups, callbacks and
lane/owner/lane transitions without an intervening index upload or explicit
synchronize. These are prerequisite checks, not service qualification.

The input-view probe additionally checks 72 actual NPU input cases and four
actual tick generations against the frozen 6465 parent and current option-off
path. Three single requests use seeds 17/42/43; cancellation forces a 1→2→1
transition. Every tick checks request state, the complete packed KV arena, every
Codec buffer, streamed PCM and normal EOS; random tables stay unchanged. These
component arenas have two slots and share read-only weights and owned-output
graph implementations. Full service comparisons independently use capacity 64.
The frozen unsuccessful parent source is retained in the access-controlled
archive and identified by its complete 103-file manifest and snapshot digest.

Local targeted CPU checks pass (35 tests). The complete Windows CPU discovery
executes 70 tests with eight missing-runtime import errors (qwen_tts/torch_npu),
so its overall result is INCONCLUSIVE. The separate A2 suite supplies the actual
complete target-environment result. A candidate smoke starts in about 33.04 s,
generates two normal-EOS requests and exits normally; it is not a paired startup
or performance comparison.

Deferred directions include CPU random-table transfer batching, further ragged
Talker KV layouts, stream priorities and additional prewarm policies. They are not claimed
as implemented, exhausted or validated. The existing RNG is a seeded CPU
`torch.Generator` and `torch.rand`; it has not been replaced.

## Fixed acceptance and limitations

Three AB/BA/AB repeats, C1/2/4/8/16/32/64, 128 timed requests per arm/cohort:
5376 requests and 2688 original PCM pairs. Primary C32 and C64 must each satisfy
`min(candidate throughput) / max(control throughput) > 1.02`, outside measured
A/A throughput noise, and per-arm primary spread at most 5%. Low concurrency
throughput ratio must be at least 0.97. Every cohort's conservative TTFA p95
ratio must be at most 1.05. No mean/geomean can override a failed cohort.

Waveforms must match literal original PCM bytes, with real EOS, correct format,
frame accounting and complete streaming bytes. Actual service/benchmark exits
must be zero, no forced cleanup; missing evidence stays INCONCLUSIVE. Source,
workload, model assets, packages, primary installed model and Codec sources and
container image are bound before/after. This does not hash every transitive
dependency. The 32-request seeds-17/43 C4/C16 regression is reused coverage,
not unseen evaluation or an absolute ASR/MOS quality claim.

Tests use opportunistically available HBM under shared conditions. Foreign
contexts are observed and disclosed, never terminated. Context counts do not
measure accelerator compute interference. Preserve all A/A observations and
report variance; do not select only favorable windows.

## Guidance provenance

Search/revision/evidence discipline follows
[InferRSI optimization-search-policy](https://github.com/InferMatrixLab/InferRSI/blob/8eb3034b52ec192de1e8b4dfd6c9dcf0551145a4/skills/optimization-search-policy/SKILL.md)
and [inference-benchmark-design](https://github.com/InferMatrixLab/InferRSI/blob/8eb3034b52ec192de1e8b4dfd6c9dcf0551145a4/skills/inference-benchmark-design/SKILL.md).
These are methodological sources, not local performance evidence. Search design
and frozen candidates are recorded in [RFC U005](https://github.com/InferMatrixLab/qwen3-tts-rsi-npu-engine/issues/6).

## Reproduction

Stage the unchanged baseline revision and the candidate in separate checkouts.
Use the same runtime, model assets, physical accelerator, source hashes and
benchmark corpus for both. Supply the model location and available card at
runtime; keep connection details and original logs outside Git. Choose a card
with enough observed free HBM and clean up only this experiment's processes.

The qualified server options, in addition to `--model '<MODEL_PATH>'`, are:

```text
--continuous --streaming --pack-qkv --max-batch-size 64
--codec-left-context 72 --codec-batch-size 4 --incremental-codec
--pack-mlp --native-kv-update --batch-prefill --prefill-per-tick 64
--text-only-rope --prewarm-codec --native-gqa --graph-embeddings
--graph-prefill --rms-norm native --fused-swiglu --native-rope
--graph-sampling --native-rope-scope predictor --triton-snake
--cache-codec-codebooks --triton-kv-update --triton-talker-rope
--prewarm-prefill --triton-cdf --bucket-prefill --direct-prompt
```

The candidate appends `--codec-streams=2 --packed-kv-load --single-request-views`. The baseline does
not append either flag. Bind the service to a loopback address, set the chosen
physical device through `ASCEND_RT_VISIBLE_DEVICES`, and use
`OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1` for both arms. The campaign also sets
`LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` inside its container.
The recorded command/runtime
identity is authoritative for this campaign; other combinations are unqualified.

For each arm and concurrency, use the existing benchmark CLI:

```sh
python -m qwen3_tts_engine.benchmark \
  --endpoint 'http://localhost:<SERVICE_PORT>/v1/audio/speech' \
  --workload '<MAIN_128_CORPUS>' \
  --output '<PRIVATE_OUTPUT>/<REPEAT>/<ARM>/<CONCURRENCY>' \
  --requests 128 --concurrency '<CONCURRENCY>' --warmup '<WARMUP>' \
  --engine '<ARM>' --source-revision '<EXACT_SOURCE_REVISION>' \
  --model 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice' --frame-rate 12.5
```

Here ARM is `control` or `candidate`, CONCURRENCY is 1/2/4/8/16/32/64, and
WARMUP is `max(2, CONCURRENCY)`. Use three independent service pairs in
AB/BA/AB order, waiting for readiness before each arm. Preserve warmup
boundaries, real EOS, all original streamed PCM and actual process exits. The
128-request main corpus is the baseline's versioned
`evidence/campaign-20261017/plan/workload-128.jsonl`, SHA-256
`5007716c83b29d0247b2c6291b9704c652e01a94362c19e90ea3cb34987060c7`.
It uses seed 42 and maximum 2048 new tokens. Do not replace it with the known
32-request regression corpus or tune generation limits to improve results.

Before formal comparisons, run the unchanged baseline against itself at C32
and C64, 128 timed requests per arm, with the same warmups. Keep all A/A
observations; the acceptance noise floor can only increase. After formal
qualification, check `benchmarks/u005-regression.jsonl` at C4/C16, 32 timed
requests per arm. This corpus reuses seeds 17/43 and is accuracy coverage,
not unseen quality evidence or another performance target.

The component checks accept only CLI configuration:

```sh
python -m tools.probe_codec_dual_state \
  --model '<MODEL_PATH>' \
  --control-codec '<BASELINE_SOURCE>/qwen3_tts_engine/codec_slots.py' \
  --output '<PRIVATE_OUTPUT>/operator.json'
python -m tools.probe_packed_kv \
  --model '<MODEL_PATH>' \
  --control-prefill '<BASELINE_SOURCE>/qwen3_tts_engine/prefill_graph.py' \
  --output '<PRIVATE_OUTPUT>/packed-operator.json'
python -m tools.probe_single_request_views \
  --model '<MODEL_PATH>' \
  --control-continuous '<FROZEN_PARENT>/qwen3_tts_engine/continuous.py' \
  --workload '<MAIN_128_CORPUS>' \
  --output '<PRIVATE_OUTPUT>/views-operator.json'
python -m unittest discover -s tests -v
```

Original-WAV collection and metadata replay are separate checks. The public
auditor reproduces the gates from redacted evidence and runs without torch,
inference engines or accelerator runtimes:

```sh
python -m tools.audit_u005_campaign \
  --evidence '<EVIDENCE_DIRECTORY>' --output '<PRIVATE_OUTPUT>/audit.json'
```

FROZEN_PARENT is the retained unsuccessful 103-file snapshot
`646520ace810ee85b77fa44a513b75e6f1adb85644e885ee375be83b50736630`,
including its complete source-manifest.json. Its continuous.py SHA-256 is
`402ff616a6108b0ab78c65d39a7698b747b41db508870348bb2cf38a86675b11`.
Resolve that source snapshot from the access-controlled campaign archive at
runtime. The operator verifies every parent file before import. This reference
isolates the input-preparation change; formal service acceptance still compares
the declared upstream baseline against the complete candidate.

Public JSON records are evidence, not a task or runtime configuration interface.
The access-controlled archive records original media, exact measured source,
controllers, raw commands, failed searches and logs; its digest binds that
private provenance. Public metadata alone cannot independently re-read media
that has not been supplied to the auditor.
