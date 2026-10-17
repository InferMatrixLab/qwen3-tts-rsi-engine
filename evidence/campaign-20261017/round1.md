# Exploratory first baseline sweep

Status: INCONCLUSIVE. This is one repetition, with independent quality scoring still running.

The [baseline charts](baseline-charts/README.md) provide PNG/SVG comparisons and
machine-readable inputs for all four metrics. They compare the two upstream
baselines only; they are not a standalone-engine speedup claim.

vLLM-Omni main `b164fea797531c1fa10f8b87829632cad006b9fc`, jemalloc environment workaround, one physical NPU 0, 128 fixed requests per concurrency. All 896 timed requests completed; benchmark and server exited 0.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 12.297 | 270.93 | 284.76 | 1.0210 |
| 2 | 22.943 | 324.36 | 375.00 | 1.0921 |
| 4 | 43.831 | 344.96 | 430.13 | 1.1361 |
| 8 | 79.491 | 391.49 | 520.53 | 1.2217 |
| 16 | 124.596 | 498.53 | 661.14 | 1.3816 |
| 32 | 200.265 | 709.22 | 1016.53 | 1.7185 |
| 64 | 242.077 | 1395.92 | 1793.43 | 2.6148 |

The host also ran separate-device implementation probes and ASR qualification during this exploratory sweep. Final acceptance requires isolated matched repetitions and completed quality gates.

The independent combined graph service completed a separate 8-request C1 qualification on physical NPU 1 at 22.016 frames/s, with server exit 0. This different sample/device cohort does not establish a speedup ratio. It still delivers a complete waveform in one audio event.

SGLang main plus the documented torch 2.10 type compatibility patch is now being scanned on physical NPU 0. No best-baseline or 2x acceptance conclusion is available yet.

## Continuous scheduling candidate

Commit `d8cb066d3812986a8e7474b71a8f86a05c935f47`, physical NPU 1, one repetition of the same 128 requests at each concurrency. All 896 requests completed and the server exited 0. Native RMSNorm, packed QKV, 72-frame codec context and codec batch size 4 were enabled. These are exploratory results on a different physical card, with other host work running; they do not establish a matched speedup or acceptance.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 27.357 | 152.85 | 158.88 | 0.4627 |
| 2 | 45.679 | 191.27 | 226.92 | 0.5498 |
| 4 | 68.932 | 229.26 | 401.77 | 0.7175 |
| 8 | 97.609 | 296.84 | 786.76 | 0.9956 |
| 16 | 123.621 | 429.92 | 1488.35 | 1.5482 |
| 32 | 154.422 | 697.16 | 3203.42 | 2.3671 |
| 64 | 180.846 | 1215.15 | 6874.90 | 3.8451 |

The all-concurrency 2x objective remains unachieved. Full-corpus candidate quality is still being scored. SGLang C1 ASR completed with English WER 0.011380 and Chinese CER 0.026490; speaker similarity and the remaining cohorts are pending.

Candidate C1 ASR is complete: English WER 0.009957 and Chinese CER 0.049669. The Chinese result exceeds the frozen minimum-baseline-plus-0.01 gate (SGLang C1: 0.026490). This candidate therefore fails the current C1 ASR gate. Traditional/simplified transcription differences contribute to the discrepancy; raw transcripts and the original scoring are retained, and no normalization change or relaxed threshold is used to claim acceptance. Speaker scoring remains pending.

## SGLang full sweep

Pinned main plus the recorded torch type compatibility patch and jemalloc workaround, physical NPU 0. All 896 timed requests completed; server exit 0. This remains an exploratory single repetition; quality scoring for remaining cohorts is running.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 4.275 | 424.17 | 565.99 | 2.9568 |
| 2 | 7.917 | 601.96 | 793.14 | 3.1687 |
| 4 | 14.116 | 658.99 | 986.29 | 3.4984 |
| 8 | 25.026 | 689.92 | 1267.35 | 3.9222 |
| 16 | 38.304 | 762.42 | 1739.05 | 4.7895 |
| 32 | 57.102 | 811.75 | 2462.35 | 6.2362 |
| 64 | 66.607 | 3451.91 | 7305.23 | 10.1708 |

## Baseline speaker similarity

All 128 retained clips at every concurrency were scored with the same pinned WavLM verifier and independent same-preset anchors. Values below are raw mean cosine similarity, not percentages. ASR scoring of the remaining SGLang cohorts is still in progress. This is not a subjective naturalness evaluation.

| Concurrency | vLLM mean cosine | SGLang mean cosine |
|---:|---:|---:|
| 1 | 0.770878 | 0.771980 |
| 2 | 0.768591 | 0.765452 |
| 4 | 0.771503 | 0.767026 |
| 8 | 0.775485 | 0.766908 |
| 16 | 0.771416 | 0.769462 |
| 32 | 0.768243 | 0.777898 |
| 64 | 0.772615 | 0.771402 |

## Incremental codec candidate

Commit `2eeddd4db6e662def610143ac2a1b6dd43d606a5`, physical NPU 1, one exploratory repetition of 128 requests per concurrency. All 896 timed requests completed; server exit 0. Full-corpus candidate quality has not yet been evaluated. The all-concurrency 2x objective is not achieved.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 29.493 | 153.36 | 165.50 | 0.4274 |
| 2 | 48.771 | 190.04 | 215.26 | 0.5167 |
| 4 | 77.129 | 218.73 | 348.53 | 0.6420 |
| 8 | 110.056 | 265.05 | 685.88 | 0.8838 |
| 16 | 154.755 | 386.26 | 1474.12 | 1.2271 |
| 32 | 198.679 | 548.60 | 3177.24 | 1.8526 |
| 64 | 231.937 | 1059.13 | 6859.43 | 3.1055 |

## Baseline ASR quality

All 128 retained clips at each concurrency have now been independently transcribed with the same pinned Whisper assets and float32 NPU evaluation. English uses WER; Chinese uses CER. Raw transcripts, including script and number-format mismatches, are retained without post-hoc normalization changes.

| Concurrency | vLLM EN WER | vLLM ZH CER | SGLang EN WER | SGLang ZH CER |
|---:|---:|---:|---:|---:|
| 1 | 0.008535 | 0.037252 | 0.011380 | 0.026490 |
| 2 | 0.011380 | 0.034768 | 0.014225 | 0.038907 |
| 4 | 0.005690 | 0.030629 | 0.012802 | 0.033940 |
| 8 | 0.008535 | 0.027318 | 0.008535 | 0.033940 |
| 16 | 0.036984 | 0.025662 | 0.014225 | 0.027318 |
| 32 | 0.009957 | 0.036424 | 0.009957 | 0.028974 |
| 64 | 0.011380 | 0.035596 | 0.007112 | 0.031457 |

Incremental candidate C1 ASR completed with EN WER 0.009957 and ZH CER 0.049669, the same aggregate errors as the previous continuous candidate. It still fails the frozen Chinese ASR noninferiority gate; the codec optimization does not resolve that quality failure.

## Reference RMSNorm and larger admission budget controls

The original RMSNorm C1 control (commit `37538a4`, physical NPU 6) completed 128 requests at 27.929 frames/s, with EN WER 0.011380, ZH CER 0.028974, and speaker mean cosine 0.780207. These C1 quality values satisfy the frozen tolerances for this exploratory cohort. They do not establish quality at other concurrency levels or a formal repeated comparison.

Increasing the admission budget from 4 to 64 requests per tick produced the following single-cohort results; all 128 requests in each row completed, with benchmark and server exit 0. High-concurrency quality remains pending. Native RMSNorm uses physical NPU 1, revision `4812630`; reference RMSNorm uses physical NPU 0, revision `a693aea`. The different cards and generated outputs preclude attributing the entire difference to normalization.

| Normalization | Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---|---:|---:|---:|---:|---:|
| Native | 16 | 175.592 | 324.56 | 493.67 | 1.0839 |
| Native | 64 | 312.458 | 1004.64 | 1275.59 | 2.2162 |
| Reference | 16 | 141.786 | 373.01 | 683.89 | 1.3553 |
| Reference | 64 | 250.558 | 1306.19 | 1691.99 | 2.6788 |

The C64 target is approximately 484.15 frames/s; neither control reaches it. No all-concurrency 2x result is established.

Synchronized diagnostic ranges (not service benchmarks) for 16 C32 ticks with original RMSNorm attribute 1.650 seconds inclusively to audio generation, 0.563 to predictor, and 0.549 to talker, within 3.662 seconds total. Codec batching of 8 instead of 4 changed total diagnostic time to 3.717 seconds and audio to 1.638 seconds; this probe provides no convincing gain. Synchronization perturbs execution, nested ranges overlap, and these values must not be converted into service throughput.

## Text rotary and codec prewarming

Revision `49e46f8`, physical NPU 6, one exploratory repetition: C16 149.230 frames/s and C64 269.739 frames/s, all 256 requests successful, benchmark/server exit 0. Every retained PCM hash equals the corresponding reference-RMS admission-budget-64 control, including both concurrency cohorts. This is exact output evidence, not a formal matched performance comparison.

The reference control's full ASR evaluation reports C16 EN WER 0.009957 / ZH CER 0.036424 and C64 EN WER 0.009957 / ZH CER 0.051325. Both Chinese scores fail the frozen matched-concurrency gates (0.035662 and 0.041457 respectively). The identical-output optimized candidate inherits these failures; no quality improvement is claimed. Speaker evaluation is running. The all-concurrency 2x objective remains unmet.

Reference-RMS high-concurrency speaker evaluation completed: C16 mean cosine 0.771005 and C64 0.761468, each within the frozen speaker margin. The Chinese ASR failures above still reject the candidate; passing speaker similarity does not override them.

## Attention experiments

Single exploratory C16/C64 cohorts completed with 128 requests each and clean server exits. Bucketed KV (revision `f0facb9`, physical NPU 6) reached 142.434 / 270.120 frames/s. Native GQA (revision `02f100e`, physical NPU 1) reached 155.531 / 288.384 frames/s. Both retain original RMSNorm, packed QKV/MLP, incremental prewarmed codec, and text-only rotary. These are different-card exploratory results, not matched repeated attribution. Neither reaches the 2x target. Both alter numerical behavior; GQA full ASR evaluation is running and neither has accepted full quality evidence.

## Predictor embedding capture and prefill graph experiment

Revision `4ae4550`, physical NPU 6: C16 191.955 frames/s, C64 330.616 frames/s, 128/128 successful requests at each concurrency, benchmark/server exit 0. This combines native GQA, original RMSNorm, text rotary, packed QKV/MLP, captured subcode embeddings, graph prefill, and prewarmed incremental codec. Quality has not yet been evaluated; numerical changes from GQA and prefill padding preclude inheriting earlier quality scores. The 2x target remains unmet.

A separate synchronized C32 diagnostic with GQA, captured embeddings and prewarmed codec (eager batched prefill) measured 2.098 seconds across 16 ticks: batch prefill 0.667 seconds, predictor 0.559, talker 0.253, audio 0.478 (codec nested within audio 0.446). These perturbed inclusive diagnostic ranges are not service measurements.

## Native fusion candidate and reference-rounding normalization control

Single exploratory C16/C64 cohorts, each 128 requests and clean server exit: the graph-prefill candidate with native RMSNorm and fused SwiGLU (`b18450b`, physical NPU 6) reached **271.774 / 444.433 frames/s**. C16 exceeds the numerical 2x throughput target in this single cohort; C64 does not, and no quality or formal repeated comparison is yet accepted.

The separate native-GQA/eager-prefill control with FP32 fused normalization followed by BF16 rounding and weight multiplication (`acd2685`, physical NPU 0) reached 187.962 / 328.034 frames/s. Its ASR evaluation is running. This is not an isolated ablation against the preceding graph-prefill candidate.

The earlier original-RMS native-GQA candidate completed ASR: C16 EN WER 0.007112, ZH CER 0.023179; C64 EN WER 0.011380, ZH CER 0.031457. Both cohorts meet the frozen ASR margins. Speaker similarity is still pending, and its throughput remains below 2x.

Native-GQA speaker evaluation completed: C16 mean cosine 0.772866 and C64 0.774282, both within the frozen margins. Together with its ASR results this passes those two exploratory quality cohorts, but its throughput is below target.

The faster native-fusion candidate's C16 ASR is now complete: EN WER 0.009957 and ZH CER 0.025662, both within the frozen margins. Speaker evaluation and C64 ASR remain pending. The reference-rounding control's C16 CER is 0.042219, exceeding its 0.035662 gate (EN WER 0.008535 passes); it is not accepted as a quality-preserving alternative.

Power-of-two prefill batch padding additionally passed a complete BF16 continuous three-request pair (indices 0, 64, 65): all PCM hashes and frame counts matched the unbucketed graph-prefill control, with EOS completion and clean process exit. This narrow identity check does not establish all-batch or full-service quality.

Native-fusion C64 ASR completed: EN WER 0.008535 and ZH CER 0.032285, both within the frozen C64 margins. Its speaker evaluation is running. The reference-rounding control also passes C64 ASR (EN 0.009957, ZH 0.029801), but retains its C16 quality failure.

Adding prefill batch bucketing to native fusion (`ac82d89`, physical NPU 1) measured C16 272.217 and C64 434.461 frames/s, with all requests and clean lifecycle passing. It provides no clear speed benefit over the prior exploratory cohort. All 128 C64 PCM hashes match that prior candidate; only 9 of 128 C16 hashes match, so C16 quality cannot be inherited. It is not selected on the available performance evidence.

Native-fusion speaker scoring completed: mean cosine C16 0.781158 and C64 0.779979, both within the frozen margins. These two exploratory cohorts now pass ASR, speaker, request-completion and clean-lifecycle gates. They do not establish all-concurrency quality or a formal 2x result; C64 throughput remains below target.

## Remaining native-fusion concurrency sweep

Revision `ac82d89`, physical NPU 7, 128 requests per concurrency, all successful and server exit 0. This is another exploratory card/cohort, not a combined formal seven-concurrency result.

| Concurrency | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | Mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 49.874 | 64.58 | 66.33 | 0.2522 |
| 2 | 79.163 | 87.23 | 99.01 | 0.3162 |
| 4 | 122.737 | 110.28 | 151.87 | 0.4046 |
| 8 | 187.095 | 142.98 | 214.46 | 0.5194 |
| 32 | 357.207 | 284.03 | 1111.34 | 1.0462 |

C1 ASR EN WER 0.008535 passes, but ZH CER 0.045530 exceeds its 0.036490 gate. Thus the globally fused configuration is rejected as an all-concurrency solution despite its low-concurrency speed. Other cohorts' ASR scoring is running.

Decode bucket step 8 (`5887ab7`, physical NPU 1) measured C32 336.511 / C64 467.910 frames/s. Prewarmed codec batch 8 (`47a0ad0`, physical NPU 6) measured C32 320.247 / C64 476.856 frames/s. Both completed all requests and clean server exits. Their quality is unverified, and neither meets both high-concurrency throughput targets. Gains from these separate experiments must not be multiplied or combined as measured evidence.

Batching codec host transfers (`64d8505`, physical NPU 6) completed C32/C64 at 338.107 / 463.065 frames/s, with 128 successful requests per cohort and clean server exit. Four standalone probe outputs matched PCM exactly, but this remains an exploratory service comparison; it does not establish an all-concurrency improvement or achieve the target.

## Exact sampling graphs and single-request reference dispatch

Sampling graphs (`51570b7`, physical NPU 1) completed C32/C64 at 344.970 / 482.417 frames/s, with 128 successful requests per cohort and clean lifecycle. Neither exceeds its 2x target.

The first single-request reference-dispatch implementation (`0ecd620`, physical NPU 7) completed C1/C2/C32/C64 at 33.204 / 71.225 / 280.326 / 413.612 frames/s. All 128 C1 PCM hashes match the previously scored original-RMS C1 control, so that cohort's ASR/speaker evidence transfers by exact audio identity. Its C2 ASR is running. The implementation also routed single-arrival refills at high concurrency through slow reference prefill; the subsequent fix restricts that behavior to a genuinely solitary active request. The fixed dispatch still needs service validation.

The native-RoPE + sampling-graph + reference-single combination (`9942ca8`, physical NPU 1) completed C32/C64 at 345.017 / 468.613 frames/s with clean lifecycle. It does not meet either target; quality has not been evaluated for those cohorts. A no-reference-dispatch control is required to separate rotary improvement from the slow single-row tail path.

The earlier reference-single C2 control's ASR completed: EN WER 0.007112 passes, ZH CER 0.049669 fails its 0.044768 gate. The globally fused native candidate's full remaining ASR also completed: C2 CER 0.045530 fails, C4 CER 0.040563 passes its 0.040629 gate narrowly, C8 CER 0.045530 fails, and C32 CER 0.025662 passes. English WER passes in all these cohorts. This configuration still fails all-concurrency quality acceptance.

The no-reference native-RoPE/sampling control (`3ef961a`, physical NPU 1) completed C32/C64 at 352.083 / 468.490 frames/s with clean lifecycle. It still misses both throughput targets. A separate complete C1..C64 sweep on physical NPU 0 and full ASR evaluation are running.

The installed TorchAir backend was discovered inside `torch_npu.dynamo.torchair` (import `torch_npu` before `torchair`); no package installation was required. A synthetic full-talker B32 decode probe compiled with `fullgraph=True`, `dynamic=False` and default configuration, taking 63.61 seconds for its first call. Steady ABBA means were approximately 9.37/39.57/39.51/9.39 ms for NPUGraph/TorchAir/TorchAir/NPUGraph. Hidden-state cosine was 0.946223 with max absolute error 28.5625. This candidate is rejected for integration on current evidence. The synthetic cache distribution is not speech-quality evidence, and this initial probe does not verify persistent KV mutation semantics; real-prefix and intermediate-state checks are required before any compiler acceptance.


### Native rotary and sampling graph full sweep (3ef961a, physical NPU 0)

Exploratory single repeat: all 896 requests succeeded; benchmark and server exited 0. Quality is still incomplete and this is not formal 2x acceptance.

| Concurrency | Frames/s | TTFA p50 ms | TTFA p95 ms | Mean RTF |
|---|---|---|---|---|
| 1 | 56.608 | 59.365 | 63.027 | 0.22216 |
| 2 | 85.702 | 85.842 | 90.831 | 0.29296 |
| 4 | 135.000 | 99.936 | 140.369 | 0.36819 |
| 8 | 207.879 | 130.526 | 199.618 | 0.47186 |
| 16 | 295.962 | 173.535 | 360.882 | 0.64319 |
| 32 | 386.540 | 243.571 | 1021.944 | 0.95677 |
| 64 | 473.183 | 861.812 | 1259.486 | 1.39269 |

C32 (386.540 frames/s) and C64 (473.183 frames/s) remain below their matched target thresholds (approximately 400.53 and 484.15). C1 full ASR: EN WER 0.0142247511, ZH CER 0.0331125828, within the frozen ASR gates. Speaker similarity remains pending; other concurrency ASR assessments continue. A separate same-device codec-batch-8 control is running; its results cannot be presumed from previous codec experiments.


### Codec batch 8 and fixed predictor rotary follow-up

Same physical NPU 0 codec-batch-8 control (source b73c8ea): C32 353.970987 frames/s, C64 486.154755 frames/s, 256/256 requests, server exit 0. C64 exceeds the exploratory 2x threshold by less than 0.5%; C32 regresses. Full quality and independent repeats are absent, so this is not an accepted all-concurrency configuration.

Native rotary full-sweep C2 ASR completed: EN WER 0.0142247511 and ZH CER 0.0331125828, within frozen ASR gates. Speaker scoring is running separately.

Fixed predictor rotary hoisting probe on physical NPU 6 uses synthetic hidden inputs and identical per-step uniforms; it is not an end-to-end quality test. All four batches preserve both subcodes and combined embeddings exactly. Synchronized ABBA means:

| Batch | Original ms | Cached ms |
|---|---|---|
| 1 | 9.1576 | 8.9261 |
| 8 | 13.3747 | 13.3092 |
| 32 | 17.5587 | 17.4093 |
| 64 | 21.6366 | 21.4195 |

This saves only approximately 0.06–0.23 ms per predictor call. It remains a probe, not integrated into serving; no service-speedup claim follows from these component timings.


### Optional residual-add RMSNorm fusion

Added a default-off `--fused-residual` experiment requiring native RMSNorm. It fuses the post-attention residual add and normalization in both talker and predictor decoder layers. Attention/cache arguments and optional attention outputs are preserved; single-reference mode bypasses the fused decoder. The operator probe covers hidden widths 1024/2048 and decode/prefill shapes. All retained residuals are bit-identical; normalization differs by up to 0.03125 in BF16. B64 width2048 replay approximately 0.0241 ms separate versus 0.0181 ms fused. CPU FP32 errors and ABBA samples are retained. These are component diagnostics, not audio-quality acceptance. All 35 unit tests passed; full service measurement and speech quality remain pending.


Residual-fusion service probe (72176a0, physical NPU 6) completed C32/C64 with 256/256 successful requests and server exit 0: 342.220967 / 469.690772 frames/s. It remains below both 2x targets. A same-device, same-runtime-code control with fusion disabled is running to distinguish component attribution from shared-host/device variation; audio quality for the fused variant is not yet evaluated.


The residual-fusion same-device control completed: C32 352.082548 and C64 457.430306 frames/s, 256 successful requests, server exit 0. The fused arm was approximately 2.8% slower at C32 and 2.7% faster at C64 in this single sequential comparison. It is not promoted.

A new default-off `--cache-codec-constants` experiment hoists SnakeBeta alpha exponentials and beta inverse scales out of graph replay. It preserves operation ordering and requires fixed evaluation weights/device/dtype. All 38 unit tests passed. The strict-FP32 101-frame codec test passed across two interleaved groups, live-state prewarming, move and reset (maximum error versus eager 9.54e-6); direct BF16 graph-control equality and service throughput are being measured separately.


Codec constant cache BF16 graph-control test completed with exact waveform equality for all 101-frame chunks and slot moves/resets. Live-state prewarming was exact. The separate BF16 graph-versus-eager comparison still differs (maximum 0.128784), so graph-control equality must not be described as eager parity or speech-quality acceptance.

Native-RoPE full-sweep quality update: C8 passes both ASR and speaker gates; all seven speaker cohorts pass. C16 EN WER is 0.0042674253, but ZH CER 0.0380794702 exceeds its frozen 0.0356622517 limit. Consequently this configuration is not accepted at all concurrencies even if later throughput optimization succeeds. C32/C64 ASR remains pending.


Codec-constant service measurement (d769981, physical NPU 0): C32 361.892265 and C64 469.078374 frames/s, 256/256 successful requests, clean server exit. This does not demonstrate a gain against the earlier same-device native-RoPE sweep and remains below both target thresholds. The flag stays default-off. A second full 128-request-per-concurrency vLLM-Omni baseline repeat has started on physical NPU 0 with the same pinned clean main and allocator workaround, to extend repeat evidence and check temporal variability; it is not yet complete.


### Bounded codec code history and new component profiling

Added default-off `--codec-code-buffer`: one batched write per tick retains the last eight code frames per slot; grouped audio decode gathers the required interval directly. Slot moves accompany cancellation/completion compaction. All 43 tests passed; four complete mixed-language requests matched the list-based control PCM hashes exactly, including normal slot compaction. Explicit cancellation parity and full service measurement remain pending.

The latest synchronized C32 diagnostic measured 16 ticks: 1.3575 s total, 0.3621 s batch prefill, 0.2879 s predictor, 0.1536 s talker, 0.4731 s audio (including 0.4380 s codec). Inclusive ranges overlap, synchronization perturbs execution, and the workspace was under development; these are bottleneck clues, not throughput or attribution evidence.

Triton Ascend 3.2.0 is available. A vector-only SnakeBeta prototype preserves explicit BF16 rounding and successfully captures into NPUGraph. The first 1D logical-grid implementation was slower than separate native operators; raw errors and timings are retained. A physical-core-capped, channel/time-tiled follow-up is running before any serving integration.


Native-RoPE/sampling full-sweep quality is now complete: all 896 clips have ASR and speaker scores. C1/C2/C4/C8/C32/C64 pass the frozen ASR and speaker gates. C16 fails Chinese CER (0.03807947 versus maximum 0.03566225); all speaker gates pass. C32 EN WER/ZH CER: 0.00853485 / 0.03062914; C64: 0.00995733 / 0.02649007. This is a rejected all-concurrency candidate, not a quality-qualified 2x result.


The codec code-buffer cancellation control passed: cancelling request 0 after frame 3 left requests 1/64/65 with identical complete PCM hashes. The 101-frame real decoder test with Triton SnakeBeta also passed exact graph-control equality across interleaved groups, slot move and reset; BF16 graph-versus-eager differences remain unchanged and are not described as eager parity. Service throughput is the next gate.

Added default-all `--native-rope-scope {all,talker,predictor}` for component attribution of the C16 ASR regression. It changes no sampling parameters or quality thresholds. The predictor-only C16 control is prepared; it is not an accepted numerical policy.


Predictor-only native RoPE C16 service control (a96f3f1, physical NPU 1) completed 128 requests, 283.171092 frames/s, server exit 0. This exceeds the exploratory C16 throughput threshold but full ASR is running; no quality acceptance is claimed. Triton codec serving remains in first-time compilation/prewarming. The codec probe now records actual Triton invocation shapes/count and rejects a BF16 run that never reaches the kernel, to distinguish graph equality from an unexercised fallback.
