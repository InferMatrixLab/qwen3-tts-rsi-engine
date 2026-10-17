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


Predictor-only RoPE C16 full ASR completed: EN WER 0.0099573257, ZH CER 0.0223509934, both within frozen gates. Speaker scoring remains pending. The all-RoPE candidate comparison now has reproducible PNG/SVG charts and machine-readable reports for all four metrics under `native-rope-charts/`; every chart is marked FAIL because that older candidate fails C16 quality, with n=1 and shared-host limitations explicit.

Added default-off `--cache-codec-codebooks`: precompute the sixteen fixed embedding_sum / clamp(cluster_usage) tables without changing codebook accumulation order or mutating the reference quantizer. CPU FP32/BF16 tests cover negative, zero and tiny usage counts and partial/full codebook inputs; all 49 tests pass. The NPU 101-frame graph-control test is running before any service claim.


Cached normalized codebooks passed the real NPU BF16 graph control: all 101-frame chunks and reset output are bit-identical to the uncached quantizer, including interleaved streams and slot compaction. Prewarming all 32 codec shapes preserved live state exactly. This is graph parity only; service throughput remains unmeasured for this flag.

Predictor-only RoPE C16 speaker scoring completed on 128 clips: mean cosine 0.7649079487, above the frozen minimum 0.751416. Thus this single C16 cohort passes ASR/speaker gates and exceeds its exploratory throughput target; no other concurrency or repeat is implied.


### Separate codec optimization service results

Triton SnakeBeta (a96f3f1, physical NPU 6): C16 303.381, C32 392.329, C64 489.908 frames/s. Cached codebooks (ac540ca, physical NPU 1): C16 286.291, C32 387.506, C64 485.374 frames/s. Each run completed 384/384 requests and server exit 0. These are separate, single-repeat exploratory configurations without their own full quality assessments; C32 still misses even the original approximately 400.53 threshold. Their gains must not be multiplied. A direct combined experiment with predictor-only RoPE is prepared for all seven concurrencies.

Added `tools/audit_cohort.py` to recompute timings from retained requests, hash actual WAV PCM, check chunk totals/terminal events and match quality sample hashes to those exact waveforms. ASR edit distances and aggregate speaker/ASR scores are independently recomputed from retained records. Five corruption/regression tests pass; C16 baseline vLLM, SGLang and predictor-only candidate integrity audits all pass. This audit deliberately does not claim hardware comparability, lifecycle, repeat independence, quality noninferiority or 2x acceptance.


All 21 first-repeat baseline/native-RoPE cohorts passed retained-audio integrity audits (2688 WAVs): metrics were recomputed, actual PCM digests matched request and quality records, and ASR edit/speaker aggregates matched. This does not override the known C16 quality rejection.

Second vLLM baseline sweep completed on physical NPU 0: 896/896 requests, server exit 0; C1/2/4/8/16/32/64 throughput 12.791 / 24.712 / 47.193 / 82.576 / 137.542 / 214.294 / 232.600 frames/s. Full repeat-2 quality is pending. C32 exceeds the first-repeat throughput, emphasizing that the final threshold cannot be selected from the lowest observed baseline. SGLang repeat 2 has started on the same device with the previously documented import compatibility patch.

Added opt-in `--prefill-interval 4` scheduling experiment: while decoding active requests, admit queued arrivals every fourth tick; idle admission remains immediate and cancellation is processed every tick. A controlled async test verifies deferred admission, prompt cancellation and balanced queue accounting. Existing server tests and integrity tests pass (9 tests). This may increase TTFA; no service speedup is assumed.


Direct combined candidate (a9fac06, physical NPU 6: predictor-only native RoPE + Triton SnakeBeta + cached codebooks) completed all 896 requests with clean server exit. C1/2/4/8/16/32/64 throughput: 55.038 / 85.701 / 135.796 / 208.199 / 313.009 / 394.717 / 491.399 frames/s. C32 still misses the original 400.53 threshold, and the second vLLM repeat strengthens that concern. C1 ASR passes (EN WER 0.01137980, ZH CER 0.03145695); full remaining quality is in progress. The same-device prefill-interval-4 control is next, with TTFA tradeoffs reported rather than hidden.

### Combined candidate: prefill interval 4 control

Source `8b8bd08`, physical NPU 6, one exploratory trial. C16/C32/C64 throughput: 310.746 / 349.326 / 506.764 frames/s; TTFA p95: 419.821 / 1024.928 / 1264.018 ms. All 384 requests completed and server exited 0. Quality was not evaluated. C32 regressed versus the interval-1 combined candidate (394.717 frames/s), so interval 4 is not selected. C64 improvement alone does not meet the all-concurrency objective.

### Combined candidate: codec ring buffer control

Source `4613082`, physical NPU 6, one exploratory trial: C32 369.961 and C64 493.910 frames/s; all 256 requests completed, server exit 0. Compared with the combined candidate without this buffer (394.717 / 491.399), C32 regressed and C64 was nearly unchanged. Full quality was not evaluated; this variant is not selected.

### Fixed-capacity predictor attention diagnostic

Sixteen synthetic BF16 cases (B1/8/32/64, valid KV length 2/3/8/16, head dimension 64) passed with exact output equality across three graph replays, including poisoned unused KV positions. Process exited 0. Explicit valid lengths avoided prefix contiguous copies; B32 attention measured approximately 0.056–0.057 ms versus 0.057–0.064 ms for sliced KV. This is only an operator diagnostic: DynamicCache updates, model subcodes, audio quality and service speed remain untested.

### Fixed-capacity predictor KV: model diagnostic rejected

Real weights with synthetic embeddings and identical random draws: B1/8/32/64 produced exactly equal subcodes and combined embeddings, including three changed-input graph replays. ABBA mean times (dynamic / fixed, ms): approximately 9.36 / 10.54, 13.60 / 14.82, 17.65 / 18.65, 21.56 / 22.46. Process exit 0. The fixed buffer implementation is slower at every batch and is not integrated into serving. Attention-only savings did not survive the full cache-update path.

### Bounded prefill prewarming

Source `5dd348a`, physical NPU 6, one trial. C32/C64: 392.226 / 500.101 frames/s; TTFA p95 1068.516 / 1269.826 ms. All 256 requests completed and server exited 0. All 256 WAVs are byte-identical to the combined candidate without prewarming. C32 did not improve; C64 showed a modest exploratory increase. This does not establish an all-concurrency or repeated-trial gain.

### Combined candidate: complete first-trial quality

All 896 waveforms now have independently computed ASR and speaker scores. Every C1/2/4/8/16/32/64 cohort passes the frozen ASR and speaker noninferiority gates against both round-1 baselines. Every cohort also passes retained waveform/hash/timing/quality consistency audit. C16 speaker scoring was rerun independently in this complete sweep; it agrees with the earlier identity-linked evidence. Both evaluation loops exited 0. This establishes first-trial quality only: the C32 throughput shortfall, matched repeated trials, and shared-host comparability remain unresolved.

|Concurrency|EN WER|ZH CER|Speaker mean cosine|
|---|---|---|---|
|1|0.011380|0.031457|0.775129|
|2|0.011380|0.031457|0.774560|
|4|0.011380|0.029801|0.772809|
|8|0.012802|0.031457|0.775390|
|16|0.009957|0.022351|0.764908|
|32|0.008535|0.031457|0.767611|
|64|0.009957|0.027318|0.771295|

### vLLM baseline round 2: complete quality

All 896 retained WAVs have ASR and speaker scores; all seven cohort integrity audits pass, and the evaluation loop exited 0. This is the second baseline repetition, not a completed three-repeat comparison.

|Concurrency|EN WER|ZH CER|Speaker mean cosine|
|---|---|---|---|
|1|0.008535|0.037252|0.770878|
|2|0.011380|0.034768|0.767873|
|4|0.005690|0.042219|0.773062|
|8|0.008535|0.030629|0.773518|
|16|0.036984|0.032285|0.772258|
|32|0.008535|0.041391|0.773680|
|64|0.011380|0.037252|0.772542|

### Combined candidate: codec batch 8 and KV bucket controls

Codec batch 8 (source `08f29a0`, physical NPU 6): C32/C64 364.465 / 504.726 frames/s. KV context bucketing (source `0053a64`, physical NPU 1): 345.160 / 458.975 frames/s. Each completed 256 requests with server exit 0. Both regress at C32 relative to the current combined candidate; neither is selected. Full quality was not run for these rejected variants. The KV comparison also spans different physical cards on the shared host.

### Paired KV service control

Source `c8480c7`, physical NPU 6: C32/C64 373.151 / 495.822 frames/s, all 256 requests completed, server exit 0. C32 regresses relative to the combined candidate (394.717), so this configuration is not selected. Operator speedup does not establish service speedup. Separate cancellation control produced exactly identical survivor audio after cancelling request 0 at frame 3.

### Paired KV: adjacent same-device control qualifies the earlier comparison

The subsequent flag-off control on physical NPU 6 produced C32/C64 364.234 / 486.940 frames/s, all 256 requests successful and server exit 0. Paired KV (373.151 / 495.822) is therefore approximately 2.45% / 1.82% faster than this adjacent control. Its lower value versus the earlier 394.717 result cannot establish a causal regression: temporal variability is material. Keep this candidate experimental pending repeated same-device attribution and full target acceptance. Do not multiply this gain with isolated operator gains.

### Sixteen-frame codec state validation

FP32 101-frame streams passed tolerance (maximum absolute error 1.43e-5). BF16 Triton Snake plus cached codebooks is exactly equal to the unoptimized graph control for the same chunk sequence; prewarming preserves live state, and move/reset checks pass. The probe observed 7424 actual Triton calls. BF16 graph-versus-eager batch-2 maximum error is 0.09644, so this is not an eager-parity or full-speech-quality claim. Both probes exited 0. Full service throughput and quality remain to be measured.


### 16-frame codec and attention length diagnostics

The 16-frame codec service completed 256 requests with server exit 0 on physical NPU 6. C32: 376.5229 frames/s, TTFA p50 244.9299 ms, p95 1395.4177 ms, mean RTF 0.977677. C64: 508.8541 frames/s, TTFA p50 995.7474 ms, p95 1423.1247 ms, mean RTF 1.320684. Full speech quality has not been evaluated; this configuration is not selected.

The full-arena FIA valid-length diagnostic completed 16 synthetic operator cases with exit 0. All passed the stated operator tolerance; maximum absolute difference was 0.001953125, so this is not bitwise equivalence. C32 length 128 measured 0.0648443 ms without a valid-length bound and 0.0585291 ms with it. These isolated timings do not establish service acceleration or speech quality. No runtime integration is enabled. See results/attention-context.json and results/continuous-codec-chunk16-status.json.


### Valid-length KV integration (experimental)

Added default-off `--valid-length-kv`, requiring continuous native GQA without reference-single. It buckets the attention bound but keeps full contiguous KV arenas and full-capacity causal masks; FIA receives the bound directly, avoiding prefix materialization. Four cache-history/attention cases plus four server tests passed (8 tests total). A real-model four-request probe completed with slot reuse, capacity growth from 64 to 128, and clean exit 0; matched control and service measurements are pending. Operator tolerance is not a speech-quality pass.

The vLLM third performance repeat completed all 896 requests with server exit 0 on physical NPU 1 (earlier repeats used physical NPU 0). C1/2/4/8/16/32/64 throughput: 13.2014 / 25.4881 / 48.9262 / 89.1285 / 145.9159 / 210.7236 / 251.6414 frames/s. Full audio quality evaluation is running; hardware differences and repeat variance prevent selecting a convenient baseline for acceptance.

Pointwise-convolution versus linear diagnostics completed with real weights and synthetic activations. Results vary by shape and include numerical differences; no runtime replacement is selected. See results/codec-pointwise.json.


The valid-length KV probe and matched flag-off control both exited 0; all four retained WAV files are byte-identical (indices 0, 1, 64, 65). This narrow identity check does not replace full-cohort quality. C32/C64 service validation is now running in `runs/continuous-valid-length-kv-c32-c64-retry`; an earlier launch stopped before server startup because Git safe-directory lookup produced an empty revision, with its log retained. SGLang second performance repeat completed with exit 0; the third repeat uses the same physical NPU 0 and pinned main plus import compatibility patch.


### Valid-length service and repeated-baseline progress

Valid-length C32/C64 completed 256 requests with clean server exit 0: 346.3942 / 469.6754 frames/s. Only 13/128 C32 and 27/128 C64 WAVs matched the earlier combined candidate exactly, so its quality evidence cannot transfer. A directly adjacent flag-off control is running on the same physical NPU 6. The full current test suite passed 57 tests with exit 0. SGLang round-2 C8/C16/C32 and vLLM round-3 C1 passed retained-audio/timing/quality-provenance integrity audits; this is separate from noninferiority and throughput acceptance.

`results/vllm-three-repeat-descriptive.json` preserves all four metrics per trial with sample standard deviations. It explicitly records the physical-card change and incomplete quality. TTFA p95 values are cohort percentiles, not pooled percentiles. A combined exact-transform candidate plan is prepared for direct service measurement; isolated gains will not be multiplied.


The adjacent valid-length flag-off control completed all 256 requests and exited 0: C32 369.5605 frames/s, C64 490.8549 frames/s. Candidate/control ratios are 0.93731 and 0.95685, respectively. This one adjacent pair does not establish repeated causal attribution, but provides no reason to select the option; it remains off. The exact-transform combination (paired KV writes, exact talker rotary, prefill prewarming) is now measured directly in `runs/continuous-exact-combined-c32-c64`, with no addition or multiplication of isolated gains.


### Direct exact-transform combination

Paired KV writes + exact talker rotary + prefill prewarming completed 256 service requests on physical NPU 6 with server exit 0. C32: 418.04745 frames/s; C64: 526.22459 frames/s. All 128 WAV files at each concurrency are byte-identical to the full-quality combined candidate. This supports explicitly identified quality transfer, not an independent scoring rerun. These are direct combination measurements, not compounded isolated gains. The adjacent flag-off control was 369.56054 / 490.85491 frames/s. No repeated-trial 2x claim is made; C32 still falls short of twice vLLM round 2, and physical-card/repetition matching remains outstanding.


### Second-repeat quality and next full candidate

SGLang round 2 completed full ASR and speaker evaluation of all 896 retained waveforms, quality loop exit 0; all seven cohort integrity audits passed. The previous full combined candidate passes EN WER, ZH CER and speaker margins against both round-2 baselines at every concurrency (summary-level gate, not repeated performance acceptance). vLLM round-3 C2 also passed its integrity audit.

The output-head/sampling diagnostic completed with exact sampled tokens and seen masks for four changed-input replays at B1/17/32/64, real output-head weights and synthetic hidden states. ABBA timings showed no stable B32 benefit, so no runtime integration was selected. See results/sampling-head.json. The exact-transform combination is now running a complete C1–C64 trial in runs/continuous-exact-combined-full-round1, with separate ASR and speaker scoring scheduled on physical NPU 4.


### Full combined trial progress and prefill follow-up

The full exact-combination trial completed C1 at 59.71978 frames/s; all 128 WAV files match the previous full-quality combined candidate byte-for-byte. The server is still running later cohorts and independent audio scoring is in progress, so this is not a completed trial.

Review of retained shutdown graph-shape logs found no prefill batches 5–8 in either the preceding combination or adjacent control. A proposed larger prewarm limit was therefore reverted before measurement. Actual additional shapes were (28,64), (31,64), and (63,64). The next prepared experiment uses the existing power-of-two prefill bucketing option with the directly measured combination; it may change BF16 outputs and requires independent full quality. No gain is inferred from the graph-shape observation.


### Exact combination: full performance trial 1

All 896 requests completed and server exited 0 on physical NPU 6. All 896 WAV files are byte-identical to the previous full-quality combined candidate; independent ASR/speaker scoring remains in progress. This is one full trial, not three-repeat acceptance. C32 remains below twice vLLM round 2.

| Concurrency | frames/s | TTFA p50 ms | TTFA p95 ms | mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 59.7198 | 56.5387 | 58.0846 | 0.210790 |
| 2 | 92.0091 | 74.5824 | 87.1076 | 0.272086 |
| 4 | 146.3947 | 99.0561 | 120.7224 | 0.339074 |
| 8 | 223.0894 | 120.6934 | 194.8373 | 0.440722 |
| 16 | 330.9024 | 161.2696 | 352.4982 | 0.587034 |
| 32 | 409.1234 | 229.0160 | 1096.5623 | 0.888437 |
| 64 | 520.5177 | 754.4532 | 1236.8211 | 1.293950 |

The prefill-bucket combination C32/C64 service experiment has started on the same physical card after the full trial exited. See runs/continuous-exact-bucket-prefill-c32-c64.


### Exact combination with prefill bucketing

All 256 requests completed with server exit 0 on physical NPU 6. Every WAV matches the corresponding exact-combination full-trial WAV byte-for-byte (128/128 at each concurrency). This supports explicit identity-based quality transfer; independent scoring and repeated full-trial acceptance remain separate.

| Concurrency | frames/s | TTFA p50 ms | TTFA p95 ms | mean RTF |
|---:|---:|---:|---:|---:|
| 32 | 429.1365 | 224.0407 | 628.6868 | 0.837848 |
| 64 | 530.1827 | 734.3281 | 1203.6463 | 1.272795 |

Against vLLM round 2, the exploratory ratios are C32 2.00256x and C64 2.27938x. The C32 margin is only 0.128% above the 2x line and this comparison uses different physical cards; it does not establish stable final acceptance. A complete C1–C64 trial of this configuration is running in runs/continuous-exact-bucket-prefill-full-round1. The earlier exact-combination full-trial C1 independent ASR/speaker scores and retained-audio audit now pass.


### Prefill-bucket full performance trial

All 896 requests completed and server exited 0 on physical NPU 6. This repeat did not reproduce a stable C32 2x margin. Relative to the exact-combination full trial, WAV identity was 128/128 at C1/C2/C4/C32/C64, but only 13/128 at C8 and 7/128 at C16. Quality cannot transfer for those changed cohorts; the configuration remains unqualified.

| Concurrency | frames/s | TTFA p50 ms | TTFA p95 ms | mean RTF |
|---:|---:|---:|---:|---:|
| 1 | 59.8501 | 56.1925 | 59.8856 | 0.210397 |
| 2 | 91.6895 | 74.9874 | 89.1899 | 0.273019 |
| 4 | 149.0185 | 97.9961 | 120.4445 | 0.333034 |
| 8 | 225.3128 | 120.8146 | 198.6704 | 0.436125 |
| 16 | 333.0264 | 160.3463 | 358.6177 | 0.587186 |
| 32 | 419.4268 | 228.3700 | 663.9449 | 0.858584 |
| 64 | 515.6635 | 774.6860 | 1277.1045 | 1.310878 |

Independent quality integrity audits now pass for exact-combination C4 and vLLM third-repeat C16/C32. The CDF-tail isolated operator probe is running; no kernel gain or model correctness is assumed.


### CDF-tail fusion diagnostic and model control

The isolated CDF-tail kernel preserves the reference FP32 cumulative sum and fuses normalization, comparison and counting using round-to-nearest division. Eight NPU shape cases (B1/17/32/64, widths 2048/3072), each with 12 random/exact-boundary/neighbor replays, matched selected tokens exactly and exited 0. At B32 width 2048, ABBA reference timings were 0.04788/0.04770 ms and fused 0.01116/0.01097 ms. This is isolated operator evidence, not service gain.

Default-off --triton-cdf now applies the tail kernel to predictor and semantic sampling. Six existing sampling/server tests pass. A four-request real-model probe completed with slot reuse, capacity growth 64 to 128, and exit 0; all four WAV files match the flag-off control byte-for-byte. Full cohort service/quality validation remains pending.


### Third vLLM exploratory repeat: full quality complete

All 896 third-repeat vLLM waveforms now have ASR and speaker scores; the scoring loop exited 0 and all seven retained-cohort integrity audits pass. Performance for this repeat was on physical NPU 1, unlike the first two on NPU 0; it remains exploratory under the prospective same-card final-validation plan. The previously scored full combined candidate passes all seven frozen quality margins against vLLM round 3 and SGLang round 2 (an explicitly mixed-repeat quality check, not final acceptance).

Exact-combination full-trial C8 independent scores and integrity audit are also complete. The changed prefill-bucket C8/C16 outputs are being independently scored on physical NPU 1. The CDF service process remains in initial graph/kernel prewarming; no restart or throughput claim has been made.


### CDF service evidence

The CDF-tail service experiment completed all 256 requests and server exit 0. Initial sampler compilation was allowed to finish in the original process. All WAV files at C32 and C64 match the exact-combination full trial byte-for-byte. Isolated operator savings did not establish a stable end-to-end 2x result.

| Concurrency | frames/s | TTFA p50 ms | TTFA p95 ms | mean RTF |
|---:|---:|---:|---:|---:|
| 32 | 412.1818 | 226.5083 | 1104.2413 | 0.883748 |
| 64 | 519.6544 | 776.2377 | 1281.0302 | 1.304500 |

The next direct experiment combines this option with existing prefill bucketing. Prefill-bucket C8 independent ASR/speaker evidence and integrity audit are complete; C16 remains pending. Exact-combination C16 independent quality integrity audit also passed.


### Direct CDF + bucket measurement and prompt-preparation diagnostic

CDF plus prefill bucketing completed 256 requests and server exit 0: C32 423.39616 and C64 519.81898 frames/s. Every WAV matches the bucketed full trial at the corresponding concurrency. This does not establish a stable C32 2x margin. The bucketed full-trial changed C8/C16 cohorts now independently pass ASR, speaker and retained-data integrity checks; other cohorts have separately recorded waveform identity evidence.

The direct CustomVoice prompt diagnostic compared all four prepared tensors and sampling settings for the full 128-record corpus plus instruction/automatic-language/dialect cases (131 total). All tensors were exactly equal, process exit 0. ABBA preparation timing was upstream 6.5263/6.5891 ms per request versus direct 1.3163/1.3123 ms. Only model-constant special-token and speaker/language embeddings are cached; request text and instruction projections are recomputed. This is preparation-level timing, not service performance. A default-off --direct-prompt option for the 1.7B CustomVoice model is under real-waveform validation; reference-single dispatch is excluded.

The direct-prompt real-model probe completed all four English/Chinese requests, slot reuse and capacity growth 64 to 128 with exit 0. All four WAVs exactly match the upstream-prompt control. The complete current test suite passed 57 tests with exit 0. C32/C64 service validation is the next step.
