# Qwen3-TTS NPU engine

Independent inference work for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` on Ascend 910B1.

**Status: INCONCLUSIVE — full repeated, quality-qualified 2x acceptance is pending.**
Both baseline exploratory sweeps are complete. The independent streaming engine
has completed all seven concurrency levels; high-concurrency optimization and
full candidate audio-quality evaluation continue.

The acceptance target is at least 2x the better of pinned vLLM-Omni and
SGLang-Omni main throughput at each concurrency: 1, 2, 4, 8, 16, 32, 64.
TTFA p50/p95 and mean request RTF are reported alongside throughput. All arms
must use identical model weights, hardware budget, input corpus and sampling.
Full audio, request failures, terminal events and clean server exits are retained.

See [the campaign contract](evidence/campaign-20261017/plan/contract.json)
and [source provenance](evidence/campaign-20261017/plan/sources.md).
The [first exploratory sweep](evidence/campaign-20261017/round1.md) contains
all four requested metrics and full ASR/speaker scores for both baselines at
C1–C64. SGLang requires the documented import compatibility patch; it is not an
untouched-main run. These are exploratory single-repeat results, not the final
repeated comparison.

The native-RoPE/sampling-graph candidate (`3ef961a`, physical NPU 0) measured
56.61 / 85.70 / 135.00 / 207.88 / 295.96 / 386.54 / 473.18 frames/s at
C1 / C2 / C4 / C8 / C16 / C32 / C64, with 896 successful requests and clean
server exit. C32 and C64 remain below their respective approximately 400.53
and 484.15 frames/s targets. Changing codec batch size to 8 yielded C64
486.15 frames/s but regressed C32 to 353.97; it is not an accepted all-concurrency
solution. See the campaign report for TTFA, RTF, retained quality evidence,
and the limitations of comparing single runs on a shared host.

`--fused-residual --rms-norm native` is a new default-off experiment that fuses
the post-attention residual addition and RMSNorm. Operator validation and
contract tests pass, but BF16 normalization changes and full speech-quality
validation is required. No service gain is presumed from operator timings.

Benchmark against an already running OpenAI speech SSE endpoint:

```bash
python -m qwen3_tts_engine.benchmark --help
```

The initial standalone service owns its admission queue, batching, HTTP delivery
and model lifecycle; it has no vLLM/SGLang runtime dependency. The reference model
adapter uses upstream `qwen-tts==0.1.1` with `transformers==4.57.3`, BF16 and SDPA.
This bootstrap path delivers the complete waveform in one SSE audio event;
its TTFA therefore includes the whole synthesis. Incremental codec delivery and
optimized decode have not yet been qualified. It is not the final performance candidate.

The opt-in `--predictor-graph` path is experimental. Whole-predictor graph capture
at commit `1761366` passed reseeded component checks but failed full-model EOS
termination and was rejected. Capture constants were not retained for replay;
preserving their lifetime fixes the observed per-step graph discrepancy.
The per-step path now passes sequential token/RNG comparison and exact full PCM
parity on two English and two Chinese prompts, with clean process exit. This
limited check does not establish corpus-wide quality or service performance.
The corrected whole-predictor path and its combination with `--talker-graph`
also pass those four complete-waveform checks. Use `--predictor-graph whole
--talker-graph` to select the combination. The talker graph retains eager prefill
and uses tensor-indexed KV storage with capacity buckets during decode.
These graphs are not enabled by default. Mixed English/Chinese batch size four
also passed complete PCM parity. The separate 8-request streaming qualification
completed with clean shutdown, TTFA p50/p95 208.5/215.5 ms and 21.23 frames/s;
English WER was zero for those eight clips. This is not a matched 2x result.
`--streaming` uses an initial short codec window followed by larger windows.
The earlier measurements at commit `4ce009a` used 25 frames of left context;
current experiments use 72 to match the decoder attention window and expose
`--codec-batch-size` for bounded codec batching. Chunk decoding can change PCM rounding, so full
streaming ASR and speaker quality are evaluated independently.
`--pack-qkv` passed the four-prompt exact check. `--rms-norm` remains an
approximate numerical experiment: its reference-rounding variant failed exact
parity, and native fusion requires the frozen full-corpus quality gates before
promotion. None of these options changes the default reference path.
See the retained counterexample in `evidence/campaign-20261017/results/`.

`--continuous --streaming --max-batch-size 64` selects the experimental native
iteration scheduler. It owns per-request KV slots, seeded uniform sampling,
EOS completion, slot compaction and capacity growth. It reuses official prompt
construction and model weights but does not call Hugging Face's generation loop.
The inverse-CDF sampler preserves the configured categorical distribution;
its seed-to-token mapping differs from `torch.multinomial` and is checked through
independent audio scoring. Slot reuse and capacity growth passed a four-request
NPU probe, and C1/C2/C4 each completed eight HTTP requests with clean shutdown.
The complete concurrency/quality acceptance run is still pending.

`--continuous --incremental-codec` selects the experimental stateful codec. It
retains convolution tails and per-request sliding attention KV state, and captures
graphs for new chunks of one to eight frames instead of decoding a full left
context again. Strict FP32 tests cover interleaved streams, capture while other
streams are active, slot movement and reset. BF16 real-prompt output remains
numerically approximate and requires independent full-corpus audio scoring.
See [the codec investigation](evidence/campaign-20261017/plan/incremental-codec.md).

`--continuous --batch-prefill` experimentally groups new admissions into a padded
prefill batch, crops KV storage back to each prompt's length and projects logits
only for each prompt's final token. The initial two-admission BF16 scheduling probe
retained identical PCM to individual prefill, but a mixed four-admission batch
changed generated tokens and PCM. The stricter FP32 per-layer KV
comparison exceeded its 1e-4 tolerance on one observed element. That counterexample
is retained, and this option still requires full-corpus quality and speed checks.

`--continuous --graph-prefill` captures padded prefill computation. Same-shape
FP32 eager/graph checks produced identical hidden states, logits and all KV layers
under two request permutations. Padding buckets still change BF16 generation,
so this does not establish complete-waveform parity or quality acceptance.
`--native-gqa` uses the inference-specific Ascend grouped attention operator;
its full-waveform checks changed generated output and it remains approximate.

`--pack-mlp` combines the gate and up projections and passed the four-prompt
complete-waveform comparison. `--fused-swiglu` additionally replaces SiLU and
multiply with Ascend SwiGLU; it is a separate numerical experiment and is not
covered by the packed-projection parity result.

`--continuous --native-kv-update` uses Ascend ScatterND for per-request KV writes.
Exact graph-write tests cover batches 1, 4, 32 and 64; a four-request scheduling
probe retained identical PCM across slot reuse and capacity growth. Its operator
timing improvement is not an end-to-end throughput claim.

`tools/profile_continuous.py` collects a separate CPU/NPU trace with ranges for
prefill, the predictor, talker, semantic sampling and codec. It warms the engine
before tracing and records admissions, completions and KV capacity per tick.
New shape captures can still occur in the trace; its timings are diagnostic only.
Run it on an available device separately from performance measurements:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python tools/profile_continuous.py \
  --model /path/to/pinned/model \
  --workload evidence/campaign-20261017/plan/workload-128.jsonl \
  --slots 32 --output runs/profile-continuous-c32
```

```bash
python -m qwen3_tts_engine.server --model /path/to/pinned/model --device npu:0
bash tools/serve_baseline.sh vllm
SGLANG_PYTHON=/path/to/npu/venv/bin/python bash tools/serve_baseline.sh sglang
python tools/audit_model.py --tree /path/to/model-tree.json --model /path/to/model --output model-audit.json
```

`tools/serve_baseline.sh` uses an existing NPU container and preserves its CANN
Python paths. Configure `NPU_CONTAINER`, `RESEARCH_ROOT`, `NPU_DEVICE` and `PORT`.
The SGLang configuration is a concurrency qualification candidate, not a measured
best configuration. Setup scripts do not alter either upstream main checkout.
See [environment exceptions and reproduction](evidence/campaign-20261017/plan/environment.md)
for the allocator workaround and SGLang type compatibility patch used by the
successful qualifications; untouched-main startup failures are retained too.

Model weights and generated waveforms are kept outside Git. Compact results and
SHA256 manifests identify retained local artifacts. CPU tests validate the
measurement contract, not NPU execution or model quality.

`--text-only-rope` is an experimental packed-attention option specific to CustomVoice's identical three position axes. It selects one rotary axis instead of repeating the multimodal interleaving copies; it must not be reused for unequal-axis multimodal input. Four mixed English/Chinese prompts at batch size 4 passed complete PCM parity against the official path with graph predictor and talker, and 18 focused tests passed. It remains opt-in pending continuous-service performance and full-corpus quality validation.

For component diagnosis without shape-recording profiler overhead, `tools/profile_continuous.py --synchronized-timings` records inclusive synchronized ranges. The tool also accepts `--rms-norm reference`, `--pack-mlp`, `--prefill-per-tick`, and `--codec-batch-size`. These diagnostic times perturb execution and are not benchmark results.

`--prewarm-codec` (with `--continuous --incremental-codec`) captures every codec batch/length shape up to the configured batch size before the server reports readiness. The capture preserves slot histories, KV buffers, and frame counters. A strict FP32 NPU test verified exact live-state preservation across all 32 shapes and subsequent interleaved 101-frame decoding, movement, and reset. Startup cost and graph memory increase; service benefit must be measured separately.

`--bucket-kv` is an opt-in continuous attention experiment. KV storage retains its full capacity, while attention reads a power-of-two prefix covering every active position (minimum 64). Cache updates and masked-tail equivalence passed focused tests; the complete suite reports 20 passes. An NPU pair with a 512-position arena produced different PCM for 3 of 4 prompts despite successful EOS completion, so this changes numerical behavior and requires full quality validation. It is not an exact-output optimization.

`--graph-embeddings` captures the 15 subcode embedding lookups and their sum with the first-code embedding inside the continuous predictor graph. It retains the original reduction order. Four full continuous generations (two English, two Chinese, slot reuse and KV growth) matched the prior path's PCM hashes exactly. Full-service speed and quality remain separate validation steps.

`--bucket-prefill` (requires `--graph-prefill`) rounds the admission batch to the next power of two using independent dummy rows and admits only real requests. This limits repeated capture for varying admission counts. Strict FP32 NPU comparisons of 4-to-4, 3-to-4 and 1-to-1 batches matched hidden states, logits and all KV tensors exactly in the retained probe. An intentionally overpadded 1-to-4 stress case exceeded the strict 1e-5 tolerance and is retained separately; that shape choice is not used by this option. BF16 full-service performance and quality still require validation.

`--decode-bucket-step 8` selects multiples of eight for larger active batches while retaining power-of-two graphs up to eight requests. It reduces padded rows (for example, 33 active requests use 40 rows instead of 64). The option is experimental: all arena/count boundaries passed tests, and 17 complete mixed-language generations exercised the 24-row graph and shrinking active batches. Full quality and service performance are not established by that probe.

`--batch-audio-transfer` queues owned codec output clones for all groups due in a tick and performs one host transfer, then emits the same per-request PCM chunks in the original callback order. Four complete continuous generations matched the prior PCM hashes exactly; a ragged-group regression checks lengths, bytes and callback order. This may defer the earliest callback until all groups are ready, so TTFA must be measured with throughput.

`--reference-single` selects upstream RMSNorm, unfused SiLU/multiply and SDPA when the model batch has one row, and uses unpadded eager prefill for one arriving request. Larger batches retain the configured fusion options. This dispatch depends only on batch shape, never prompt content. Four full B1 generations with all fusion flags enabled matched the official-reference PCM exactly under this option. The suite passed 28 tests plus two focused dispatch tests; full continuous-service performance and quality remain to be measured, including its effect on high-concurrency tails.

`--graph-sampling` prewarms semantic-sampling graphs for every actual request count and updates seen-token masks in the same graph. The output-head matrix retains its original shape; no sampling rows are padded. NPU probes at B1/B17/B64 preserved sampled tokens, minimum-EOS masking and seen masks exactly across repeated replays, including retained output ownership. Four full continuous outputs also matched PCM exactly. The current test suite reports 30 passes. Service performance remains a separate measurement.

`--native-rope` uses the NPU rotary operator on contiguous BSND query/key norm outputs, then exposes BNSD views to attention. It implies the CustomVoice equal-axis rotary selection. Twelve NPU graph shapes completed with finite outputs; the largest measured BF16 absolute difference was 0.03125, so full speech-quality validation is mandatory. Four complete continuous generations passed functional checks. The suite now reports 32 passing tests. The initial rotary probe incorrectly compared outputs before the first replay; its retained error numbers are invalid and are explicitly superseded by `rotary-bsnd-probe-result.json`.

Continuous `--reference-single` now limits eager reference prefill to a genuinely solitary active request. A single new arrival joining existing requests uses native prefill arithmetic; the temporary dispatch override is restored even if prefill raises. Single-row decode still uses the reference path. This corrects a measured high-concurrency slowdown from applying the low-load policy to every one-request refill.
