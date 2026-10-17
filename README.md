# Qwen3-TTS NPU engine

Independent inference work for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` on Ascend 910B1.

**Status: INCONCLUSIVE — implementation and baseline qualification in progress.**
No 2x performance or audio-quality result has been established.

The acceptance target is at least 2x the better of pinned vLLM-Omni and
SGLang-Omni main throughput at each concurrency: 1, 2, 4, 8, 16, 32, 64.
TTFA p50/p95 and mean request RTF are reported alongside throughput. All arms
must use identical model weights, hardware budget, input corpus and sampling.
Full audio, request failures, terminal events and clean server exits are retained.

See [the campaign contract](evidence/campaign-20261017/plan/contract.json)
and [source provenance](evidence/campaign-20261017/plan/sources.md).
The [first exploratory sweep](evidence/campaign-20261017/round1.md) contains
all four requested metrics for vLLM C1–C64; SGLang and quality evaluation are
still running. It is not the final repeated comparison.

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
`--streaming` uses an initial short codec window followed by larger windows with
25 frames of left context. Chunk decoding can change PCM rounding, so full
streaming ASR and speaker quality are evaluated independently.
`--pack-qkv` passed the four-prompt exact check. `--rms-norm` remains an
approximate numerical experiment: its reference-rounding variant failed exact
parity, and native fusion requires the frozen full-corpus quality gates before
promotion. None of these options changes the default reference path.
See the retained counterexample in `evidence/campaign-20261017/results/`.

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
