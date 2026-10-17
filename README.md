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

Model weights and generated waveforms are kept outside Git. Compact results and
SHA256 manifests identify retained local artifacts. CPU tests validate the
measurement contract, not NPU execution or model quality.
