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

Model weights and generated waveforms are kept outside Git. Compact results and
SHA256 manifests identify retained local artifacts. CPU tests validate the
measurement contract, not NPU execution or model quality.
