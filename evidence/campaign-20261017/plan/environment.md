# Qualified environment and exceptions

The exploratory campaign uses `why_vllm_omni_v0.31.0` on Ascend 910B1,
with torch 2.10.0, torch_npu 2.10.0.post4, vLLM 0.31.0 and
vllm-ascend 0.19.1rc2.dev2590+ge25912733. The standalone adapter uses
qwen-tts 0.1.1 and transformers 4.57.3 in an isolated environment.
Model and source identities are pinned in `contract.json` and `sources.md`.

The untouched upstream startup probes failed in this installed environment:
vLLM aborted during model-registry import; SGLang additionally imported a
`torch.xpu._POOL_HANDLE` type absent from torch 2.10. The working qualifications
used `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2`. This is an
allocator workaround, not evidence that the underlying heap issue was fixed.
SGLang also used the retained type-only patch in `../patch/` in a separate
checkout. Its installed SGLang core revision was
`e00930c5489053f26d86b179cee0d087f846acbb` (0.5.21), whereas the upstream NPU
manifest requested 0.5.20. Results must retain these qualifications.

To reproduce those environment choices with the convenience launcher:

```bash
ALLOCATOR_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
  NPU_DEVICE=0 bash tools/serve_baseline.sh vllm
ALLOCATOR_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
  SGLANG_SOURCE_ROOT=/home/why/qwen3-tts-research/sglang-omni-compat \
  SGLANG_PYTHON=/home/why/drama3-research/venv-sglang/bin/python \
  NPU_DEVICE=0 PORT=8098 bash tools/serve_baseline.sh sglang
```

The compatibility checkout must already contain the documented patch. The
launcher does not apply patches or install dependencies. Preserve the container
CANN `PYTHONPATH`; replacing it can break NPU initialization.

Full 128-request scans use `workload-128.jsonl`. The older `workload.jsonl`
contains 84 records and remains unchanged to preserve qualification provenance.
The first attempted 128-request scan rejected that shorter corpus before issuing
requests. The corrected corpus and source hashes are in `workload-128-manifest.json`.

Exploratory smoke runs used different available devices and are not matched
performance comparisons. Formal acceptance requires sequential matched arms on
one device, repeated measurements, independent audio scoring and clean shutdown.
