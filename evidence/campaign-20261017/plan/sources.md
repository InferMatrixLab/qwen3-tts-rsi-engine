# Sources and applicable mechanisms

- [OmniRSI PR 9](https://github.com/ThinkFlowLab/OmniRSI/pull/9),
  `7b86234b7ae2cefeacc96a611f38535c7363a6ad`: source-grounded Strata,
  vLLM-Omni and SGLang-Omni skills, frozen validation and retained rejected trials.
- [vLLM-Omni main](https://github.com/vllm-project/vllm-omni/tree/b164fea797531c1fa10f8b87829632cad006b9fc):
  `benchmarks/tts/bench_tts.py`, Seed-TTS default-voice workload,
  streaming speech API and NPU platform/model implementation.
- [SGLang-Omni main](https://github.com/sgl-project/sglang-omni/tree/6bdd47967caa55c38430d18175b750008aecab61):
  `benchmarks/eval/benchmark_tts_seedtts.py`, `benchmarks/tts_serving`,
  CustomVoice request semantics and NPU recipes. Single-request NPU defaults
  must be explicitly distinguished from qualified concurrency configurations.
- [Ascend skills](https://github.com/ascend-ai-coding/awesome-ascend-skills):
  torch_npu environment, profiling and operator validation.
- [MindIE agent](https://github.com/mindie-agent/mindie-agent): evidence-driven
  Ascend performance investigation; imported guidance is not measured evidence.

Strata contributes native runtime/state-ownership and constrained search ideas.
Its MoE expert placement is not applicable to this dense TTS model. CUDA graph
results do not establish NPU graph safety. Keep codec state request-owned;
validate cancellation, slot reuse and batching before accepting concurrency gains.

The local performance guide requires profiling separately from timing, one
mechanism per experiment, three independent repeats and a final direct comparison.
