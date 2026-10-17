#!/usr/bin/env bash
set -euo pipefail
engine=${1:?usage: serve_baseline.sh vllm|sglang}
engine_root=$(cd "$(dirname "$0")/.." && pwd)
research_root=${RESEARCH_ROOT:-$(dirname "$engine_root")/qwen3-tts-research}
container=${NPU_CONTAINER:-why_vllm_omni_v0.31.0}
device=${NPU_DEVICE:-0}
port=${PORT:-8096}
allocator=${ALLOCATOR_PRELOAD:-}
sglang_source=${SGLANG_SOURCE_ROOT:-$research_root/sglang-omni}
if [[ "$engine" == vllm ]]; then
  exec docker exec -e ASCEND_RT_VISIBLE_DEVICES="$device" \
    -e LD_PRELOAD="$allocator" \
    -e RESEARCH_ROOT="$research_root" -e PORT="$port" \
    -w "$research_root/vllm-omni" "$container" bash -c '
    export PYTHONPATH="$RESEARCH_ROOT/vllm-omni:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
    exec "$RESEARCH_ROOT/venv-vllm/bin/vllm-omni" serve "$RESEARCH_ROOT/model" \
      --omni --host 127.0.0.1 --port "$PORT" --trust-remote-code
    '
elif [[ "$engine" == sglang ]]; then
  : "${SGLANG_PYTHON:?Set SGLANG_PYTHON to the qualified NPU environment interpreter}"
  exec docker exec -e ASCEND_RT_VISIBLE_DEVICES="$device" \
    -e LD_PRELOAD="$allocator" -e SGLANG_SOURCE_ROOT="$sglang_source" \
    -e RESEARCH_ROOT="$research_root" -e ENGINE_ROOT="$engine_root" \
    -e SGLANG_PYTHON="$SGLANG_PYTHON" -e PORT="$port" \
    -w "$sglang_source" "$container" bash -c '
    export PYTHONPATH="$SGLANG_SOURCE_ROOT:$RESEARCH_ROOT/Qwen3-TTS:$RESEARCH_ROOT/extra:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
    exec "$SGLANG_PYTHON" -m sglang_omni.cli serve \
      --model-path "$RESEARCH_ROOT/model" \
      --config "$ENGINE_ROOT/configs/sglang-npu.yaml" --host 127.0.0.1 --port "$PORT"
    '
else
  echo "Unknown engine: $engine" >&2
  exit 2
fi
