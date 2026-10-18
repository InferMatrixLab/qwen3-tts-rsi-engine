# Qwen3-TTS CustomVoice：三个推理引擎对比

模型：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`  
机器：Ascend 910B3（64 GiB），每个引擎一张卡  
日期：2026-10-18

| 引擎 | 版本 | 设备 | 端口 |
| --- | --- | --- | --- |
| vLLM-Omni | `quay.io/ascend/vllm-omni:v0.31.0rc1`（vLLM 0.31.0，`32ead8e`） | NPU 0 | 18111 |
| SGLang-Omni | `sglang-omni:v0.1.8-cann9.0.0-910b`（镜像内 SGLang 0.5.20） | NPU 1 | 18112 |
| 独立引擎 | [qwen3-tts-engine](https://github.com/amy-why-3459/qwen3-tts-engine) `main` `3a8b9a5` | NPU 0 | 8099 |

三套请求全部成功。吞吐统一成 **frames/s = 生成音频秒数 × 12.5 / 墙钟时间**，越高越好。RTF 是单请求耗时除以音频时长，越低越好。

## 对比

| 并发 | 框架 | 吞吐 (frames/s) | RTF | 英文 WER |
| ---: | --- | ---: | ---: | ---: |
| 1 | SGLang-Omni | 4.8 | 2.63 | 10.65% |
| 1 | vLLM-Omni | 17.3 | 0.73 | 5.54% |
| 1 | 独立引擎 | **59.9** | **0.24** | **1.00%** |
| 2 | SGLang-Omni | 8.8 | 2.84 | 1.14% |
| 2 | vLLM-Omni | 31.3 | 0.80 | 3.84% |
| 2 | 独立引擎 | **97.9** | **0.26** | **1.00%** |
| 4 | SGLang-Omni | 16.6 | 2.96 | 1.14% |
| 4 | vLLM-Omni | 57.9 | 0.85 | 1.28% |
| 4 | 独立引擎 | **130.4** | **0.42** | **1.00%** |
| 8 | SGLang-Omni | 29.5 | 3.24 | 1.28% |
| 8 | vLLM-Omni | 104.0 | 0.93 | 0.85% |
| 8 | 独立引擎 | **185.0** | **0.58** | 1.00% |
| 16 | SGLang-Omni | 46.7 | 3.79 | **0.43%** |
| 16 | vLLM-Omni | 160.0 | 1.10 | 0.71% |
| 16 | 独立引擎 | **240.5** | **0.90** | 0.71% |
| 32 | SGLang-Omni | 76.1 | 4.23 | **0.57%** |
| 32 | vLLM-Omni | 228.0 | 1.39 | 1.85% |
| 32 | 独立引擎 | **359.4** | **1.11** | 0.71% |
| 64 | SGLang-Omni | 92.9 | 4.18 | 1.99% |
| 64 | vLLM-Omni | 230.2 | **1.85** | 1.14% |
| 64 | 独立引擎 | **325.2** | 2.48 | **1.00%** |

每一档独立引擎都比另外两个快。并发 1 时吞吐大约是 vLLM-Omni 的 3.5 倍、SGLang-Omni 的 12 倍。三者的峰值分别是独立引擎并发 32 的 359 frames/s、vLLM-Omni 并发 64 的 230 frames/s、SGLang-Omni 并发 64 的 93 frames/s。并发 64 时独立引擎的 RTF 升到 2.48，高于 vLLM-Omni 的 1.85，因为大约一半请求要等 11 秒才出首包。

英文词错误率大多在 0.4%–2%。vLLM-Omni 和 SGLang-Omni 在并发 1 各有一条词错误率超过 50% 的样本，把该档拉到 5.54% 和 10.65%。

独立引擎另外测了中文字错误率：

| 并发 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 中文 CER | 3.48% | 3.48% | 3.39% | 3.39% | 3.23% | 2.73% | 4.14% |

## 测法差异

vLLM-Omni 和 SGLang-Omni 用的是同一套客户端：seed-tts 英文 `meta.lst` 前 64 句，说话人 Ryan，非流式整段 WAV，预热 4 条不计入。词错误率用 Whisper large-v3，24 kHz 线性重采样到 16 kHz。采样温度 0.9，所以各并发档不是同一条音频。

独立引擎按仓库 README 的实测配置：`workload-128.jsonl` 共 128 条（64 英 Ryan + 64 中 Vivian），流式 PCM，预热 2 条。精度用仓库的 `tools/evaluate_audio.py` 和同一份 Whisper large-v3（FP32）。从并发 8 起，首包两极分化：多数请求仍是一两百毫秒，另有一批要等数秒。并发 8 是 8 条，16 是 16 条，32 是 31 条，64 时一半请求大约 11 秒才出第一包。

这是单次运行，不是 README 里三次重复的 910B1 表。SGLang 的 NPU 配置把 `max_running_requests` 从镜像默认的 1 调到了 64，图和编译保持关闭。镜像里的 SGLang 0.5.20 和 Omni 0.1.8 的接口不完全匹配，服务是打过容器内兼容补丁后拉起的。

## 启动和压测命令

模型目录：`/home/zhuangzeliang/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`  
seed-tts 英文列表：`/home/zhuangzeliang/models/hub/datasets--zhaochenyang20--seed-tts-eval/snapshots/8f5e1aa2a35d42f42e940074c1983358b9491f89/en/meta.lst`  
Whisper：`/home/zhuangzeliang/models/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1`

### vLLM-Omni

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
vllm serve /home/zhuangzeliang/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni --host 127.0.0.1 --port 18111 \
  --deploy-config /vllm-workspace/vllm-omni/vllm_omni/deploy/qwen3_tts.yaml \
  --trust-remote-code
```

```bash
python3 /home/zhuangzeliang/why/qwen3tts-seedtts-20261018/bench_speech.py \
  --url http://127.0.0.1:18111 \
  --model /home/zhuangzeliang/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --meta /home/zhuangzeliang/models/hub/datasets--zhaochenyang20--seed-tts-eval/snapshots/8f5e1aa2a35d42f42e940074c1983358b9491f89/en/meta.lst \
  --locale en --concurrency 1 --num-prompts 64 --warmup 4 --timeout 300 \
  --wav-dir /home/zhuangzeliang/why/qwen3tts-seedtts-20261018/wavs/vllm-en-c1 \
  --out /home/zhuangzeliang/why/qwen3tts-seedtts-20261018/vllm-en-c1.json
```

`--concurrency` 换成 2、4、8、16、32、64。词错误率用同目录的 `score_wer.py`。

### SGLang-Omni

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
sgl-omni serve \
  --model-path /home/zhuangzeliang/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config /home/zhuangzeliang/why/qwen3tts-seedtts-20261018/sglang_qwen3_tts_1_7b_customvoice_npu.yaml \
  --host 127.0.0.1 --port 18112
```

压测命令与 vLLM-Omni 相同，把 `--url` 改成 `http://127.0.0.1:18112`。

### 独立引擎

在 `/home/zhuangzeliang/why/qwen3-tts-engine` 下执行。裸 `python -m qwen3_tts_engine.server` 的默认值和下面这组实测开关不同。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export QWEN_TTS_MODEL=/home/zhuangzeliang/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
python -m qwen3_tts_engine.server \
  --model "$QWEN_TTS_MODEL" --port 8099 \
  --continuous --streaming --pack-qkv --max-batch-size 64 \
  --codec-left-context 72 --codec-batch-size 4 --incremental-codec \
  --pack-mlp --native-kv-update --batch-prefill --prefill-per-tick 64 \
  --text-only-rope --prewarm-codec --native-gqa \
  --graph-embeddings --graph-prefill --rms-norm native --fused-swiglu \
  --native-rope --graph-sampling --native-rope-scope predictor \
  --triton-snake --cache-codec-codebooks --triton-kv-update \
  --triton-talker-rope --prewarm-prefill --triton-cdf \
  --bucket-prefill --direct-prompt
```

```bash
python -m qwen3_tts_engine.benchmark \
  --endpoint http://127.0.0.1:8099/v1/audio/speech \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --engine independent \
  --source-revision "$(git rev-parse HEAD)" \
  --workload evidence/campaign-20261017/plan/workload-128.jsonl \
  --output runs/c1 \
  --concurrency 1 --requests 128 --warmup 2 --frame-rate 12.5
```

```bash
python tools/evaluate_audio.py \
  --run runs/c1 \
  --asr-model /home/zhuangzeliang/models/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --output runs/c1/asr \
  --device npu:0 --dtype float32
```

## 原始结果

- vLLM-Omni / SGLang-Omni：`vllm-en-c*.json`、`sglang-en-c*.json`、`wer-*-en-c*.json`
- 独立引擎：`engine-runs/c*/results.json`、`engine-runs/c*/asr/summary.json`
