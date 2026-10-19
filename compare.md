# Qwen3-TTS CustomVoice：本机 910B1 三框架对比

参考 [910B3 报告](compare-910b3.md) 的结构，整理本机已有实测。本文不是新一轮压测结果。
910B3 文档保留用户提供的原文；其中工作负载、版本及预热与本报告不同，不可直接据此归因硬件性能差异。

模型：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`，固定 revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`。  
机器：Ascend **910B1，64 GiB**，每次只使用物理 NPU 0 一张卡，三套框架顺序运行。  
测量日期：2026-10-17 至 2026-10-18。每框架三个独立服务进程，每轮覆盖 C1/2/4/8/16/32/64。

| 引擎 | 固定版本 | 设备 | 端口 |
| --- | --- | --- | --- |
| vLLM-Omni | Omni `b164fea797531c1fa10f8b87829632cad006b9fc`；vLLM 0.31.0 | NPU 0 | 8096 |
| SGLang-Omni | Omni `6bdd47967caa55c38430d18175b750008aecab61` + 类型导入兼容补丁；core 0.5.21 | NPU 0 | 8098 |
| 独立引擎 | `e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a` | NPU 0 | 8099 |

所有 8,064 条计时请求成功，九个服务进程均正常退出（exit 0）。完整精度与生命周期验收为 **PASS**。
各轮 `status.json` 中的初始 `INCONCLUSIVE / NOT_RUN` 不包含随后进行的精度验收；最终结论以
[汇总验收](evidence/campaign-20261017/acceptance.md) 为准。

## 性能对比

下表为三轮统计量的 **均值 ± 样本标准差**，并非置信区间。TTFA 的 p50/p95 是各轮分位数的均值，不是合并请求后的分位数。

| Concurrency | Framework | Throughput (frames/s) ↑ | TTFA p50 (ms) ↓ | TTFA p95 (ms) ↓ | RTF (mean) ↓ |
|---:|---|---:|---:|---:|---:|
| 1 | SGLang-Omni | 4.31 ± 0.04 | 439.72 ± 17.08 | 556.22 ± 9.41 | 2.922 ± 0.036 |
| 1 | vLLM-Omni | 12.71 ± 0.39 | 263.51 ± 6.94 | 273.21 ± 10.52 | 0.989 ± 0.030 |
| 1 | Independent engine | 61.12 ± 0.54 | 50.41 ± 0.39 | 51.97 ± 0.58 | 0.206 ± 0.002 |
| 2 | SGLang-Omni | 7.97 ± 0.05 | 599.49 ± 2.64 | 788.08 ± 4.73 | 3.146 ± 0.020 |
| 2 | vLLM-Omni | 24.32 ± 1.23 | 305.15 ± 16.96 | 349.79 ± 22.07 | 1.032 ± 0.053 |
| 2 | Independent engine | 95.72 ± 0.61 | 67.55 ± 0.31 | 79.76 ± 0.53 | 0.261 ± 0.002 |
| 4 | SGLang-Omni | 14.35 ± 0.31 | 632.13 ± 24.59 | 919.25 ± 59.49 | 3.443 ± 0.075 |
| 4 | vLLM-Omni | 46.43 ± 2.32 | 320.69 ± 21.08 | 402.43 ± 24.04 | 1.073 ± 0.056 |
| 4 | Independent engine | 154.23 ± 0.55 | 89.56 ± 0.70 | 107.59 ± 0.90 | 0.322 ± 0.001 |
| 8 | SGLang-Omni | 25.59 ± 0.50 | 662.13 ± 24.15 | 1123.95 ± 125.27 | 3.840 ± 0.073 |
| 8 | vLLM-Omni | 83.36 ± 4.32 | 375.63 ± 16.61 | 492.49 ± 25.92 | 1.169 ± 0.059 |
| 8 | Independent engine | 231.65 ± 1.87 | 113.23 ± 1.27 | 162.26 ± 3.51 | 0.424 ± 0.004 |
| 16 | SGLang-Omni | 40.20 ± 1.71 | 716.00 ± 45.37 | 1605.89 ± 115.90 | 4.649 ± 0.131 |
| 16 | vLLM-Omni | 134.52 ± 8.81 | 466.73 ± 29.24 | 643.21 ± 15.53 | 1.313 ± 0.059 |
| 16 | Independent engine | 348.02 ± 3.69 | 146.89 ± 0.96 | 267.99 ± 8.56 | 0.561 ± 0.006 |
| 32 | SGLang-Omni | 57.62 ± 0.68 | 808.37 ± 20.98 | 2435.48 ± 23.69 | 6.127 ± 0.155 |
| 32 | vLLM-Omni | 209.45 ± 7.96 | 669.10 ± 34.99 | 994.41 ± 40.55 | 1.630 ± 0.077 |
| 32 | Independent engine | 439.73 ± 1.50 | 214.90 ± 0.89 | 489.36 ± 7.81 | 0.813 ± 0.003 |
| 64 | SGLang-Omni | 71.60 ± 4.43 | 3410.41 ± 663.46 | 7436.21 ± 304.93 | 9.386 ± 0.706 |
| 64 | vLLM-Omni | 228.07 ± 16.73 | 1333.99 ± 53.72 | 1721.06 ± 62.77 | 2.477 ± 0.120 |
| 64 | Independent engine | 549.39 ± 8.01 | 582.98 ± 19.72 | 920.75 ± 39.28 | 1.212 ± 0.020 |

各并发下，每次独立引擎实测吞吐都超过两套基线所有三轮中最高值的 2 倍；最小观察比值为 C32 的 **2.047683×**。
此结论仅适用于本报告固定的硬件、配置和工作负载，不是所有场景的保证。

## 精度对比

每个计时 cohort 全量评估 64 条英文和 64 条中文；Whisper-large-v3 FP32 测英文 WER / 中文 CER，
固定 WavLM-large verifier 与相同独立生成的 Ryan/Vivian anchors 测说话人余弦相似度。
下表也是三轮均值 ± 样本标准差；WER/CER 用百分比，越低越好，余弦值越高越好。

| Concurrency | Framework | English WER (%) ↓ | Chinese CER (%) ↓ | Speaker cosine ↑ |
|---:|---|---:|---:|---:|
| 1 | SGLang-Omni | 1.14 ± 0.00 | 2.65 ± 0.00 | 0.7720 ± 0.0000 |
| 1 | vLLM-Omni | 0.85 ± 0.00 | 3.73 ± 0.00 | 0.7709 ± 0.0000 |
| 1 | Independent engine | 1.14 ± 0.00 | 3.15 ± 0.00 | 0.7751 ± 0.0000 |
| 2 | SGLang-Omni | 1.42 ± 0.00 | 3.89 ± 0.00 | 0.7655 ± 0.0000 |
| 2 | vLLM-Omni | 1.14 ± 0.00 | 3.48 ± 0.00 | 0.7682 ± 0.0004 |
| 2 | Independent engine | 1.14 ± 0.00 | 3.15 ± 0.00 | 0.7746 ± 0.0000 |
| 4 | SGLang-Omni | 1.28 ± 0.00 | 3.39 ± 0.00 | 0.7670 ± 0.0000 |
| 4 | vLLM-Omni | 0.57 ± 0.00 | 3.84 ± 0.67 | 0.7725 ± 0.0009 |
| 4 | Independent engine | 1.14 ± 0.00 | 2.98 ± 0.00 | 0.7728 ± 0.0000 |
| 8 | SGLang-Omni | 0.85 ± 0.00 | 3.39 ± 0.00 | 0.7669 ± 0.0000 |
| 8 | vLLM-Omni | 0.85 ± 0.00 | 2.90 ± 0.17 | 0.7741 ± 0.0012 |
| 8 | Independent engine | 1.14 ± 0.00 | 2.98 ± 0.00 | 0.7769 ± 0.0000 |
| 16 | SGLang-Omni | 1.09 ± 0.30 | 2.84 ± 0.59 | 0.7698 ± 0.0014 |
| 16 | vLLM-Omni | 2.75 ± 1.64 | 2.98 ± 0.36 | 0.7731 ± 0.0022 |
| 16 | Independent engine | 0.85 ± 0.00 | 2.73 ± 0.00 | 0.7620 ± 0.0000 |
| 32 | SGLang-Omni | 1.00 ± 0.00 | 3.17 ± 0.25 | 0.7763 ± 0.0021 |
| 32 | vLLM-Omni | 0.95 ± 0.08 | 3.73 ± 0.38 | 0.7725 ± 0.0038 |
| 32 | Independent engine | 0.85 ± 0.00 | 3.15 ± 0.00 | 0.7676 ± 0.0000 |
| 64 | SGLang-Omni | 0.71 ± 0.00 | 3.09 ± 0.33 | 0.7705 ± 0.0008 |
| 64 | vLLM-Omni | 1.14 ± 0.00 | 3.48 ± 0.30 | 0.7710 ± 0.0028 |
| 64 | Independent engine | 1.00 ± 0.00 | 2.73 ± 0.00 | 0.7713 ± 0.0000 |

全部 21 个独立引擎 trial/concurrency 组合通过冻结阈值：相对同并发最优基线观察值，
WER/CER 最多增加 1 个百分点，speaker cosine 最多下降 0.02。自动评估不等于主观自然度评价。

## 统一测法

- 三套框架全部使用 `evidence/campaign-20261017/plan/workload-128.jsonl`：64 英文 Ryan + 64 中文 Vivian，seed 42，max_new_tokens 2048。
- 全部使用同一个 `qwen3_tts_engine.benchmark` 客户端，流式 PCM over SSE，24 kHz，单声道。
- 每档预热 **max(2, concurrency)** 条且不计时；C64 预热 64 条。启动预热没有覆盖全部大批次图，仅预热 2 条不能复现高并发测试流程。
- 吞吐 = 总生成音频秒数 × 12.5 / 整个 cohort 墙钟时间。
- TTFA = 客户端发出请求到收到首个非空 PCM，包含服务端排队；不包含等待客户端并发信号量的时间。
- RTF = 每请求完成耗时 / 音频时长，再求算术平均。
- 共享主机、顺序运行，不作统计显著性或全局最优性声明。精度评估在计时完成后独立运行，不能与被测服务争用同一张卡。

## 本机环境及准备

以下命令针对已有本机容器和模型目录，不是全新机器的安装脚本。已有容器为
`why_vllm_omni_v0.31.0`，Python 3.12、torch 2.10.0、torch_npu 2.10.0.post4；
独立引擎环境含 qwen-tts 0.1.1、transformers 4.57.3、Triton Ascend。
保留容器原有 CANN `PYTHONPATH`。SGLang 兼容 checkout 必须已包含记录的类型导入补丁；
jemalloc preload 是环境 workaround，不代表修复了上游堆问题。

每个需要执行启动或压测命令的终端，先进入同一容器：

```bash
docker exec -it -w /home/why/qwen3-tts-engine why_vllm_omni_v0.31.0 bash
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export QWEN_COMPARE_ROOT=/home/why/qwen3-tts-engine
export QWEN_RESEARCH_ROOT=/home/why/qwen3-tts-research
export QWEN_TTS_MODEL="$QWEN_RESEARCH_ROOT/model"
export QWEN_FROZEN_ROOT="$QWEN_RESEARCH_ROOT/engine-direct-prompt-frozen"
export QWEN_BENCH_PYTHON="$QWEN_RESEARCH_ROOT/venv-engine/bin/python"
export PYTHONPATH="$QWEN_FROZEN_ROOT:${PYTHONPATH:-}"
cd "$QWEN_FROZEN_ROOT"
npu-smi info
```

确认 NPU 0 空闲；同一时间只启动一个被测框架。服务与客户端均从冻结 checkout 执行，以导入原实测版本；
不能只设置 `PYTHONPATH` 后留在主仓库运行 `python -m`，因为当前目录优先。
开始前用 `git -C "$QWEN_FROZEN_ROOT" rev-parse HEAD` 核对上表 SHA，并检查相关 checkout 无未记录修改。
模型权重与 tokenizer 的完整哈希见 [model audit](evidence/campaign-20261017/results/model-audit.json)。

## 启动命令

### vLLM-Omni

在完成上述环境准备的终端运行，使用固定源码 checkout 的默认模型部署配置：

```bash
PYTHONPATH="$QWEN_RESEARCH_ROOT/vllm-omni:$PYTHONPATH" \
  "$QWEN_RESEARCH_ROOT/venv-vllm/bin/vllm-omni" serve --omni \
  "$QWEN_TTS_MODEL" --host 127.0.0.1 --port 8096 --trust-remote-code
```

### SGLang-Omni

固定配置已经设置 `max_running_requests: 64`、`max_queued_requests: 128`，关闭图、编译和 overlap schedule；
完整参数见 [sglang-npu.yaml](configs/sglang-npu.yaml)。

```bash
PYTHONPATH="$QWEN_RESEARCH_ROOT/sglang-omni-compat:$QWEN_RESEARCH_ROOT/Qwen3-TTS:$QWEN_RESEARCH_ROOT/extra:$PYTHONPATH" \
  /home/why/drama3-research/venv-sglang/bin/python -m sglang_omni.cli serve \
  --model-path "$QWEN_TTS_MODEL" \
  --config "$QWEN_COMPARE_ROOT/configs/sglang-npu.yaml" \
  --host 127.0.0.1 --port 8098
```

### 独立引擎

直接加载保存的完整命令，避免漏掉优化开关；Python 和模型路径替换成本机变量。

```bash
"$QWEN_BENCH_PYTHON" - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
command = json.loads((Path(os.environ["QWEN_COMPARE_ROOT"]) /
    "evidence/campaign-20261017/plan/continuous-direct-prompt-command.json"
).read_text())
command[0] = sys.executable
command[command.index("--model") + 1] = os.environ["QWEN_TTS_MODEL"]
subprocess.run(command, cwd=os.environ["QWEN_FROZEN_ROOT"], check=True)
PY
```

该命令包括 `--max-batch-size 64 --prefill-per-tick 64 --continuous --streaming` 及全部实测优化选项。
启动会预热、捕获部分图，等待健康检查成功后再压测。

## 三套框架共用的压测命令

在第二个终端完成同样的容器环境准备，设置当前正在运行的框架和轮次（1、2、3）。
每轮必须重新启动服务；所有输出目录必须是新目录。

```bash
export QWEN_ARM=vllm  # 依次改为 vllm、sglang、independent
export QWEN_TRIAL=1
case "$QWEN_ARM" in
  vllm)
    QWEN_PORT=8096
    QWEN_REV=b164fea797531c1fa10f8b87829632cad006b9fc ;;
  sglang)
    QWEN_PORT=8098
    QWEN_REV=6bdd47967caa55c38430d18175b750008aecab61 ;;
  independent)
    QWEN_PORT=8099
    QWEN_REV=e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a ;;
  *) echo 'Unknown framework'; exit 1 ;;
esac
curl --fail "http://127.0.0.1:$QWEN_PORT/health" || exit 1
for concurrency in 1 2 4 8 16 32 64; do
  warmup=$((concurrency > 2 ? concurrency : 2))
  "$QWEN_BENCH_PYTHON" -m qwen3_tts_engine.benchmark \
    --endpoint "http://127.0.0.1:$QWEN_PORT/v1/audio/speech" \
    --model "$QWEN_TTS_MODEL" --engine "$QWEN_ARM" \
    --source-revision "$QWEN_REV" \
    --workload "$QWEN_COMPARE_ROOT/evidence/campaign-20261017/plan/workload-128.jsonl" \
    --output "$QWEN_COMPARE_ROOT/runs/reproduce-$QWEN_ARM-round$QWEN_TRIAL/requests-c$concurrency" \
    --concurrency "$concurrency" --requests 128 --warmup "$warmup" \
    --sample-rate 24000 --frame-rate 12.5 --timeout 1800 || exit $?
done
```

`--source-revision` 只是客户端记录的标签，不能替代核对实际服务 checkout。
上述分终端命令便于手动复现，但不会自动记录服务退出状态；正式验收应使用
[tools/qualify.py](tools/qualify.py) 管理服务、健康检查、预热、完整压测与退出码。
它内部同样使用 `max(2, concurrency)`。原三个独立引擎进程的执行记录见
[run-final-candidate.py](evidence/campaign-20261017/plan/run-final-candidate.py)，该历史脚本有本机路径及前置状态检查，不应直接当通用安装脚本执行。

## 与历史脚本的核对及正式复现

本次核对使用九轮 `status.json` 中的服务/压测命令、`provenance.json`、客户端参数及历史 runner。
修正前的文档存在以下差异，表中的历史测量值未因此改写：

| 项目 | 原文档与实测脚本的差异 | 本文修正 |
|---|---|---|
| Python 导入 | 原文档从主仓库运行，可能遮蔽冻结版本；候选实测 runner 指定 `cwd=frozen` | 从冻结目录执行，输入/输出使用绝对路径 |
| 生命周期 | 手动启动仅单次检查 health；实测由 `qualify.py` 等待就绪、管理进程并记录退出码 | 补充下方自动运行命令 |
| 失败处理 | `|| break` 可使失败后的循环返回成功 | 改为 `|| exit $?` |
| 环境记录 | 历史各轮 `PYTHONPATH` 的源码顺序不完全相同 | 复现统一冻结客户端并保留 CANN 路径；历史环境以各轮 status 为准 |

已保存的 [源码等价审计](evidence/campaign-20261017/results/engine-source-equivalence.json)
仅证明当时源码排除模块 docstring 后 AST 一致，不保证未来主分支一致，也不代表重新执行验证。
128 请求、七档并发、每档 `max(2, C)` 预热、24000 Hz PCM、12.5 frames/s 和 1800 秒请求超时与实测一致。

正式复现优先使用下面命令，**不要同时执行手动启动/压测**。
先完成环境准备，选择一个框架；每轮脚本都会重新启动服务并覆盖七档并发。
三个 command JSON 保留原机器绝对路径，要求模型、venv 和 checkout 位于本文约定位置。
SGLang checkout 必须保留已记录的兼容补丁。输出必须为新目录。

```bash
export QWEN_ARM=vllm  # 分别执行 vllm、sglang、independent
export QWEN_REPRO_TAG=verify-20261019
case "$QWEN_ARM" in
  vllm)
    QWEN_COMMAND=vllm-command.json
    QWEN_ENGINE=vllm-omni-main
    QWEN_PORT=8096
    QWEN_REV=b164fea797531c1fa10f8b87829632cad006b9fc
    QWEN_SOURCE="$QWEN_RESEARCH_ROOT/vllm-omni"
    export PYTHONPATH="$QWEN_FROZEN_ROOT:$QWEN_SOURCE:$PYTHONPATH" ;;
  sglang)
    QWEN_COMMAND=sglang-command.json
    QWEN_ENGINE=sglang-omni-main-importcompat
    QWEN_PORT=8098
    QWEN_REV=6bdd47967caa55c38430d18175b750008aecab61
    QWEN_SOURCE="$QWEN_RESEARCH_ROOT/sglang-omni-compat"
    export PYTHONPATH="$QWEN_FROZEN_ROOT:$QWEN_SOURCE:$QWEN_RESEARCH_ROOT/Qwen3-TTS:$QWEN_RESEARCH_ROOT/extra:$PYTHONPATH" ;;
  independent)
    QWEN_COMMAND=continuous-direct-prompt-command.json
    QWEN_ENGINE=standalone-direct-prompt
    QWEN_PORT=8099
    QWEN_REV=e783a64b0fb4f9ade2ffb3980b3e1f74a00c9a6a
    QWEN_SOURCE="$QWEN_FROZEN_ROOT" ;;
  *) echo 'Unknown framework'; exit 1 ;;
esac
cd "$QWEN_FROZEN_ROOT"
for trial in 1 2 3; do
  output="$QWEN_COMPARE_ROOT/runs/$QWEN_REPRO_TAG-$QWEN_ARM-round$trial"
  "$QWEN_BENCH_PYTHON" tools/qualify.py \
    --command-json "$QWEN_COMPARE_ROOT/evidence/campaign-20261017/plan/$QWEN_COMMAND" \
    --output "$output" \
    --workload "$QWEN_COMPARE_ROOT/evidence/campaign-20261017/plan/workload-128.jsonl" \
    --model "$QWEN_TTS_MODEL" --engine "$QWEN_ENGINE" \
    --source-revision "$QWEN_REV" --source-directory "$QWEN_SOURCE" \
    --port "$QWEN_PORT" --requests 128 \
    --concurrency-list 1 2 4 8 16 32 64 --startup-timeout 1800 || exit $?
  "$QWEN_BENCH_PYTHON" - "$output/status.json" <<'PY'
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
assert s["phase"] == "finished" and not s.get("error"), s
assert s["server_exit"] == s["benchmark_exit"] == 0, s
assert [c["concurrency"] for c in s["cohorts"]] == [1, 2, 4, 8, 16, 32, 64]
assert all(c["metrics"]["successful_requests"] == 128
           and c["metrics"]["failed_requests"] == 0 for c in s["cohorts"])
PY
  check_exit=$?
  if [ "$check_exit" -ne 0 ]; then exit "$check_exit"; fi
done
```

这是按历史参数整理的复现入口，不是原九轮调度脚本的逐字副本。
`qualify.py` 只负责性能与生命周期，返回 0 后仍须完成精度评估。
下方手动评估示例的运行路径须替换为自动运行生成的
`runs/$QWEN_REPRO_TAG-<arm>-round<trial>/requests-c<C>`。

## 精度评估和原始结果

结束服务并释放 NPU 0 后，用相同 Whisper 权重逐 cohort 评估。以下示例评估当前框架/轮次的 C64，
需对全部 C1/2/4/8/16/32/64 和三轮分别执行：

```bash
"$QWEN_BENCH_PYTHON" "$QWEN_FROZEN_ROOT/tools/evaluate_audio.py" \
  --run "$QWEN_COMPARE_ROOT/runs/reproduce-$QWEN_ARM-round$QWEN_TRIAL/requests-c64" \
  --asr-model /dataset/models/whisper-large-v3 \
  --output "$QWEN_COMPARE_ROOT/runs/reproduce-$QWEN_ARM-round$QWEN_TRIAL/quality-c64" \
  --device npu:0 --dtype float32
```

ASR 完成不等于全部精度通过；还须使用相同 anchors/verifier 运行 `tools/evaluate_speaker.py`，
并用 `tools/audit_cohort.py` 检查音频、请求记录和评分一致性。

- [每轮吞吐、TTFA、RTF](evidence/campaign-20261017/results/final-three-arm-performance.json)
- [每轮 WER、CER、speaker 分数和验收检查](evidence/campaign-20261017/results/final-numerical-audit.json)
- [九个历史运行目录及选取规则](evidence/campaign-20261017/plan/final-comparison-runs.json)
- [完整质量阈值](evidence/campaign-20261017/plan/quality-gates.json)
- [可下载的逐请求计时、转录和 speaker 记录](evidence/campaign-20261017/final-request-records.tar.gz)

本次仅整理已完成实验，不声称重新运行了上述命令。
