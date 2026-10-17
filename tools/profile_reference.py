"""Collect an untimed NPU trace separately from benchmark measurements."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu

from qwen3_tts_engine.model import CustomVoiceModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--workload", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--predictor", choices=["reference", "step", "whole"], default="reference")
options = parser.parse_args()
options.output.mkdir(parents=True, exist_ok=False)
payload = json.loads(options.workload.read_text().splitlines()[0])
payload.setdefault("instructions", "")
model = CustomVoiceModel(str(options.model), "npu:0")
graph = None
reference = model.wrapper.model.talker.code_predictor
if options.predictor != "reference":
    from qwen3_tts_engine.predictor import PredictorGraph
    from qwen3_tts_engine.whole_predictor import WholePredictorGraph
    graph = (PredictorGraph if options.predictor == "step" else WholePredictorGraph)(reference)
    model.wrapper.model.talker.code_predictor = graph
try:
    model.generate([payload])
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(options.output)),
    ):
        with torch.profiler.record_function("reference_custom_voice"):
            outputs = model.generate([payload])
    output = outputs[0]
    (options.output / "summary.json").write_text(json.dumps({
        "status": "DIAGNOSTIC_ONLY", "pcm_sha256": hashlib.sha256(output["pcm"]).hexdigest(),
        "pcm_bytes": len(output["pcm"]), "codec_frames": output["codec_frames"],
        "sample_rate": output["sample_rate"], "finish_reason": output["finish_reason"],
        "predictor": options.predictor,
        "note": "Profiler trace is not a performance measurement.",
    }, indent=2) + "\n")
finally:
    if graph is not None:
        graph.close()
        model.wrapper.model.talker.code_predictor = reference
    model.close()
