"""Compare sequential sampled predictor calls and interleaved talker RNG draws."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.predictor import PredictorGraph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--whole", action="store_true")
options = parser.parse_args()
model = CustomVoiceModel(str(options.model), "npu:0")
reference = model.wrapper.model.talker.code_predictor
candidate = PredictorGraph(reference)
if options.whole:
    from qwen3_tts_engine.whole_predictor import WholePredictorGraph
    candidate = WholePredictorGraph(reference)
result = {"status": "INCONCLUSIVE", "scope": "sampled sequential component parity", "batch_size": options.batch_size}
result["implementation"] = type(candidate).__name__
try:
    with torch.inference_mode():
        torch.npu.manual_seed_all(1234)
        inputs = [torch.randn(options.batch_size, 2, 2048, device="npu", dtype=torch.bfloat16) for _ in range(4)]
        result["input_seed"] = 1234
        probabilities = torch.ones(options.batch_size, 3072, device="npu") / 3072
        arms = {}
        for name, predictor in [("reference", reference), ("candidate", candidate)]:
            torch.npu.manual_seed_all(42)
            rows = []
            for embeddings in inputs:
                codes = predictor.generate(
                    inputs_embeds=embeddings, max_new_tokens=15, do_sample=True,
                    top_k=50, top_p=1.0, temperature=0.9,
                    output_hidden_states=True, return_dict_in_generate=True,
                ).sequences
                outer_token = torch.multinomial(probabilities, 1)
                torch.npu.synchronize()
                rng = torch.npu.get_rng_state().cpu().numpy().tobytes()
                rows.append({"codes": codes.cpu().tolist(), "outer_token": outer_token.cpu().tolist(),
                             "rng_sha256": hashlib.sha256(rng).hexdigest()})
            arms[name] = rows
        result["arms"] = arms
        result["exact"] = arms["reference"] == arms["candidate"]
        result["status"] = "PASS" if result["exact"] else "FAIL"
        if not result["exact"]:
            raise ValueError("Sequential predictor outputs or RNG progression differ")
except Exception as error:
    result["status"] = "FAIL"
    result["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    candidate.close()
    model.close()
