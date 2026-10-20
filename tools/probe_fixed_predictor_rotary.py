"""Check actual predictor graph replays; not a service performance benchmark."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

from qwen3_tts_engine.continuous import UniformPredictor
from qwen3_tts_engine.model import CustomVoiceModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "INCONCLUSIVE", "scope": "predictor graphs, not service throughput", "cases": []}
    model = None
    try:
        model = CustomVoiceModel(args.model, "npu:0", rms_norm="native", pack_qkv=True,
                                 pack_mlp=True, fused_swiglu=True, native_gqa=True,
                                 native_rope=True, native_rope_scope="predictor")
        predictor = model.wrapper.model.talker.code_predictor
        original_rotary = predictor.model.rotary_emb
        with torch.inference_mode():
            torch.manual_seed(42)
            for graph_embeddings in (False, True):
                control = UniformPredictor(predictor, graph_embeddings, triton_cdf=True)
                candidate = UniformPredictor(predictor, graph_embeddings, triton_cdf=True, fixed_rope=True)
                try:
                    for batch in (1, 4, 16, 32, 64):
                        for replay in range(4):
                            inputs = torch.randn(batch, 2, model.wrapper.model.talker.config.hidden_size,
                                                 device="npu:0", dtype=torch.bfloat16) * .02
                            draws = torch.rand(batch, 16, device="npu:0", dtype=torch.float32)
                            expected = control.run(inputs, draws)
                            actual = candidate.run(inputs, draws)
                            expected = expected if isinstance(expected, tuple) else (expected,)
                            actual = actual if isinstance(actual, tuple) else (actual,)
                            equal = [torch.equal(a, b) for a, b in zip(expected, actual)]
                            assert len(expected) == len(actual) and all(equal)
                            assert predictor.model.rotary_emb is original_rotary
                            result["cases"].append({"batch": batch, "replay": replay,
                                                    "graph_embeddings": graph_embeddings, "outputs_exact": equal})
                        # Returned tensors must not alias replay-owned outputs.
                        owned = [value.clone() for value in actual]
                        candidate.run(inputs * .5, 1 - draws)
                        assert all(torch.equal(a, b) for a, b in zip(owned, actual))
                finally:
                    candidate.close()
                    control.close()
                assert predictor.model.rotary_emb is original_rotary
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
