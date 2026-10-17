"""Exact greedy-code parity probe for an entire 15-codebook NPU graph."""

import argparse
import json
from pathlib import Path
import time

import torch
import torch_npu
from transformers.cache_utils import DynamicCache

from qwen3_tts_engine.model import CustomVoiceModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--sample", action="store_true")
options = parser.parse_args()
torch.manual_seed(42)
model = CustomVoiceModel(str(options.model), "npu:0")
predictor = model.wrapper.model.talker.code_predictor
batch_size = options.batch_size
embeddings = torch.randn(batch_size, 2, 2048, device="npu", dtype=torch.bfloat16)
positions = [torch.arange(2, device="npu")] + [torch.tensor([i + 1], device="npu") for i in range(1, 15)]
first_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16, device="npu")
first_mask[0, 0, 0, 1] = torch.finfo(torch.bfloat16).min


def manual(inputs):
    cache = DynamicCache()
    codes = []
    token = None
    for step in range(15):
        output = predictor(
            inputs_embeds=inputs if step == 0 else None,
            input_ids=token, generation_steps=step,
            past_key_values=cache, use_cache=True,
            attention_mask={"full_attention": first_mask if step == 0 else None},
            cache_position=positions[step], position_ids=positions[step].unsqueeze(0),
            output_hidden_states=False, return_dict=True,
        )
        scores = output.logits[:, -1].float()
        if options.sample:
            scores = scores / 0.9
            threshold = torch.topk(scores, 50, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, -float("inf"))
            token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
        else:
            token = scores.argmax(dim=-1, keepdim=True)
        codes.append(token)
    return torch.cat(codes, dim=-1)


sampling = {"do_sample": options.sample, "top_k": 50, "top_p": 1.0, "temperature": 0.9}
result = {"status": "INCONCLUSIVE", "scope": "predictor component; not service performance", "sampling": sampling,
          "batch_size": batch_size, "parity": []}
try:
    with torch.inference_mode():
        torch.npu.manual_seed_all(42)
        reference = predictor.generate(inputs_embeds=embeddings, max_new_tokens=15,
                                       **sampling, return_dict_in_generate=True).sequences
        torch.npu.manual_seed_all(42)
        eager = manual(embeddings)
        result["eager_exact"] = bool(torch.equal(reference, eager))
        if not result["eager_exact"]:
            raise ValueError("Manual predictor differs from official greedy generation")
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                manual(embeddings)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        static_input = embeddings.clone()
        with torch.npu.graph(graph, stream=stream):
            static_output = manual(static_input)
        for index in range(3):
            current = torch.randn_like(embeddings)
            torch.npu.manual_seed_all(100 + index)
            expected = predictor.generate(inputs_embeds=current, max_new_tokens=15,
                                          **sampling, return_dict_in_generate=True).sequences
            static_input.copy_(current)
            torch.npu.manual_seed_all(100 + index)
            graph.replay()
            torch.npu.synchronize()
            exact = bool(torch.equal(expected, static_output))
            result["parity"].append({"input": index, "exact": exact})
            if not exact:
                raise ValueError("Graph replay differs from official greedy generation")
        timings = {}
        for name, operation in [("manual_eager", lambda: manual(static_input)), ("graph", graph.replay)]:
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                operation()
            torch.npu.synchronize()
            timings[name + "_ms"] = (time.perf_counter() - started) * 1000 / 20
        result["component_timings"] = timings
        result["status"] = "PASS"
        graph.reset()
except Exception as error:
    result["status"] = "FAIL"
    result["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    model.close()
