"""Capture the fixed-depth codec predictor without changing its sampler."""

from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache


class PredictorGraph(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.captures = {}
        self.fallbacks = 0

    def get_input_embeddings(self):
        return self.reference.get_input_embeddings()

    def forward(self, *args, **kwargs):
        return self.reference(*args, **kwargs)

    @torch.inference_mode()
    def generate(self, *, inputs_embeds, max_new_tokens, do_sample, top_p,
                 top_k, temperature, output_hidden_states, return_dict_in_generate):
        if (max_new_tokens != 15 or top_p != 1.0 or top_k != 50
                or temperature != 0.9 or not return_dict_in_generate):
            self.fallbacks += 1
            return self.reference.generate(
                inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
            )
        key = (tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device, do_sample)
        if key not in self.captures:
            self.captures[key] = self.capture(inputs_embeds, do_sample)
        graph, static_input, static_output = self.captures[key]
        static_input.copy_(inputs_embeds)
        graph.replay()
        return SimpleNamespace(sequences=static_output.clone())

    def capture(self, inputs, do_sample):
        static_input = inputs.clone()
        positions = [torch.arange(2, device=inputs.device)] + [
            torch.tensor([i + 1], device=inputs.device) for i in range(1, 15)
        ]
        mask = torch.zeros(1, 1, 2, 2, dtype=inputs.dtype, device=inputs.device)
        mask[0, 0, 0, 1] = torch.finfo(inputs.dtype).min

        def decode():
            cache = DynamicCache()
            codes = []
            token = None
            for step in range(15):
                output = self.reference(
                    inputs_embeds=static_input if step == 0 else None,
                    input_ids=token, generation_steps=step,
                    past_key_values=cache, use_cache=True,
                    attention_mask={"full_attention": mask if step == 0 else None},
                    cache_position=positions[step], position_ids=positions[step].unsqueeze(0),
                    output_hidden_states=False, return_dict=True,
                )
                scores = output.logits[:, -1].float()
                if do_sample:
                    scores = scores / 0.9
                    threshold = torch.topk(scores, 50, dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -float("inf"))
                    token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
                else:
                    token = scores.argmax(dim=-1, keepdim=True)
                codes.append(token)
            return torch.cat(codes, dim=-1)

        rng = torch.npu.get_rng_state(inputs.device)
        stream = torch.npu.Stream(device=inputs.device)
        stream.wait_stream(torch.npu.current_stream())
        try:
            with torch.npu.stream(stream):
                for _ in range(3):
                    decode()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                static_output = decode()
        finally:
            torch.npu.set_rng_state(rng, inputs.device)
        return graph, static_input, static_output

    def close(self):
        torch.npu.synchronize()
        for graph, _, _ in self.captures.values():
            graph.reset()
        self.captures.clear()
