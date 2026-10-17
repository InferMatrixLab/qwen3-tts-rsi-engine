"""Capture predictor compute while keeping random sampling outside graphs."""

from types import SimpleNamespace

import torch
from transformers.cache_utils import Cache


class PredictorCache(Cache):
    def __init__(self, reference, inputs):
        super().__init__(layers=[])
        config = reference.config
        shape = (inputs.shape[0], config.num_key_value_heads, 16, config.head_dim)
        self.keys = [torch.zeros(shape, device=inputs.device, dtype=inputs.dtype)
                     for _ in range(config.num_hidden_layers)]
        self.values = [torch.zeros_like(tensor) for tensor in self.keys]
        self.step = 0

    def update(self, keys, values, layer_index, cache_kwargs):
        start = 0 if self.step == 0 else self.step + 1
        end = self.step + 2
        self.keys[layer_index][:, :, start:end].copy_(keys)
        self.values[layer_index][:, :, start:end].copy_(values)
        return self.keys[layer_index][:, :, :end], self.values[layer_index][:, :, :end]

    def get_seq_length(self, layer_idx=0):
        return 0 if self.step == 0 else self.step + 1


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
        key = (tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device)
        if key not in self.captures:
            self.captures[key] = self.capture(inputs_embeds)
        graphs, static_input, token_input, logits, cache = self.captures[key]
        static_input.copy_(inputs_embeds)
        codes = []
        for step, graph in enumerate(graphs):
            if step:
                token_input.copy_(codes[-1])
            graph.replay()
            scores = logits[step][:, -1].float()
            if do_sample:
                scores = scores / temperature
                threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
                scores = scores.masked_fill(scores < threshold, -float("inf"))
                token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
            else:
                token = scores.argmax(dim=-1, keepdim=True)
            codes.append(token)
        return SimpleNamespace(sequences=torch.cat(codes, dim=-1))

    def capture(self, inputs):
        static_input = inputs.clone()
        token_input = torch.zeros((inputs.shape[0], 1), device=inputs.device, dtype=torch.long)
        positions = [torch.arange(2, device=inputs.device)] + [
            torch.tensor([i + 1], device=inputs.device) for i in range(1, 15)
        ]
        mask = torch.zeros(1, 1, 2, 2, dtype=inputs.dtype, device=inputs.device)
        mask[0, 0, 0, 1] = torch.finfo(inputs.dtype).min
        cache = PredictorCache(self.reference, inputs)
        # Captured kernels retain addresses, not Python tensor ownership. Keep
        # constants alive for every replay; otherwise the allocator may reuse
        # position or mask storage after capture() returns.
        cache.graph_constants = (positions, mask)
        stream = torch.npu.Stream(device=inputs.device)
        stream.wait_stream(torch.npu.current_stream())
        graphs, logits = [], []
        for step in range(15):
            cache.step = step

            def decode():
                return self.reference(
                    inputs_embeds=static_input if step == 0 else None,
                    input_ids=token_input if step else None, generation_steps=step,
                    past_key_values=cache, use_cache=True,
                    attention_mask={"full_attention": mask if step == 0 else None},
                    cache_position=positions[step], position_ids=positions[step].unsqueeze(0),
                    output_hidden_states=False, return_dict=True,
                ).logits

            with torch.npu.stream(stream):
                for _ in range(3):
                    decode()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                logits.append(decode())
            graphs.append(graph)
        return graphs, static_input, token_input, logits, cache

    def close(self):
        torch.npu.synchronize()
        for graphs, *_ in self.captures.values():
            for graph in graphs:
                graph.reset()
        self.captures.clear()
