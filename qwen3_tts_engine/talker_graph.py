"""Experimental single-token talker decoder graph with bounded tensor KV storage."""

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast


class DecodeCache(Cache):
    @property
    def is_compileable(self):
        return False

    def __init__(self, reference, batch, capacity, device, dtype):
        super().__init__(layers=[])
        config = reference.config
        shape = (batch, config.num_key_value_heads, capacity, config.head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in reference.layers]
        self.values = [torch.zeros_like(key) for key in self.keys]
        self.capacity = capacity
        self.length = 0

    def update(self, keys, values, layer_idx, cache_kwargs):
        position = cache_kwargs['cache_position']
        self.keys[layer_idx].index_copy_(2, position, keys)
        self.values[layer_idx].index_copy_(2, position, values)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.length

    def load(self, cache):
        length = cache.get_seq_length()
        if length >= self.capacity:
            raise ValueError('Prefill exceeds decode graph capacity')
        pairs = zip(cache.keys, cache.values) if isinstance(cache, DecodeCache) else (
            (layer.keys, layer.values) for layer in cache.layers)
        for index, (keys, values) in enumerate(pairs):
            self.keys[index].zero_()
            self.values[index].zero_()
            self.keys[index][:, :, :length].copy_(keys[:, :, :length])
            self.values[index][:, :, :length].copy_(values[:, :, :length])
        self.length = length


class TalkerDecodeGraph:
    """A component experiment; callers own prefill and request isolation."""

    def __init__(self, reference, inputs, capacity):
        self.reference = reference
        self.inputs = torch.zeros_like(inputs)
        batch = inputs.shape[0]
        self.positions = torch.zeros((3, batch, 1), device=inputs.device, dtype=torch.long)
        self.cache_position = torch.zeros(1, device=inputs.device, dtype=torch.long)
        self.mask = torch.full((batch, 1, 1, capacity), torch.finfo(inputs.dtype).min,
                               device=inputs.device, dtype=inputs.dtype)
        self.cache = DecodeCache(reference, batch, capacity, inputs.device, inputs.dtype)
        self.graph = None

    def compute(self):
        hidden = self.inputs
        positions = self.reference.rotary_emb(hidden, self.positions)
        all_hidden = []
        for layer in self.reference.layers:
            all_hidden.append(hidden)
            hidden = layer(hidden, attention_mask=self.mask, position_ids=self.positions[0],
                           past_key_values=self.cache, output_attentions=False, use_cache=True,
                           cache_position=self.cache_position, position_embeddings=positions)[0]
        hidden = self.reference.norm(hidden)
        all_hidden.append(hidden)
        return hidden, tuple(all_hidden)

    @torch.inference_mode()
    def capture(self):
        stream = torch.npu.Stream(device=self.inputs.device)
        stream.wait_stream(torch.npu.current_stream())
        # Valid first position for warmup and capture, overwritten by load().
        self.mask[..., 0] = 0
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.output, self.hidden_states = self.compute()

    @torch.inference_mode()
    def decode(self, inputs, position_ids, cache_position, attention_mask):
        if self.graph is None:
            raise RuntimeError('Decode graph has not been captured')
        length = attention_mask.shape[-1]
        if length > self.cache.capacity:
            raise ValueError('Decode graph capacity exceeded')
        self.inputs.copy_(inputs)
        self.positions.copy_(position_ids)
        self.cache_position.copy_(cache_position)
        self.mask.fill_(torch.finfo(inputs.dtype).min)
        self.mask[:, 0, 0, :length].masked_fill_(attention_mask.bool(), 0)
        self.graph.replay()
        self.cache.length = length
        return BaseModelOutputWithPast(last_hidden_state=self.output.clone(), past_key_values=self.cache,
                                       hidden_states=tuple(tensor.clone() for tensor in self.hidden_states))

    def close(self):
        torch.npu.synchronize()
        if self.graph is not None:
            self.graph.reset()
            self.graph = None


class TalkerGraph(torch.nn.Module):
    """Keep eager prefill and reuse bounded decode graphs across requests."""

    def __init__(self, reference):
        super().__init__()
        if reference.config.sliding_window is not None:
            raise ValueError('Sliding-window talker graphs are not supported')
        self.reference = reference
        self.config = reference.config
        self.captures = {}

    def get_input_embeddings(self):
        return self.reference.get_input_embeddings()

    def get_text_embeddings(self):
        return self.reference.get_text_embeddings()

    @torch.inference_mode()
    def forward(self, *, inputs_embeds=None, past_key_values=None, attention_mask=None,
                position_ids=None, cache_position=None, **kwargs):
        if inputs_embeds.shape[1] != 1 or past_key_values is None:
            return self.reference(inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                                  attention_mask=attention_mask, position_ids=position_ids,
                                  cache_position=cache_position, **kwargs)
        if kwargs.get('output_attentions'):
            raise ValueError('Attention weight output is not supported by talker graphs')
        length = attention_mask.shape[-1]
        capacity = max(128, 1 << (length - 1).bit_length())
        key = (inputs_embeds.shape[0], capacity, inputs_embeds.dtype, inputs_embeds.device)
        if key not in self.captures:
            graph = TalkerDecodeGraph(self.reference, inputs_embeds, capacity)
            graph.capture()
            self.captures[key] = graph
        graph = self.captures[key]
        if past_key_values is not graph.cache:
            graph.cache.load(past_key_values)
        return graph.decode(inputs_embeds, position_ids, cache_position, attention_mask)

    def close(self):
        for graph in self.captures.values():
            graph.close()
        self.captures.clear()
