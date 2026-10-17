"""Experimental graph-captured incremental codec with independent request slots."""
from types import SimpleNamespace

import torch
from transformers.cache_utils import Cache

from qwen3_tts_engine.incremental_codec import IncrementalCodec


class SlotHistory:
    def __init__(self, arena, indices):
        self.arena, self.indices = arena, indices

    def get(self, module):
        return self.arena.histories[module].index_select(0, self.indices)

    def __setitem__(self, module, value):
        self.arena.histories[module].index_copy_(0, self.indices, value)


class CodecCache(Cache):
    def __init__(self, arena, indices):
        super().__init__(layers=[])
        self.arena, self.indices = arena, indices

    def update(self, keys, values, layer_idx, cache_kwargs):
        old_keys, old_values = self.arena.keys[layer_idx], self.arena.values[layer_idx]
        keys = torch.cat((old_keys.index_select(0, self.indices), keys), dim=2)
        values = torch.cat((old_values.index_select(0, self.indices), values), dim=2)
        old_keys.index_copy_(0, self.indices, keys[:, :, -self.arena.history:])
        old_values.index_copy_(0, self.indices, values[:, :, -self.arena.history:])
        return keys, values


class CodecSlotGraph:
    def __init__(self, arena, batch, length):
        self.arena = arena
        weight = next(arena.decoder.parameters())
        self.indices = torch.arange(batch, device=weight.device)
        self.codes = torch.zeros(batch, arena.decoder.config.num_quantizers, length,
                                 dtype=torch.long, device=weight.device)
        self.offsets = torch.arange(length, device=weight.device)
        self.key_offsets = torch.arange(-arena.history, length, device=weight.device)
        self.state = SimpleNamespace(history=SlotHistory(arena, self.indices))
        self.cache = CodecCache(arena, self.indices)
        self.graph = None

    def compute(self):
        decoder = self.arena.decoder
        run = self.arena.eager.run_module
        hidden = self.arena.eager.quantizer.decode(self.codes)
        hidden = run(decoder.pre_conv, hidden, self.state).transpose(1, 2)
        transformer = decoder.pre_transformer
        hidden = transformer.input_proj(hidden)
        frames = self.arena.frames.index_select(0, self.indices)
        positions = frames[:, None] + self.offsets[None, :]
        key_positions = frames[:, None] + self.key_offsets[None, :]
        allowed = ((key_positions[:, None, :] >= 0)
                   & (key_positions[:, None, :] <= positions[:, :, None])
                   & (key_positions[:, None, :] > positions[:, :, None] - self.arena.window))
        mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[:, None]
        rotary = transformer.rotary_emb(hidden, positions)
        for layer in transformer.layers:
            hidden = layer(hidden, attention_mask=mask, position_ids=positions,
                           past_key_values=self.cache, use_cache=True,
                           cache_position=positions, position_embeddings=rotary)
        hidden = transformer.output_proj(transformer.norm(hidden)).permute(0, 2, 1)
        for blocks in decoder.upsample:
            for block in blocks:
                hidden = run(block, hidden, self.state)
        for block in decoder.decoder:
            hidden = run(block, hidden, self.state)
        self.arena.frames.index_copy_(0, self.indices, frames + self.codes.shape[-1])
        return hidden.clamp(min=-1, max=1)

    @torch.inference_mode()
    def capture(self):
        saved = [buffer.index_select(0, self.indices).clone() for buffer in self.arena.buffers]
        stream = torch.npu.Stream(device=self.codes.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.output = self.compute()
        for buffer, original in zip(self.arena.buffers, saved):
            buffer.index_copy_(0, self.indices, original)

    @torch.inference_mode()
    def run(self, indices, codes):
        self.indices.copy_(torch.tensor(indices, device=self.indices.device))
        self.codes.copy_(codes)
        self.graph.replay()
        return self.output.clone()


class CodecSlots:
    def __init__(self, decoder, slots, cache_constants=False, triton_snake=False, cache_codebooks=False):
        if slots < 1:
            raise ValueError('Codec slots must be positive')
        self.decoder, self.slots = decoder, slots
        self.window = decoder.config.sliding_window
        if self.window is None or self.window < 2:
            raise ValueError('Expected a finite sliding codec attention window')
        self.history = self.window - 1
        self.eager = IncrementalCodec(decoder, cache_constants=cache_constants, triton_snake=triton_snake, cache_codebooks=cache_codebooks)
        weight = next(decoder.parameters())
        self.histories = {}
        for module in decoder.modules():
            kind = type(module).__name__
            if kind == 'Qwen3TTSTokenizerV2CausalConvNet' and module.padding:
                self.histories[module] = weight.new_zeros(slots, module.conv.in_channels, module.padding)
            elif kind == 'Qwen3TTSTokenizerV2CausalTransConvNet' and module.right_pad:
                self.histories[module] = weight.new_zeros(slots, module.conv.out_channels, module.right_pad)
        c = decoder.config
        shape = (slots, c.num_key_value_heads, self.history, c.head_dim)
        self.keys = [weight.new_zeros(shape) for _ in decoder.pre_transformer.layers]
        self.values = [weight.new_zeros(shape) for _ in decoder.pre_transformer.layers]
        self.frames = torch.zeros(slots, device=weight.device, dtype=torch.long)
        self.buffers = list(self.histories.values()) + self.keys + self.values + [self.frames]
        self.graphs = {}

    @torch.inference_mode()
    def prewarm(self, max_batch):
        """Capture all streaming shapes before readiness without advancing slots."""
        if not 1 <= max_batch <= self.slots:
            raise ValueError('Codec warmup batch must fit the slot arena')
        for batch in range(1, max_batch + 1):
            for length in range(1, 9):
                key = (batch, length)
                if key not in self.graphs:
                    graph = CodecSlotGraph(self, batch, length)
                    graph.capture()  # capture snapshots and restores every touched state buffer
                    self.graphs[key] = graph

    @torch.inference_mode()
    def decode(self, indices, codes):
        if (not indices or len(set(indices)) != len(indices)
                or min(indices) < 0 or max(indices) >= self.slots
                or codes.ndim != 3 or codes.shape[0] != len(indices)
                or codes.shape[1] != self.decoder.config.num_quantizers
                or not 1 <= codes.shape[-1] <= 8):
            raise ValueError('Expected unique valid slots and [batch, codebooks, 1..8] codes')
        key = (len(indices), codes.shape[-1])
        if key not in self.graphs:
            graph = CodecSlotGraph(self, *key)
            graph.capture()
            self.graphs[key] = graph
        return self.graphs[key].run(indices, codes)

    @torch.inference_mode()
    def reset(self, slot):
        for buffer in self.buffers:
            buffer[slot].zero_()

    @torch.inference_mode()
    def move(self, source, destination):
        if source != destination:
            for buffer in self.buffers:
                buffer[destination].copy_(buffer[source])

    def close(self):
        torch.npu.synchronize()
        for graph in self.graphs.values():
            graph.graph.reset()
        self.graphs.clear()
