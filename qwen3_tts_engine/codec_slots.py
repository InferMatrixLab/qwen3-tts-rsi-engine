"""Experimental graph-captured incremental codec with independent request slots."""

from types import SimpleNamespace

import torch
from transformers.cache_utils import Cache

from qwen3_tts_engine.incremental_codec import IncrementalCodec


def native_scatter(target, indices, source):
    target.index_copy_(0, indices, source)


class SlotHistory:
    def __init__(self, arena, indices, scatter):
        self.arena, self.indices = arena, indices
        self.scatter = scatter

    def get(self, module):
        return self.arena.histories[module].index_select(0, self.indices)

    def __setitem__(self, module, value):
        self.scatter(self.arena.histories[module], self.indices, value)


class CodecCache(Cache):
    def __init__(self, arena, indices, scatter):
        super().__init__(layers=[])
        self.arena, self.indices = arena, indices
        self.scatter = scatter

    def update(self, keys, values, layer_idx, cache_kwargs):
        old_keys, old_values = self.arena.keys[layer_idx], self.arena.values[layer_idx]
        keys = torch.cat((old_keys.index_select(0, self.indices), keys), dim=2)
        values = torch.cat((old_values.index_select(0, self.indices), values), dim=2)
        self.scatter(old_keys, self.indices, keys[:, :, -self.arena.history :])
        self.scatter(old_values, self.indices, values[:, :, -self.arena.history :])
        return keys, values


class CodecSlotGraph:
    def __init__(self, arena, batch, length, scatter=None):
        self.arena = arena
        self.scatter = arena.scatter if scatter is None else scatter
        weight = next(arena.decoder.parameters())
        self.indices = torch.arange(batch, device=weight.device)
        self.codes = torch.zeros(
            batch,
            arena.decoder.config.num_quantizers,
            length,
            dtype=torch.long,
            device=weight.device,
        )
        self.offsets = torch.arange(length, device=weight.device)
        self.key_offsets = torch.arange(-arena.history, length, device=weight.device)
        self.state = SimpleNamespace(
            history=SlotHistory(arena, self.indices, self.scatter)
        )
        self.cache = CodecCache(arena, self.indices, self.scatter)
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
        allowed = (
            (key_positions[:, None, :] >= 0)
            & (key_positions[:, None, :] <= positions[:, :, None])
            & (key_positions[:, None, :] > positions[:, :, None] - self.arena.window)
        )
        mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(
            hidden.dtype
        )[:, None]
        rotary = transformer.rotary_emb(hidden, positions)
        for layer in transformer.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=positions,
                position_embeddings=rotary,
            )
        hidden = transformer.output_proj(transformer.norm(hidden)).permute(0, 2, 1)
        for blocks in decoder.upsample:
            for block in blocks:
                hidden = run(block, hidden, self.state)
        for block in decoder.decoder:
            hidden = run(block, hidden, self.state)
        self.scatter(self.arena.frames, self.indices, frames + self.codes.shape[-1])
        return hidden.clamp(min=-1, max=1)

    @torch.inference_mode()
    def capture(self, stream=None):
        saved = [
            buffer.index_select(0, self.indices).clone()
            for buffer in self.arena.buffers
        ]
        stream = (
            stream if stream is not None else torch.npu.Stream(device=self.codes.device)
        )
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.output = self.compute()
        torch.npu.current_stream().wait_stream(stream)
        for buffer, original in zip(self.arena.buffers, saved):
            buffer.index_copy_(0, self.indices, original)

    @torch.inference_mode()
    def run(self, indices, codes):
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.indices.device)
        self.indices.copy_(indices)
        self.codes.copy_(codes)
        self.graph.replay()
        return self.output.clone()


class CodecSlots:
    def __init__(
        self,
        decoder,
        slots,
        cache_constants=False,
        triton_snake=False,
        cache_codebooks=False,
        max_chunk_frames=8,
        streams=1,
    ):
        if slots < 1:
            raise ValueError("Codec slots must be positive")
        if max_chunk_frames not in (8, 16):
            raise ValueError("Codec maximum chunk must be 8 or 16 frames")
        if streams not in (1, 2):
            raise ValueError("Codec streams must be 1 or 2")
        self.stream_count = streams
        self.streams = []
        if streams == 2:
            from qwen3_tts_engine.codec_row_scatter import scatter_rows

            self.scatter = scatter_rows
        else:
            self.scatter = native_scatter
        self.max_chunk_frames = max_chunk_frames
        self.decoder, self.slots = decoder, slots
        self.window = decoder.config.sliding_window
        if self.window is None or self.window < 2:
            raise ValueError("Expected a finite sliding codec attention window")
        self.history = self.window - 1
        self.eager = IncrementalCodec(
            decoder,
            cache_constants=cache_constants,
            triton_snake=triton_snake,
            cache_codebooks=cache_codebooks,
        )
        weight = next(decoder.parameters())
        self.histories = {}
        for module in decoder.modules():
            kind = type(module).__name__
            if kind == "Qwen3TTSTokenizerV2CausalConvNet" and module.padding:
                self.histories[module] = weight.new_zeros(
                    slots, module.conv.in_channels, module.padding
                )
            elif kind == "Qwen3TTSTokenizerV2CausalTransConvNet" and module.right_pad:
                self.histories[module] = weight.new_zeros(
                    slots, module.conv.out_channels, module.right_pad
                )
        c = decoder.config
        shape = (slots, c.num_key_value_heads, self.history, c.head_dim)
        self.keys = [weight.new_zeros(shape) for _ in decoder.pre_transformer.layers]
        self.values = [weight.new_zeros(shape) for _ in decoder.pre_transformer.layers]
        self.frames = torch.zeros(slots, device=weight.device, dtype=torch.long)
        self.buffers = (
            list(self.histories.values()) + self.keys + self.values + [self.frames]
        )
        self.graphs = {}
        if streams == 2:
            self.streams = [torch.npu.Stream(device=weight.device) for _ in range(2)]

    def _graph(self, batch, length, lane=0):
        key = (batch, length) if self.stream_count == 1 else (batch, length, lane)
        if key not in self.graphs:
            graph = CodecSlotGraph(self, batch, length)
            try:
                graph.capture(self.streams[lane] if self.streams else None)
            except Exception:
                torch.npu.synchronize()
                if graph.graph is not None:
                    graph.graph.reset()
                raise
            self.graphs[key] = graph
        return self.graphs[key]

    @torch.inference_mode()
    def prewarm(self, max_batch):
        """Capture all streaming shapes before readiness without advancing slots."""
        if not 1 <= max_batch <= self.slots:
            raise ValueError("Codec warmup batch must fit the slot arena")
        for batch in range(1, max_batch + 1):
            for length in range(1, self.max_chunk_frames + 1):
                for lane in range(self.stream_count):
                    self._graph(batch, length, lane)

    @torch.inference_mode()
    def decode(self, indices, codes):
        self._validate(indices, codes)
        return self._graph(len(indices), codes.shape[-1]).run(indices, codes)

    def _validate(self, indices, codes):
        if (
            not indices
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= self.slots
            or codes.ndim != 3
            or codes.shape[0] != len(indices)
            or codes.shape[1] != self.decoder.config.num_quantizers
            or not 1 <= codes.shape[-1] <= self.max_chunk_frames
            or codes.device != self.frames.device
            or codes.dtype != torch.long
        ):
            raise ValueError(
                f"Expected unique valid slots and [batch, codebooks, 1..{self.max_chunk_frames}] codes"
            )

    @torch.inference_mode()
    def decode_groups(self, groups, on_output=None):
        """Complete all disjoint groups before permitting owner-stream state reuse."""
        if not groups:
            return []
        seen = set()
        for indices, codes in groups:
            self._validate(indices, codes)
            if seen.intersection(indices):
                raise ValueError("Codec groups must use disjoint request slots")
            seen.update(indices)
        if self.stream_count == 1:
            outputs = []
            for index, (indices, codes) in enumerate(groups):
                output = self.decode(indices, codes)
                outputs.append(output)
                if on_output is not None:
                    on_output(index, output)
            return outputs
        if len(groups) == 1:
            indices, codes = groups[0]
            output = self._graph(len(indices), codes.shape[-1]).run(indices, codes)
            if on_output is not None:
                on_output(0, output)
            return [output]
        graphs = [
            self._graph(len(indices), codes.shape[-1], i % 2)
            for i, (indices, codes) in enumerate(groups)
        ]
        packed_indices = torch.tensor(
            [slot for indices, _ in groups for slot in indices],
            device=self.frames.device,
        )
        plans = []
        offset = 0
        for i, (indices, codes) in enumerate(groups):
            view = packed_indices[offset : offset + len(indices)]
            plans.append((graphs[i], view, codes, self.streams[i % 2]))
            offset += len(indices)
        producer = torch.npu.current_stream()
        outputs = []
        completion = []
        for stream in self.streams:
            stream.wait_stream(producer)
        try:
            for graph, indices, codes, stream in plans:
                with torch.npu.stream(stream):
                    codes.record_stream(stream)
                    indices.record_stream(stream)
                    outputs.append(graph.run(indices, codes))
                    if on_output is not None:
                        event = torch.npu.Event()
                        event.record()
                        completion.append(event)
            if on_output is not None:
                for index, (output, event) in enumerate(zip(outputs, completion)):
                    producer.wait_event(event)
                    output.record_stream(producer)
                    on_output(index, output)
        finally:
            # Also fence partial submission on exceptions before any caller can reset/move slots.
            for stream in self.streams:
                producer.wait_stream(stream)
        for output in outputs:
            output.record_stream(producer)
        return outputs

    @torch.inference_mode()
    def reset(self, slot):
        for buffer in self.buffers:
            buffer[slot].zero_()

    @torch.inference_mode()
    def reset_range(self, start, count):
        """Clear only newly admitted contiguous slots on the owner stream."""
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 0
            or start + count > self.slots
        ):
            raise ValueError("Expected a valid contiguous Codec slot range")
        if count:
            for buffer in self.buffers:
                buffer.narrow(0, start, count).zero_()

    @torch.inference_mode()
    def move(self, source, destination):
        if source != destination:
            for buffer in self.buffers:
                buffer[destination].copy_(buffer[source])

    def close(self):
        torch.npu.synchronize()
        for graph in self.graphs.values():
            if graph.graph is not None:
                graph.graph.reset()
        self.graphs.clear()
