"""Experimental iteration-level CustomVoice execution with reusable request slots."""
from dataclasses import dataclass, field
import logging
import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicCache
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.codec_graph import CodecGraph
from qwen3_tts_engine.batching import decode_batch_size


from qwen3_tts_engine.sampling import categorical, random_table


class PromptReady(Exception):
    def __init__(self, arguments):
        self.arguments = arguments


class UniformPredictor:
    def __init__(self, reference, graph_embeddings=False):
        self.reference = reference
        self.graph_embeddings = graph_embeddings
        self.captures = {}

    @torch.inference_mode()
    def run(self, embeddings, uniforms):
        batch = embeddings.shape[0]
        if batch not in self.captures:
            inputs = embeddings.clone()
            draws = uniforms.clone()
            positions = [torch.arange(2, device=embeddings.device)] + [
                torch.tensor([i + 1], device=embeddings.device) for i in range(1, 15)]
            mask = torch.zeros(1, 1, 2, 2, device=embeddings.device, dtype=embeddings.dtype)
            mask[..., 0, 1] = torch.finfo(embeddings.dtype).min

            def compute():
                cache = DynamicCache()
                token = None
                codes = []
                for step in range(15):
                    output = self.reference(
                        inputs_embeds=inputs if step == 0 else None, input_ids=token,
                        generation_steps=step, past_key_values=cache, use_cache=True,
                        attention_mask={'full_attention': mask if step == 0 else None},
                        cache_position=positions[step], position_ids=positions[step].unsqueeze(0),
                        output_hidden_states=False, return_dict=True)
                    token = categorical(output.logits[:, -1], draws[:, step + 1])
                    codes.append(token)
                subcodes = torch.cat(codes, dim=-1)
                if self.graph_embeddings:
                    combined = torch.cat([inputs[:, 1:2]] + [
                        self.reference.get_input_embeddings()[i](subcodes[:, i:i+1])
                        for i in range(15)], dim=1).sum(1, keepdim=True)
                    return subcodes, combined
                return subcodes

            stream = torch.npu.Stream(device=embeddings.device)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    compute()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                output = compute()
            self.captures[batch] = (graph, inputs, draws, output, positions, mask)
        graph, inputs, draws, output, *_ = self.captures[batch]
        inputs.copy_(embeddings)
        draws.copy_(uniforms)
        graph.replay()
        return tuple(value.clone() for value in output) if self.graph_embeddings else output.clone()

    def close(self):
        torch.npu.synchronize()
        for graph, *_ in self.captures.values():
            graph.reset()
        self.captures.clear()


class SlotArena:
    def __init__(self, reference, slots, capacity, native_kv_update=False, triton_kv_update=False):
        self.reference = reference
        self.slots = slots
        self.capacity = capacity
        self.native_kv_update = native_kv_update
        self.triton_kv_update = triton_kv_update
        weight = next(reference.parameters())
        c = reference.config
        shape = (slots, c.num_key_value_heads, capacity, c.head_dim)
        self.keys = [torch.zeros(shape, device=weight.device, dtype=weight.dtype) for _ in reference.layers]
        self.values = [torch.zeros_like(key) for key in self.keys]

    def load(self, slot, cache, row=0, length=None):
        length = cache.get_seq_length() if length is None else length
        for i, layer in enumerate(cache.layers):
            self.keys[i][slot, :, :length].copy_(layer.keys[row, :, :length])
            self.values[i][slot, :, :length].copy_(layer.values[row, :, :length])

    def move(self, source, destination):
        if source != destination:
            for keys, values in zip(self.keys, self.values):
                keys[destination].copy_(keys[source])
                values[destination].copy_(values[source])


class SlotCache(Cache):
    def __init__(self, arena, positions, context_length=None):
        super().__init__(layers=[])
        self.arena = arena
        self.positions = positions
        self.context_length = arena.capacity if context_length is None else context_length
        if arena.native_kv_update:
            batch, heads = positions.shape[0], arena.keys[0].shape[1]
            self.rows = torch.arange(batch, device=positions.device)[:, None].expand(batch, heads)
            self.columns = torch.arange(heads, device=positions.device)[None, :].expand(batch, heads)

    def prepare(self):
        if self.arena.native_kv_update and not self.arena.triton_kv_update:
            self.indices = torch.stack((self.rows, self.columns,
                                        self.positions[:, None].expand_as(self.rows)), -1).reshape(-1, 3)

    def update(self, keys, values, layer_idx, cache_kwargs):
        batch, heads, _, width = keys.shape
        index = self.positions.view(batch, 1, 1, 1).expand(batch, heads, 1, width)
        target_keys = self.arena.keys[layer_idx][:batch]
        target_values = self.arena.values[layer_idx][:batch]
        if self.arena.triton_kv_update:
            from qwen3_tts_engine.kv_triton_ascend import write
            write(keys, values, target_keys, target_values, self.positions, row_tiled=True)
        elif self.arena.native_kv_update:
            import torch_npu
            torch_npu.npu_scatter_nd_update_(target_keys, self.indices, keys.reshape(-1, width))
            torch_npu.npu_scatter_nd_update_(target_values, self.indices, values.reshape(-1, width))
        else:
            target_keys.scatter_(2, index, keys)
            target_values.scatter_(2, index, values)
        return target_keys[:, :, :self.context_length], target_values[:, :, :self.context_length]


class SlotGraph:
    def __init__(self, arena, batch, context_length=None):
        self.arena = arena
        self.batch = batch
        self.context_length = arena.capacity if context_length is None else context_length
        if not 1 <= self.context_length <= arena.capacity:
            raise ValueError("Attention context must fit the KV arena")
        self.reference = arena.reference
        weight = next(self.reference.parameters())
        self.inputs = torch.zeros(batch, 1, self.reference.config.hidden_size, device=weight.device, dtype=weight.dtype)
        self.positions = torch.zeros(batch, device=weight.device, dtype=torch.long)
        self.rotary = torch.zeros(3, batch, 1, device=weight.device, dtype=torch.long)
        self.indices = torch.arange(self.context_length, device=weight.device)
        self.cache = SlotCache(arena, self.positions, self.context_length)
        self.graph = None

    def compute(self):
        self.cache.prepare()
        hidden = self.inputs
        allowed = self.indices.view(1, -1) <= self.positions.view(-1, 1)
        mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[:, None, None, :]
        rotary = self.reference.rotary_emb(hidden, self.rotary)
        for layer in self.reference.layers:
            hidden = layer(hidden, attention_mask=mask, position_ids=self.rotary[0],
                           past_key_values=self.cache, output_attentions=False, use_cache=True,
                           cache_position=self.positions, position_embeddings=rotary)[0]
        return self.reference.norm(hidden)

    @torch.inference_mode()
    def capture(self):
        saved = [(k[:self.batch, :, 0].clone(), v[:self.batch, :, 0].clone())
                 for k, v in zip(self.arena.keys, self.arena.values)]
        stream = torch.npu.Stream(device=self.inputs.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.output = self.compute()
        for layer, (keys, values) in enumerate(saved):
            self.arena.keys[layer][:self.batch, :, 0].copy_(keys)
            self.arena.values[layer][:self.batch, :, 0].copy_(values)

    @torch.inference_mode()
    def run(self, inputs, positions, rotary):
        self.inputs.copy_(inputs)
        self.positions.copy_(positions)
        self.rotary.copy_(rotary)
        self.graph.replay()
        return self.output.clone()

    def close(self):
        torch.npu.synchronize()
        if self.graph is not None:
            self.graph.reset()


@dataclass
class RequestState:
    identifier: int
    payload: dict
    hidden: torch.Tensor
    token: torch.Tensor
    trailing: torch.Tensor
    pad: torch.Tensor
    uniforms: torch.Tensor
    seen: torch.Tensor
    position: int
    rope_delta: int
    frames: int = 0
    emitted: int = 0
    codes: list = field(default_factory=list)
    audio: list = field(default_factory=list)


class ContinuousModel:
    def __init__(self, model_path, device, max_slots=64, initial_capacity=512,
                 rms_norm=None, pack_qkv=False, codec_left_context=72, codec_batch_size=4,
                 incremental_codec=False, batch_prefill=False, pack_mlp=False, fused_swiglu=False,
                 native_kv_update=False, native_gqa=False, graph_prefill=False, text_only_rope=False, prewarm_codec=False, bucket_kv=False, graph_embeddings=False, bucket_prefill=False, decode_bucket_step=0, batch_audio_transfer=False, reference_single=False, graph_sampling=False, native_rope=False, fused_residual=False, cache_codec_constants=False, codec_code_buffer=False, triton_snake=False, native_rope_scope="all", cache_codec_codebooks=False, prewarm_prefill=False, triton_kv_update=False, triton_talker_rope=False, codec_chunk_frames=8):
        if codec_chunk_frames not in (8, 16) or (codec_chunk_frames != 8 and (not incremental_codec or codec_code_buffer)):
            raise ValueError("16-frame chunks require incremental codec without the 8-frame code buffer")
        self.codec_chunk_frames = codec_chunk_frames
        if triton_kv_update and not native_kv_update:
            raise ValueError("Triton KV update requires native KV update")
        decode_batch_size(1, max_slots, decode_bucket_step)
        self.decode_bucket_step = decode_bucket_step
        self.batch_audio_transfer = batch_audio_transfer
        if codec_code_buffer and not incremental_codec:
            raise ValueError("Codec code buffer requires incremental codec")
        self.codec_codes = None
        self.reference_single = reference_single
        self.graph_sampling = graph_sampling
        self.samplers = {}
        if bucket_prefill and not graph_prefill:
            raise ValueError('Prefill batch bucketing requires graph prefill')
        if prewarm_prefill and not graph_prefill:
            raise ValueError('Prefill prewarming requires graph prefill')
        if prewarm_prefill and reference_single:
            raise ValueError('Prefill prewarming does not support reference-single dispatch')
        if prewarm_codec and not incremental_codec:
            raise ValueError('Codec prewarming requires the incremental codec')
        self.model = CustomVoiceModel(model_path, device, rms_norm=rms_norm, pack_qkv=pack_qkv,
                                     pack_mlp=pack_mlp, fused_swiglu=fused_swiglu, native_gqa=native_gqa, text_only_rope=text_only_rope, reference_single=reference_single, native_rope=native_rope, fused_residual=fused_residual, native_rope_scope=native_rope_scope, triton_talker_rope=triton_talker_rope)
        self.device = device
        self.speakers = self.model.speakers
        self.languages = self.model.languages
        self.talker = self.model.wrapper.model.talker
        self.max_slots = max_slots
        self.bucket_kv = bucket_kv
        self.batch_prefill = batch_prefill
        self.graph_prefill = graph_prefill
        self.bucket_prefill = bucket_prefill
        self.prefills = {}
        if max_slots < 1 or max_slots & (max_slots - 1):
            raise ValueError('Continuous slot count must be a positive power of two')
        self.arena = SlotArena(self.talker.model, max_slots, initial_capacity, native_kv_update, triton_kv_update)
        self.graphs = {}
        self.graph_embeddings = graph_embeddings
        self.predictor = UniformPredictor(self.talker.code_predictor, graph_embeddings)
        decoder = self.model.wrapper.model.speech_tokenizer.model.decoder
        self.incremental_codec = incremental_codec
        if incremental_codec:
            from qwen3_tts_engine.codec_slots import CodecSlots
            self.codec = CodecSlots(decoder, max_slots, cache_constants=cache_codec_constants, triton_snake=triton_snake, cache_codebooks=cache_codec_codebooks, max_chunk_frames=codec_chunk_frames)
            if codec_code_buffer:
                from qwen3_tts_engine.codec_code_buffer import CodecCodeBuffer
                self.codec_codes = CodecCodeBuffer(max_slots, decoder.config.num_quantizers, device)
            if prewarm_codec:
                self.codec.prewarm(min(codec_batch_size, max_slots))
        else:
            self.codec = CodecGraph(decoder)
        self.context = codec_left_context
        self.codec_batch_size = codec_batch_size
        self.states = []
        self.eos = self.talker.config.codec_eos_token_id
        self.allowed = torch.ones(self.talker.config.vocab_size, device=device, dtype=torch.bool)
        self.allowed[-1024:] = False
        self.allowed[self.eos] = True
        if graph_sampling:
            from qwen3_tts_engine.sampling_graph import SamplingGraph
            for batch in range(1, max_slots + 1):
                graph = SamplingGraph(self.semantic, batch, self.talker.config.vocab_size, device)
                graph.capture()
                self.samplers[batch] = graph
        if prewarm_prefill:
            from qwen3_tts_engine.prefill_graph import PrefillGraph
            # Bound startup memory. These generic short-prompt shapes cover
            # small replacement batches; all other shapes retain lazy capture.
            for batch in range(1, min(max_slots, 4) + 1):
                if bucket_prefill and batch & (batch - 1):
                    continue
                for length in (32, 64, 128):
                    graph = PrefillGraph(self.talker, batch, length)
                    graph.capture()
                    self.prefills[batch, length] = graph

    def ensure_capacity(self, required):
        if required < self.arena.capacity:
            return
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        old = self.arena
        self.arena = SlotArena(self.talker.model, self.max_slots, max(old.capacity * 2, 1 << required.bit_length()),
                               old.native_kv_update, old.triton_kv_update)
        for target, source in zip(self.arena.keys + self.arena.values, old.keys + old.values):
            target[:, :, :old.capacity].copy_(source)

    def semantic(self, logits, uniforms, seen, steps):
        logits = logits.float()
        logits = torch.where(seen, torch.where(logits < 0, logits * 1.05, logits / 1.05), logits)
        logits = logits.masked_fill(~self.allowed, -float('inf'))
        logits[:, self.eos] = torch.where(steps < 2, -float('inf'), logits[:, self.eos])
        return categorical(logits, uniforms)

    def prepare_prompt(self, payload):
        original = self.talker.generate
        def capture(**kwargs):
            raise PromptReady(kwargs)
        self.talker.generate = capture
        try:
            self.model.wrapper.generate_custom_voice(
                text=payload['input'], speaker=payload['voice'], language=payload['language'],
                instruct=payload['instructions'], max_new_tokens=payload['max_new_tokens'], non_streaming_mode=True)
        except PromptReady as captured:
            prompt = captured.arguments
        finally:
            self.talker.generate = original
        supported = {'do_sample': True, 'top_k': 50, 'top_p': 1.0, 'temperature': 0.9,
                     'repetition_penalty': 1.05, 'subtalker_dosample': True,
                     'subtalker_top_k': 50, 'subtalker_top_p': 1.0, 'subtalker_temperature': 0.9}
        for key, value in supported.items():
            if prompt.get(key) != value:
                raise ValueError(f'Unsupported native sampling setting: {key}={prompt.get(key)}')
        return prompt

    @torch.inference_mode()
    def admit(self, identifier, payload):
        if len(self.states) >= self.max_slots:
            raise ValueError('No free continuous request slot')
        prompt = self.prepare_prompt(payload)
        prepared = {key: prompt[key] for key in ('inputs_embeds', 'attention_mask', 'trailing_text_hidden', 'tts_pad_embed')}
        output = self.talker(**prepared, use_cache=True, output_hidden_states=True, return_dict=True)
        position = output.past_key_values.get_seq_length()
        self.ensure_capacity(position)
        self.arena.load(len(self.states), output.past_key_values)
        self.add_state(identifier, payload, prompt, output.past_hidden, output.logits[:, -1],
                       position, int(self.talker.rope_deltas[0, 0].item()))

    def add_state(self, identifier, payload, prompt, hidden, logits, position, rope_delta):
        uniforms = random_table(payload['seed'], payload['max_new_tokens']).to(self.device)
        seen = torch.zeros(1, self.talker.config.vocab_size, device=self.device, dtype=torch.bool)
        token = self.semantic(logits, uniforms[0, 0:1], seen,
                              torch.zeros(1, device=self.device, dtype=torch.long))
        seen.scatter_(1, token, True)
        if self.incremental_codec:
            self.codec.reset(len(self.states))
        self.states.append(RequestState(identifier, payload, hidden, token,
                                        prompt['trailing_text_hidden'], prompt['tts_pad_embed'], uniforms, seen,
                                        position, rope_delta))

    @torch.inference_mode()
    def admit_batch(self, submissions):
        if self.reference_single and len(submissions) == 1 and not self.states:
            self.admit(*submissions[0])
            return
        if len(self.states) + len(submissions) > self.max_slots:
            raise ValueError('Insufficient continuous request slots')
        prompts = [self.prepare_prompt(payload) for _, payload in submissions]
        lengths = [prompt['inputs_embeds'].shape[1] for prompt in prompts]
        longest = max(lengths)
        padded_length = 1 << (longest - 1).bit_length() if self.graph_prefill else longest
        if not all(bool(prompt['attention_mask'].all()) for prompt in prompts):
            raise ValueError('Batched prefill requires individually unpadded prompts')
        embeddings = torch.cat([torch.nn.functional.pad(prompt['inputs_embeds'], (0, 0, 0, padded_length - length))
                                for prompt, length in zip(prompts, lengths)])
        mask = torch.cat([torch.nn.functional.pad(prompt['attention_mask'], (0, padded_length - length))
                          for prompt, length in zip(prompts, lengths)])
        if self.graph_prefill:
            from qwen3_tts_engine.prefill_graph import PrefillGraph, pad_prefill_batch
            graph_batch = 1 << (len(submissions) - 1).bit_length() if self.bucket_prefill else len(submissions)
            embeddings, mask = pad_prefill_batch(embeddings, mask, graph_batch)
            key = (graph_batch, padded_length)
            if key not in self.prefills:
                graph = PrefillGraph(self.talker, *key)
                graph.capture()
                self.prefills[key] = graph
            hidden, logits, cache = self.prefills[key].run(embeddings, mask)
        else:
            positions, _ = self.talker.get_rope_index(mask)
            output = self.talker.model(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                                       use_cache=True, output_hidden_states=False, return_dict=True)
            rows = torch.arange(len(submissions), device=self.device)
            ends = torch.tensor(lengths, device=self.device) - 1
            hidden = output.last_hidden_state[rows, ends].unsqueeze(1)
            logits = self.talker.codec_head(hidden)[:, -1]
            cache = output.past_key_values
        self.ensure_capacity(longest)
        for row, ((identifier, payload), prompt, length) in enumerate(zip(submissions, prompts, lengths)):
            # Crop padded KV entries; subsequent decode uses each original length
            # and the unpadded rotary origin, rather than the padded batch width.
            self.arena.load(len(self.states), cache, row=row, length=length)
            self.add_state(identifier, payload, prompt, hidden[row:row + 1], logits[row:row + 1], length, 0)

    def emit_audio(self, selected, emit):
        groups = {}
        slots = {state.identifier: index for index, state in enumerate(self.states)}
        for state in selected:
            if state.frames == state.emitted:
                continue
            start = state.emitted if self.incremental_codec else max(0, state.emitted - self.context)
            length = state.frames - start
            codes = None if self.codec_codes is not None else torch.stack(state.codes[start:]).transpose(0, 1)
            groups.setdefault(length, []).append((state, codes, (state.emitted - start) * 1920))
        pending = []
        for length, entries in groups.items():
            for offset in range(0, len(entries), self.codec_batch_size):
                group = entries[offset:offset + self.codec_batch_size]
                batch = 1 << (len(group) - 1).bit_length()
                codes = [entry[1] for entry in group]
                if self.incremental_codec:
                    indices = [slots[entry[0].identifier] for entry in group]
                    packed = (self.codec_codes.read(indices, [entry[0].emitted for entry in group], length)
                              if self.codec_codes is not None else torch.stack(codes))
                    waves = self.codec.decode(indices, packed)
                else:
                    codes += [codes[-1]] * (batch - len(codes))
                    waves = self.codec.decode(torch.stack(codes))
                if self.batch_audio_transfer:
                    # Each decoder returns an owned clone, so subsequent replays
                    # cannot overwrite queued waveforms before the single transfer.
                    pending.append((group, waves[:len(group), 0]))
                else:
                    self.deliver_audio(group, waves[:, 0].float().cpu().numpy(), emit)
        if pending:
            lengths = [waves.numel() for _, waves in pending]
            audio = torch.cat([waves.reshape(-1) for _, waves in pending]).float().cpu().numpy()
            offset = 0
            for (group, waves), length in zip(pending, lengths):
                self.deliver_audio(group, audio[offset:offset + length].reshape(waves.shape), emit)
                offset += length

    @staticmethod
    def deliver_audio(group, waves, emit):
        for i, (state, _, skip) in enumerate(group):
            waveform = waves[i, skip:]
            if not np.isfinite(waveform).all():
                raise ValueError('Native codec emitted nonfinite audio')
            pcm = (np.clip(waveform, -1, 1) * 32767).astype('<i2').tobytes()
            state.audio.append(pcm)
            state.emitted = state.frames
            if emit is not None:
                emit(state.identifier, pcm, 24000)

    @torch.inference_mode()
    def tick(self, submissions=(), emit=None, cancelled=()):
        torch.npu.set_device(self.device)
        cancelled = set(cancelled)
        for index in reversed(range(len(self.states))):
            if self.states[index].identifier in cancelled:
                last = len(self.states) - 1
                self.arena.move(last, index)
                if self.codec_codes is not None:
                    self.codec_codes.move(last, index)
                if self.incremental_codec:
                    self.codec.move(last, index)
                if index != last:
                    self.states[index] = self.states[last]
                self.states.pop()
        native_prefill = bool(submissions) and self.reference_single and len(self.states) + len(submissions) > 1
        if native_prefill:
            self.model.set_single_reference(False)
        try:
            if (self.batch_prefill or self.graph_prefill) and submissions:
                self.admit_batch(submissions)
            else:
                for identifier, payload in submissions:
                    self.admit(identifier, payload)
        finally:
            if native_prefill:
                self.model.set_single_reference(True)
        if not self.states:
            return []
        count = len(self.states)
        batch = decode_batch_size(count, self.max_slots, self.decode_bucket_step)
        self.ensure_capacity(max(state.position for state in self.states))
        tokens = torch.cat([state.token for state in self.states])
        hidden = torch.cat([state.hidden for state in self.states])
        draws = torch.stack([state.uniforms[state.frames] for state in self.states])
        if batch > count:
            tokens = torch.cat((tokens, torch.zeros(batch - count, 1, device=self.device, dtype=tokens.dtype)))
            hidden = torch.cat((hidden, torch.zeros(batch - count, 1, hidden.shape[-1], device=self.device, dtype=hidden.dtype)))
            draws = torch.cat((draws, torch.full((batch - count, 16), 0.5, device=self.device)))
        first = self.talker.get_input_embeddings()(tokens)
        predicted = self.predictor.run(torch.cat((hidden, first), dim=1), draws)
        if self.graph_embeddings:
            subcodes, embeds = predicted
        else:
            subcodes = predicted
            embeds = torch.cat([first] + [self.talker.code_predictor.get_input_embeddings()[i](subcodes[:, i:i+1])
                                         for i in range(15)], dim=1).sum(1, keepdim=True)
        codes = torch.cat((tokens, subcodes), dim=-1)
        text = [state.trailing[:, state.frames:state.frames + 1] if state.frames < state.trailing.shape[1]
                else state.pad for state in self.states]
        if batch > count:
            text += [torch.zeros_like(text[0])] * (batch - count)
        embeds = embeds + torch.cat(text)
        positions = torch.tensor([s.position for s in self.states] + [0] * (batch - count), device=self.device)
        rotary = torch.tensor([s.position + s.rope_delta for s in self.states] + [0] * (batch - count),
                              device=self.device).view(1, batch, 1).expand(3, -1, -1)
        required_context = max(s.position for s in self.states) + 1
        context = min(self.arena.capacity, max(64, 1 << (required_context - 1).bit_length())) if self.bucket_kv else self.arena.capacity
        graph_key = (batch, context)
        if graph_key not in self.graphs:
            graph = SlotGraph(self.arena, batch, context)
            graph.capture()
            self.graphs[graph_key] = graph
        hidden = self.graphs[graph_key].run(embeds, positions, rotary)[:count]
        logits = self.talker.codec_head(hidden)[:, -1]
        steps = torch.tensor([s.frames + 1 for s in self.states], device=self.device)
        uniforms = torch.stack([s.uniforms[s.frames + 1, 0] for s in self.states])
        seen = torch.cat([s.seen for s in self.states])
        if self.graph_sampling:
            next_tokens, updated_seen = self.samplers[count].run(logits, uniforms, seen, steps)
        else:
            next_tokens = self.semantic(logits, uniforms, seen, steps)
        if self.codec_codes is not None:
            self.codec_codes.write(codes[:count], steps - 1)
        token_ids = next_tokens[:, 0].tolist()
        due, finished = [], []
        for index, state in enumerate(self.states):
            if self.codec_codes is None:
                state.codes.append(codes[index])
            state.frames += 1
            state.position += 1
            state.hidden = hidden[index:index + 1]
            state.token = next_tokens[index:index + 1]
            if self.graph_sampling:
                state.seen = updated_seen[index:index + 1]
            else:
                state.seen.scatter_(1, state.token, True)
            target = 2 if state.emitted == 0 else state.emitted * 2 if state.emitted < 8 else state.emitted + self.codec_chunk_frames
            stopped = token_ids[index] == self.eos
            limited = state.frames >= state.payload['max_new_tokens'] - 1
            if state.frames >= target or stopped or limited:
                due.append(state)
            if stopped or limited:
                finished.append((index, state, 'stop' if stopped else 'length'))
        self.emit_audio(due, emit)
        results = []
        for index, state, reason in reversed(finished):
            results.append((state.identifier, {'pcm': b''.join(state.audio), 'sample_rate': 24000,
                                               'codec_frames': state.frames, 'finish_reason': reason}))
            last = len(self.states) - 1
            self.arena.move(last, index)
            if self.codec_codes is not None:
                self.codec_codes.move(last, index)
            if self.incremental_codec:
                self.codec.move(last, index)
            if index != last:
                self.states[index] = self.states[last]
            self.states.pop()
        return results

    def close(self):
        for graph in self.samplers.values():
            graph.close()
        self.samplers.clear()
        logging.getLogger(__name__).info("Captured graph shapes: prefill=%s talker=%s codec=%s",
                                         sorted(self.prefills), sorted(self.graphs), sorted(self.codec.graphs) if self.incremental_codec else list(self.codec.captures))
        for graph in self.prefills.values():
            graph.close()
        self.prefills.clear()
        for graph in self.graphs.values():
            graph.close()
        self.predictor.close()
        self.codec.close()
        self.model.close()
