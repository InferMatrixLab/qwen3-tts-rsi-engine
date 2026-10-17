"""Experimental iteration-level CustomVoice execution with reusable request slots."""
from dataclasses import dataclass, field
import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicCache
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.codec_graph import CodecGraph


from qwen3_tts_engine.sampling import categorical, random_table


class PromptReady(Exception):
    def __init__(self, arguments):
        self.arguments = arguments


class UniformPredictor:
    def __init__(self, reference):
        self.reference = reference
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
                return torch.cat(codes, dim=-1)

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
        return output.clone()

    def close(self):
        torch.npu.synchronize()
        for graph, *_ in self.captures.values():
            graph.reset()
        self.captures.clear()


class SlotArena:
    def __init__(self, reference, slots, capacity):
        self.reference = reference
        self.slots = slots
        self.capacity = capacity
        weight = next(reference.parameters())
        c = reference.config
        shape = (slots, c.num_key_value_heads, capacity, c.head_dim)
        self.keys = [torch.zeros(shape, device=weight.device, dtype=weight.dtype) for _ in reference.layers]
        self.values = [torch.zeros_like(key) for key in self.keys]

    def load(self, slot, cache):
        length = cache.get_seq_length()
        for i, layer in enumerate(cache.layers):
            self.keys[i][slot, :, :length].copy_(layer.keys[0])
            self.values[i][slot, :, :length].copy_(layer.values[0])

    def move(self, source, destination):
        if source != destination:
            for keys, values in zip(self.keys, self.values):
                keys[destination].copy_(keys[source])
                values[destination].copy_(values[source])


class SlotCache(Cache):
    def __init__(self, arena, positions):
        super().__init__(layers=[])
        self.arena = arena
        self.positions = positions

    def update(self, keys, values, layer_idx, cache_kwargs):
        batch, heads, _, width = keys.shape
        index = self.positions.view(batch, 1, 1, 1).expand(batch, heads, 1, width)
        target_keys = self.arena.keys[layer_idx][:batch]
        target_values = self.arena.values[layer_idx][:batch]
        target_keys.scatter_(2, index, keys)
        target_values.scatter_(2, index, values)
        return target_keys, target_values


class SlotGraph:
    def __init__(self, arena, batch):
        self.arena = arena
        self.batch = batch
        self.reference = arena.reference
        weight = next(self.reference.parameters())
        self.inputs = torch.zeros(batch, 1, self.reference.config.hidden_size, device=weight.device, dtype=weight.dtype)
        self.positions = torch.zeros(batch, device=weight.device, dtype=torch.long)
        self.rotary = torch.zeros(3, batch, 1, device=weight.device, dtype=torch.long)
        self.indices = torch.arange(arena.capacity, device=weight.device)
        self.cache = SlotCache(arena, self.positions)
        self.graph = None

    def compute(self):
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
                 rms_norm=None, pack_qkv=False, codec_left_context=72, codec_batch_size=4):
        self.model = CustomVoiceModel(model_path, device, rms_norm=rms_norm, pack_qkv=pack_qkv)
        self.device = device
        self.speakers = self.model.speakers
        self.languages = self.model.languages
        self.talker = self.model.wrapper.model.talker
        self.max_slots = max_slots
        if max_slots < 1 or max_slots & (max_slots - 1):
            raise ValueError('Continuous slot count must be a positive power of two')
        self.arena = SlotArena(self.talker.model, max_slots, initial_capacity)
        self.graphs = {}
        self.predictor = UniformPredictor(self.talker.code_predictor)
        self.codec = CodecGraph(self.model.wrapper.model.speech_tokenizer.model.decoder)
        self.context = codec_left_context
        self.codec_batch_size = codec_batch_size
        self.states = []
        self.eos = self.talker.config.codec_eos_token_id
        self.allowed = torch.ones(self.talker.config.vocab_size, device=device, dtype=torch.bool)
        self.allowed[-1024:] = False
        self.allowed[self.eos] = True

    def ensure_capacity(self, required):
        if required < self.arena.capacity:
            return
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        old = self.arena
        self.arena = SlotArena(self.talker.model, self.max_slots, max(old.capacity * 2, 1 << required.bit_length()))
        for target, source in zip(self.arena.keys + self.arena.values, old.keys + old.values):
            target[:, :, :old.capacity].copy_(source)

    def semantic(self, logits, uniforms, seen, steps):
        logits = logits.float()
        logits = torch.where(seen, torch.where(logits < 0, logits * 1.05, logits / 1.05), logits)
        logits = logits.masked_fill(~self.allowed, -float('inf'))
        logits[:, self.eos] = torch.where(steps < 2, -float('inf'), logits[:, self.eos])
        return categorical(logits, uniforms)

    @torch.inference_mode()
    def admit(self, identifier, payload):
        if len(self.states) >= self.max_slots:
            raise ValueError('No free continuous request slot')
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
        prepared = {key: prompt[key] for key in ('inputs_embeds', 'attention_mask', 'trailing_text_hidden', 'tts_pad_embed')}
        output = self.talker(**prepared, use_cache=True, output_hidden_states=True, return_dict=True)
        position = output.past_key_values.get_seq_length()
        self.ensure_capacity(position)
        self.arena.load(len(self.states), output.past_key_values)
        uniforms = random_table(payload['seed'], payload['max_new_tokens']).to(self.device)
        seen = torch.zeros(1, self.talker.config.vocab_size, device=self.device, dtype=torch.bool)
        token = self.semantic(output.logits[:, -1], uniforms[0, 0:1], seen,
                              torch.zeros(1, device=self.device, dtype=torch.long))
        seen.scatter_(1, token, True)
        self.states.append(RequestState(identifier, payload, output.past_hidden, token,
                                        prompt['trailing_text_hidden'], prompt['tts_pad_embed'], uniforms, seen,
                                        position, int(self.talker.rope_deltas[0, 0].item())))

    def emit_audio(self, selected, emit):
        groups = {}
        for state in selected:
            if state.frames == state.emitted:
                continue
            start = max(0, state.emitted - self.context)
            codes = torch.stack(state.codes[start:]).transpose(0, 1)
            groups.setdefault(codes.shape[-1], []).append((state, codes, (state.emitted - start) * 1920))
        for entries in groups.values():
            for offset in range(0, len(entries), self.codec_batch_size):
                group = entries[offset:offset + self.codec_batch_size]
                batch = 1 << (len(group) - 1).bit_length()
                codes = [entry[1] for entry in group]
                codes += [codes[-1]] * (batch - len(codes))
                waves = self.codec.decode(torch.stack(codes))[:, 0].float().cpu().numpy()
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
                if index != last:
                    self.states[index] = self.states[last]
                self.states.pop()
        for identifier, payload in submissions:
            self.admit(identifier, payload)
        if not self.states:
            return []
        count = len(self.states)
        batch = 1 << (count - 1).bit_length()
        self.ensure_capacity(max(state.position for state in self.states))
        tokens = torch.cat([state.token for state in self.states])
        hidden = torch.cat([state.hidden for state in self.states])
        draws = torch.stack([state.uniforms[state.frames] for state in self.states])
        if batch > count:
            tokens = torch.cat((tokens, torch.zeros(batch - count, 1, device=self.device, dtype=tokens.dtype)))
            hidden = torch.cat((hidden, torch.zeros(batch - count, 1, hidden.shape[-1], device=self.device, dtype=hidden.dtype)))
            draws = torch.cat((draws, torch.full((batch - count, 16), 0.5, device=self.device)))
        first = self.talker.get_input_embeddings()(tokens)
        subcodes = self.predictor.run(torch.cat((hidden, first), dim=1), draws)
        codes = torch.cat((tokens, subcodes), dim=-1)
        embeds = torch.cat([first] + [self.talker.code_predictor.get_input_embeddings()[i](subcodes[:, i:i+1])
                                     for i in range(15)], dim=1).sum(1, keepdim=True)
        text = [state.trailing[:, state.frames:state.frames + 1] if state.frames < state.trailing.shape[1]
                else state.pad for state in self.states]
        if batch > count:
            text += [torch.zeros_like(text[0])] * (batch - count)
        embeds = embeds + torch.cat(text)
        positions = torch.tensor([s.position for s in self.states] + [0] * (batch - count), device=self.device)
        rotary = torch.tensor([s.position + s.rope_delta for s in self.states] + [0] * (batch - count),
                              device=self.device).view(1, batch, 1).expand(3, -1, -1)
        if batch not in self.graphs:
            graph = SlotGraph(self.arena, batch)
            graph.capture()
            self.graphs[batch] = graph
        hidden = self.graphs[batch].run(embeds, positions, rotary)[:count]
        logits = self.talker.codec_head(hidden)[:, -1]
        steps = torch.tensor([s.frames + 1 for s in self.states], device=self.device)
        uniforms = torch.stack([s.uniforms[s.frames + 1, 0] for s in self.states])
        seen = torch.cat([s.seen for s in self.states])
        next_tokens = self.semantic(logits, uniforms, seen, steps)
        token_ids = next_tokens[:, 0].tolist()
        due, finished = [], []
        for index, state in enumerate(self.states):
            state.codes.append(codes[index])
            state.frames += 1
            state.position += 1
            state.hidden = hidden[index:index + 1]
            state.token = next_tokens[index:index + 1]
            state.seen.scatter_(1, state.token, True)
            target = 2 if state.emitted == 0 else state.emitted * 2 if state.emitted < 8 else state.emitted + 8
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
            if index != last:
                self.states[index] = self.states[last]
            self.states.pop()
        return results

    def close(self):
        for graph in self.graphs.values():
            graph.close()
        self.predictor.close()
        self.codec.close()
        self.model.close()
