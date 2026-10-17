"""Experimental causal codec with per-stream convolution and attention state.

This eager correctness implementation is not enabled by the serving path.
It preserves the pretrained modules and never mutates their forward methods.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class CodecState:
    history: dict = field(default_factory=dict)
    cache: object = None
    frames: int = 0


class IncrementalCodec:
    def __init__(self, decoder):
        self.decoder = decoder

    def run_module(self, module, hidden, state):
        kind = type(module).__name__
        if kind == 'Qwen3TTSTokenizerV2CausalConvNet':
            if module.stride != 1:
                raise ValueError('Incremental codec requires stride-one causal convolutions')
            width = module.padding
            if width:
                previous = state.history.get(module)
                if previous is None:
                    previous = hidden.new_zeros(hidden.shape[0], hidden.shape[1], width)
                joined = torch.cat((previous, hidden), dim=-1)
                state.history[module] = joined[..., -width:].clone()
            else:
                joined = hidden
            return module.conv(joined).contiguous()
        if kind == 'Qwen3TTSTokenizerV2CausalTransConvNet':
            output = module.conv(hidden)
            overlap = module.right_pad
            if overlap:
                previous = state.history.get(module)
                if previous is not None:
                    output[..., :overlap] += previous
                tail = output[..., -overlap:]
                # Bias is already present in the next chunk's output.
                if module.conv.bias is not None:
                    tail = tail - module.conv.bias.view(1, -1, 1)
                state.history[module] = tail.clone()
                output = output[..., :-overlap]
            return output.contiguous()
        if kind == 'Qwen3TTSTokenizerV2ConvNeXtBlock':
            residual = hidden
            hidden = self.run_module(module.dwconv, hidden, state).permute(0, 2, 1)
            hidden = module.pwconv2(module.act(module.pwconv1(module.norm(hidden))))
            return residual + (module.gamma * hidden).permute(0, 2, 1)
        if kind == 'Qwen3TTSTokenizerV2DecoderDecoderResidualUnit':
            residual = hidden
            hidden = self.run_module(module.conv1, module.act1(hidden), state)
            hidden = self.run_module(module.conv2, module.act2(hidden), state)
            return residual + hidden
        if kind == 'Qwen3TTSTokenizerV2DecoderDecoderBlock':
            for child in module.block:
                hidden = self.run_module(child, hidden, state)
            return hidden
        if kind == 'SnakeBeta':
            return module(hidden)
        raise TypeError(f'Unsupported incremental codec module: {kind}')

    @torch.inference_mode()
    def decode(self, codes, state):
        if codes.ndim != 3 or codes.shape[1] != self.decoder.config.num_quantizers or codes.shape[-1] < 1:
            raise ValueError('Expected nonempty [batch, codebooks, frames] input')
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.run_module(self.decoder.pre_conv, hidden, state).transpose(1, 2)
        positions = torch.arange(state.frames, state.frames + codes.shape[-1], device=codes.device)
        result = self.decoder.pre_transformer(
            inputs_embeds=hidden, past_key_values=state.cache, use_cache=True,
            cache_position=positions, position_ids=positions.unsqueeze(0))
        state.cache = result.past_key_values
        state.frames += codes.shape[-1]
        hidden = result.last_hidden_state.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = self.run_module(block, hidden, state)
        for block in self.decoder.decoder:
            hidden = self.run_module(block, hidden, state)
        return hidden.clamp(min=-1, max=1)
