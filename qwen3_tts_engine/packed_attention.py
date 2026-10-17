# Adapted from Qwen3-TTS attention, Copyright 2026 The Qwen team,
# Alibaba Group and the HuggingFace Inc. team. Licensed under Apache-2.0.
"""Pack Q/K/V projections while retaining upstream attention and rotary math."""
import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    ALL_ATTENTION_FUNCTIONS, apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb, eager_attention_forward,
)


class PackedAttention(torch.nn.Module):
    def __init__(self, reference, native_gqa=False):
        super().__init__()
        self.reference = reference
        projections = [reference.q_proj, reference.k_proj, reference.v_proj]
        self.sizes = [projection.out_features for projection in projections]
        self.weight = torch.nn.Parameter(torch.cat([p.weight.detach() for p in projections], dim=0),
                                         requires_grad=False)
        self.bias = None
        if projections[0].bias is not None:
            self.bias = torch.nn.Parameter(torch.cat([p.bias.detach() for p in projections], dim=0),
                                           requires_grad=False)
        self.multimodal = reference.__class__.__name__ == 'Qwen3TTSTalkerAttention'
        self.native_gqa = native_gqa

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        reference = self.reference
        shape = hidden_states.shape[:-1]
        heads = (*shape, -1, reference.head_dim)
        query, key, value = torch.nn.functional.linear(hidden_states, self.weight, self.bias).split(self.sizes, dim=-1)
        query = reference.q_norm(query.reshape(heads)).transpose(1, 2)
        key = reference.k_norm(key.reshape(heads)).transpose(1, 2)
        value = value.reshape(heads).transpose(1, 2)
        cos, sin = position_embeddings
        if self.multimodal:
            query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin,
                                                       reference.rope_scaling['mrope_section'],
                                                       reference.rope_scaling['interleaved'])
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, reference.layer_idx,
                                               {'sin': sin, 'cos': cos, 'cache_position': cache_position})
        attention = eager_attention_forward if reference.config._attn_implementation == 'eager' else (
            ALL_ATTENTION_FUNCTIONS[reference.config._attn_implementation])
        if self.native_gqa:
            from qwen3_tts_engine.npu_attention import attention
        output, weights = attention(reference, query, key, value, attention_mask,
                                    dropout=0.0 if not reference.training else reference.attention_dropout,
                                    scaling=reference.scaling, sliding_window=reference.sliding_window, **kwargs)
        return reference.o_proj(output.reshape(*shape, -1).contiguous()), weights


def install_packed_attention(model, native_gqa=False):
    replacements = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ in ('Qwen3TTSTalkerAttention', 'Qwen3TTSAttention'):
                replacements.append((parent, name, child))
                setattr(parent, name, PackedAttention(child, native_gqa=native_gqa))
    return replacements
