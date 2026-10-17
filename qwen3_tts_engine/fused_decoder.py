"""Opt-in native residual-add/post-attention RMSNorm fusion.

The fused normalization accumulates the unrounded sum, so BF16 outputs may
change. This requires independent speech-quality validation.
"""
import torch
import torch_npu


class FusedResidualDecoder(torch.nn.Module):
    def __init__(self, reference, reference_single=False):
        super().__init__()
        self.reference = reference
        self.reference_single = reference_single
        if hasattr(reference, 'attention_type'):
            self.attention_type = reference.attention_type
        norm = reference.post_attention_layernorm
        if getattr(norm, 'mode', None) != 'native':
            raise ValueError('Residual fusion requires native RMSNorm')

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        layer = self.reference
        arguments = dict(attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, output_attentions=output_attentions,
                         use_cache=use_cache, cache_position=cache_position,
                         position_embeddings=position_embeddings, **kwargs)
        if self.reference_single and hidden_states.shape[0] == 1:
            return layer(hidden_states, **arguments)
        attended, weights = layer.self_attn(
            hidden_states=layer.input_layernorm(hidden_states), **arguments)
        norm = layer.post_attention_layernorm
        normalized, _, residual = torch_npu.npu_add_rms_norm(
            attended, hidden_states, norm.weight, epsilon=norm.variance_epsilon)
        output = residual + layer.mlp(normalized)
        return (output, weights) if output_attentions else (output,)


def install_fused_decoders(model, reference_single=False):
    replacements = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ in ('Qwen3TTSTalkerDecoderLayer', 'Qwen3TTSDecoderLayer'):
                replacements.append((parent, name, child))
                setattr(parent, name, FusedResidualDecoder(child, reference_single))
    return replacements
