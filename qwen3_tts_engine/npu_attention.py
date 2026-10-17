"""Direct Ascend GQA attention without materializing repeated K/V heads.

Mask semantics follow torch SDPA at the boundary and Ascend FIA internally.
The training fusion operator failed graph capture on the qualified stack;
the inference operator is separately checked by tools/probe_gqa_operator.py.
"""
import torch
import torch_npu


def attention(module, query, key, value, attention_mask, **kwargs):
    if module.training or kwargs.get('dropout', 0.0) != 0.0:
        raise ValueError('Native GQA is inference-only')
    if module.sliding_window is not None:
        raise ValueError('Native GQA sliding-window masks are not implemented')
    mask = attention_mask
    if mask is None and query.shape[2] > 1 and getattr(module, 'is_causal', True):
        mask = torch.ones(query.shape[2], key.shape[2], device=query.device, dtype=torch.bool).triu(1)
    elif mask is not None:
        mask = ~mask if mask.dtype == torch.bool else mask.bool()
    output = torch_npu.npu_fused_infer_attention_score(
        query.contiguous(), key.contiguous(), value.contiguous(),
        num_heads=query.shape[1], num_key_value_heads=key.shape[1], input_layout='BNSD',
        actual_seq_lengths_kv=kwargs.get("actual_seq_lengths_kv"),
        atten_mask=mask, scale=kwargs.get('scaling', query.shape[-1] ** -0.5), sparse_mode=0)[0]
    return output.transpose(1, 2).contiguous(), None
