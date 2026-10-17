"""Direct Ascend GQA attention without materializing repeated K/V heads.

Mask semantics follow torch SDPA at the boundary and Ascend FAS internally.
API reference: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC1alpha003/apiref/fmkadptapi/ptaoplist_000442.html
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
    output = torch_npu.npu_fusion_attention(
        query.contiguous(), key.contiguous(), value.contiguous(), query.shape[1], 'BNSD',
        atten_mask=mask, scale=kwargs.get('scaling', query.shape[-1] ** -0.5),
        keep_prob=1.0, sparse_mode=1 if mask is not None else 0)[0]
    return output.transpose(1, 2).contiguous(), None
