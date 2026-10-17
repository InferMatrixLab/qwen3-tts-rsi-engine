"""Experimental padded prefill graph with explicit masks and retained KV outputs."""
import torch
from transformers.cache_utils import DynamicCache


def pad_prefill_batch(embeddings, mask, batch):
    """Append independent dummy rows; their outputs are never admitted as requests."""
    extra = batch - embeddings.shape[0]
    if extra < 0 or mask.shape != embeddings.shape[:2]:
        raise ValueError('Prefill batch or attention-mask shape is invalid')
    if not extra:
        return embeddings, mask
    embeddings = torch.cat((embeddings, embeddings.new_zeros(extra, *embeddings.shape[1:])))
    mask = torch.cat((mask, mask.new_ones(extra, mask.shape[1])))
    return embeddings, mask


class PrefillGraph:
    def __init__(self, talker, batch, length):
        self.talker = talker
        weight = next(talker.parameters())
        self.inputs = weight.new_zeros(batch, length, talker.config.hidden_size)
        self.mask = torch.ones(batch, length, device=weight.device, dtype=torch.long)
        self.rows = torch.arange(batch, device=weight.device)
        self.positions = torch.arange(length, device=weight.device)
        self.causal = self.positions[None, :] <= self.positions[:, None]
        self.graph = None

    def compute(self):
        model = self.talker.model
        hidden = self.inputs
        ids = self.mask.float().cumsum(-1) - 1
        ids = ids.masked_fill(self.mask == 0, 1).unsqueeze(0).expand(3, -1, -1)
        allowed = self.mask[:, None, :].bool() & self.causal[None]
        attention = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)[:, None]
        rotary = model.rotary_emb(hidden, ids)
        cache = DynamicCache()
        for layer in model.layers:
            hidden = layer(hidden, attention_mask=attention, position_ids=ids[0], past_key_values=cache,
                           output_attentions=False, use_cache=True, cache_position=self.positions,
                           position_embeddings=rotary)[0]
        hidden = model.norm(hidden)
        hidden = hidden[self.rows, self.mask.sum(-1) - 1].unsqueeze(1)
        return hidden, self.talker.codec_head(hidden)[:, -1], cache

    @torch.inference_mode()
    def capture(self):
        stream = torch.npu.Stream(device=self.inputs.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                self.compute()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.hidden, self.logits, self.cache = self.compute()

    @torch.inference_mode()
    def run(self, embeddings, mask):
        self.inputs.copy_(embeddings)
        self.mask.copy_(mask)
        self.graph.replay()
        # Request state survives future replays; KV is copied to its arena by
        # the owner thread before another prefill can overwrite this cache.
        return self.hidden.clone(), self.logits.clone(), self.cache

    def close(self):
        torch.npu.synchronize()
        if self.graph is not None:
            self.graph.reset()
