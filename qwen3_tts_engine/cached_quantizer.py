"""Frozen codec codebook normalization, preserving reference accumulation order."""
import torch


class CachedQuantizer:
    def __init__(self, reference):
        if reference.training:
            raise ValueError('Codebook caching requires fixed evaluation weights')
        self.reference = reference
        self.embeddings = {}
        with torch.inference_mode():
            for group in (reference.rvq_first, reference.rvq_rest):
                for layer in group.vq.layers:
                    book = layer._codebook
                    self.embeddings[book] = book.embedding_sum / book.cluster_usage.clamp(min=book.epsilon)[:, None]

    def decode_group(self, group, codes):
        quantized = torch.zeros([1], device=codes.device)[0]
        for index, layer_codes in enumerate(codes.transpose(0, 1)):
            layer = group.vq.layers[index]
            values = torch.nn.functional.embedding(layer_codes, self.embeddings[layer._codebook])
            quantized = quantized + layer.project_out(values).transpose(1, 2)
        return group.output_proj(quantized)

    def decode(self, codes):
        reference = self.reference
        first = self.decode_group(reference.rvq_first, codes[:, :reference.n_q_semantic])
        if codes.shape[1] > reference.n_q_semantic:
            first += self.decode_group(reference.rvq_rest, codes[:, reference.n_q_semantic:])
        return first
