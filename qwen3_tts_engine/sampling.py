"""Seeded inverse-CDF sampling without request-slot-dependent RNG state."""
import torch


def categorical(scores, uniform, *, temperature=0.9, top_k=50):
    scores = scores.float() / temperature
    threshold = torch.topk(scores, top_k, dim=-1).values[..., -1:]
    probabilities = torch.softmax(scores.masked_fill(scores < threshold, -float('inf')), dim=-1)
    cdf = probabilities.cumsum(-1)
    cdf = cdf / cdf[..., -1:]
    return (cdf <= uniform.unsqueeze(-1)).sum(-1, keepdim=True)


def random_table(seed, steps):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.rand(steps + 1, 16, generator=generator)
