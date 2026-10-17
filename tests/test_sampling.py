import torch
from qwen3_tts_engine.sampling import categorical, random_table


def test_categorical_skips_zero_mass_at_exact_cdf_boundaries():
    scores = torch.tensor([[-float('inf'), 0., -float('inf'), 0.]]).repeat(3, 1)
    draws = torch.tensor([0., 0.5, 1. - torch.finfo(torch.float32).eps])
    assert categorical(scores, draws, top_k=2).flatten().tolist() == [1, 3, 3]


def test_seeded_tables_are_request_local_and_sampling_is_permutation_equivariant():
    first = random_table(42, 1000)
    random_table(7, 19)
    assert torch.equal(first, random_table(42, 1000))
    draws = first.flatten()[:10000]
    scores = torch.tensor([[0., 1.0986122886681098]]).repeat(len(draws), 1)
    tokens = categorical(scores, draws, temperature=1., top_k=2)
    assert abs(tokens.float().mean().item() - 0.75) < 0.02
    order = torch.randperm(len(draws), generator=torch.Generator().manual_seed(123))
    assert torch.equal(categorical(scores[order], draws[order], temperature=1., top_k=2), tokens[order])
