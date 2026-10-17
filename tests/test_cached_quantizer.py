"""Normalization hoisting must preserve clamp semantics and codebook sum order."""
import pytest
import torch
pytest.importorskip('qwen_tts')
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import SplitResidualVectorQuantizer
from qwen3_tts_engine.cached_quantizer import CachedQuantizer


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_normalized_codebooks_match_reference(dtype):
    torch.manual_seed(42)
    reference = SplitResidualVectorQuantizer(n_q=4, n_q_semantic=1, dimension=8,
                                             input_dimension=16, output_dimension=16, bins=17).to(dtype).eval()
    with torch.inference_mode():
        for group in (reference.rvq_first, reference.rvq_rest):
            for layer in group.vq.layers:
                book = layer._codebook
                book.embedding_sum.normal_(0, 1e-4)
                book.cluster_usage.copy_(torch.linspace(-.1, 2, 17, dtype=dtype))
                book.cluster_usage[1] = 0
                book.cluster_usage[2] = 1e-7
        candidate = CachedQuantizer(reference)
        for batch, length in [(1, 1), (3, 8), (2, 17)]:
            codes = torch.randint(0, 17, (batch, 4, length))
            for count in [1, 2, 4]:
                torch.testing.assert_close(candidate.decode(codes[:, :count]), reference.decode(codes[:, :count]), atol=0, rtol=0)


def test_training_codebook_cache_rejected():
    with pytest.raises(ValueError, match='evaluation'):
        CachedQuantizer(torch.nn.Identity())
