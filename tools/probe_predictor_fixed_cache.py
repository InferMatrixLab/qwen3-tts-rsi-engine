"""Probe fixed-capacity predictor KV against the current DynamicCache graph."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.continuous import UniformPredictor
import qwen3_tts_engine.continuous as continuous
import qwen3_tts_engine.npu_attention as native_attention
from transformers.cache_utils import Cache


class FixedCache(Cache):
    def __init__(self, predictor, inputs):
        super().__init__(layers=[])
        c = predictor.config
        shape = (inputs.shape[0], c.num_key_value_heads, 16, c.head_dim)
        self.keys = [torch.zeros(shape, device=inputs.device, dtype=inputs.dtype)
                     for _ in range(c.num_hidden_layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.step = -1

    def reset(self):
        self.step = -1
        return self

    def get_seq_length(self, layer_idx=0):
        return 0 if self.step < 0 else self.step + 2

    def update(self, keys, values, layer_idx, cache_kwargs):
        if layer_idx == 0:
            self.step += 1
        start = 0 if self.step == 0 else self.step + 1
        end = self.step + 2
        self.keys[layer_idx][:, :, start:end].copy_(keys)
        self.values[layer_idx][:, :, start:end].copy_(values)
        return self.keys[layer_idx], self.values[layer_idx]


def padded_attention(cache):
    def attention(module, query, key, value, attention_mask, **kwargs):
        mask = attention_mask
        if mask is not None:
            mask = ~mask if mask.dtype == torch.bool else mask.bool()
            mask = torch.nn.functional.pad(mask, (0, 16 - mask.shape[-1]), value=False)
        out = torch_npu.npu_fused_infer_attention_score(
            query.contiguous(), key, value, atten_mask=mask,
            actual_seq_lengths_kv=[cache.step + 2] * query.shape[0],
            num_heads=query.shape[1], num_key_value_heads=key.shape[1],
            input_layout='BNSD', scale=kwargs.get('scaling', query.shape[-1] ** -.5),
            sparse_mode=0)[0]
        return out.transpose(1, 2).contiguous(), None
    return attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'DIAGNOSTIC_ONLY', 'scope': 'Real predictor weights with synthetic inputs; not speech quality or service throughput', 'cohorts': []}
    model = CustomVoiceModel(args.model, 'npu:0', rms_norm='native', pack_qkv=True,
                             pack_mlp=True, fused_swiglu=True, native_gqa=True, native_rope=True)
    predictor = model.wrapper.model.talker.code_predictor
    dynamic_factory = continuous.DynamicCache
    original_attention = native_attention.attention
    try:
        with torch.inference_mode():
            torch.manual_seed(42)
            for batch in (1, 8, 32, 64):
                inputs = torch.randn(batch, 2, model.wrapper.model.talker.config.hidden_size,
                                     device='npu:0', dtype=torch.bfloat16) * .02
                draws = torch.rand(batch, 16, device='npu:0', dtype=torch.float32)
                original = UniformPredictor(predictor, graph_embeddings=True)
                cached = UniformPredictor(predictor, graph_embeddings=True)
                try:
                    expected = original.run(inputs, draws)
                    cache = FixedCache(predictor, inputs)
                    continuous.DynamicCache = cache.reset
                    native_attention.attention = padded_attention(cache)
                    actual = cached.run(inputs, draws)
                    continuous.DynamicCache = dynamic_factory
                    native_attention.attention = original_attention
                    row = {'batch': batch, 'codes_equal': torch.equal(expected[0], actual[0]),
                           'embeddings_equal': torch.equal(expected[1], actual[1]), 'measurements': []}
                    if not row['codes_equal'] or not row['embeddings_equal']:
                        raise AssertionError('Fixed KV changed predictor output')
                    row['replays'] = []
                    for replay in range(3):
                        inputs.copy_(torch.randn_like(inputs) * .02)
                        draws.copy_(torch.rand_like(draws))
                        expected = original.run(inputs, draws)
                        actual = cached.run(inputs, draws)
                        equal = all(torch.equal(a, b) for a, b in zip(expected, actual))
                        row['replays'].append({'replay': replay, 'exact': equal})
                        if not equal:
                            raise AssertionError('Fixed KV replay changed predictor output')
                    for label, graph in [('original', original), ('cached', cached), ('cached', cached), ('original', original)]:
                        for _ in range(10): graph.run(inputs, draws)
                        durations = []
                        for _ in range(50):
                            torch.npu.synchronize()
                            start = time.perf_counter()
                            graph.run(inputs, draws)
                            torch.npu.synchronize()
                            durations.append((time.perf_counter() - start) * 1000)
                        row['measurements'].append({'arm': label, 'durations_ms': durations})
                    result['cohorts'].append(row)
                    print(batch, [(x['arm'], sum(x['durations_ms']) / len(x['durations_ms'])) for x in row['measurements']], flush=True)
                finally:
                    continuous.DynamicCache = dynamic_factory
                    native_attention.attention = original_attention
                    original.close()
                    cached.close()
    except Exception as error:
        result['status'] = 'FAIL'
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        model.close()


if __name__ == '__main__':
    main()
