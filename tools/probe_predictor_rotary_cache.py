"""Diagnostic: hoist fixed predictor rotary tensors out of graph replay."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.continuous import UniformPredictor


class CachedRotary(torch.nn.Module):
    """Probe-only cache; retained position storage prevents pointer reuse."""
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.entries = {}

    def forward(self, hidden, positions):
        key = (positions.data_ptr(), tuple(positions.shape), hidden.dtype, hidden.device)
        if key not in self.entries:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError('Rotary cache miss during graph capture')
            self.entries[key] = (positions, self.reference(hidden, positions))
        return self.entries[key][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'DIAGNOSTIC_ONLY', 'scope': 'Fixed predictor rotary hoisting; synthetic inputs, not speech quality', 'cohorts': []}
    model = CustomVoiceModel(args.model, 'npu:0', rms_norm='native', pack_qkv=True,
                             pack_mlp=True, fused_swiglu=True, native_gqa=True, native_rope=True)
    predictor = model.wrapper.model.talker.code_predictor
    rotary = predictor.model.rotary_emb
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
                    predictor.model.rotary_emb = CachedRotary(rotary)
                    actual = cached.run(inputs, draws)
                    predictor.model.rotary_emb = rotary
                    row = {'batch': batch, 'codes_equal': torch.equal(expected[0], actual[0]),
                           'embeddings_equal': torch.equal(expected[1], actual[1]), 'measurements': []}
                    if not row['codes_equal'] or not row['embeddings_equal']:
                        raise AssertionError('Fixed rotary changed predictor output')
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
                    predictor.model.rotary_emb = rotary
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
