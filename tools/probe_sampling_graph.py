"""Check exact semantic sampling, EOS masking and independent graph output ownership."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_npu
from qwen3_tts_engine.continuous import ContinuousModel
from qwen3_tts_engine.sampling_graph import SamplingGraph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
state = SimpleNamespace(eos=2150, allowed=torch.ones(3072, device='npu:0', dtype=torch.bool))
state.allowed[-1024:] = False
state.allowed[state.eos] = True
sample = lambda *values: ContinuousModel.semantic(state, *values)
result = {'status': 'INCONCLUSIVE', 'scope': 'exact synthetic sampling semantics, not quality or service throughput', 'cases': []}
graph = None
try:
    with torch.inference_mode():
        for batch in [1, 17, 64]:
            graph = SamplingGraph(sample, batch, 3072, 'npu:0')
            graph.capture()
            retained = []
            for turn in range(4):
                generator = torch.Generator().manual_seed(42 + turn)
                logits = torch.randn(batch, 3072, generator=generator, dtype=torch.bfloat16).to('npu:0')
                logits[:, state.eos] = 30
                uniforms = torch.rand(batch, generator=generator).to('npu:0')
                seen = (torch.rand(batch, 3072, generator=generator) > 0.9).to('npu:0')
                steps = torch.full((batch,), turn, device='npu:0', dtype=torch.long)
                expected = sample(logits, uniforms, seen, steps)
                tokens, updated = graph.run(logits, uniforms, seen, steps)
                torch.testing.assert_close(tokens, expected, atol=0, rtol=0)
                torch.testing.assert_close(updated, seen.scatter(1, expected, True), atol=0, rtol=0)
                assert bool((tokens != state.eos).all()) if turn < 2 else bool((tokens == state.eos).all())
                for value, snapshot in retained:
                    torch.testing.assert_close(value, snapshot, atol=0, rtol=0)
                retained += [(tokens, tokens.clone()), (updated, updated.clone())]
            result['cases'].append({'batch': batch, 'replays': 4, 'exact': True})
            graph.close()
            graph = None
        result['status'] = 'PASS'
finally:
    if graph is not None:
        graph.close()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
