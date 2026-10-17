"""Compare external versus captured semantic head at identical real batch sizes."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch_npu
from safetensors import safe_open
from qwen3_tts_engine.continuous import ContinuousModel
from qwen3_tts_engine.sampling_graph import SamplingGraph


class HeadGraph(SamplingGraph):
    def __init__(self, sample, batch, head):
        super().__init__(sample, batch, head.out_features, head.weight.device)
        self.head = head
        self.hidden = torch.zeros(batch, 1, head.in_features,
                                  device=head.weight.device, dtype=head.weight.dtype)

    def compute(self):
        logits = self.head(self.hidden)[:, -1].float()
        tokens = self.sample(logits, self.uniforms, self.seen, self.steps)
        return tokens, self.seen.scatter(1, tokens, True)

    def run(self, hidden, uniforms, seen, steps):
        self.hidden.copy_(hidden)
        self.uniforms.copy_(uniforms)
        self.seen.copy_(seen)
        self.steps.copy_(steps)
        self.graph.replay()
        return self.tokens.clone(), self.updated_seen.clone()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device('npu:0')
torch.manual_seed(42)
with safe_open(args.model / 'model.safetensors', framework='pt', device='cpu') as checkpoint:
    names = [name for name in checkpoint.keys() if name.endswith('talker.codec_head.weight')]
    if len(names) != 1:
        raise ValueError(f'Expected one talker output head, got {names}')
    weight = checkpoint.get_tensor(names[0]).to(device='npu:0', dtype=torch.bfloat16)
head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device='npu:0', dtype=weight.dtype)
head.weight = torch.nn.Parameter(weight, requires_grad=False)
state = SimpleNamespace(eos=2150, allowed=torch.ones(weight.shape[0], device='npu:0', dtype=torch.bool))
state.allowed[-1024:] = False
state.allowed[state.eos] = True
sample = lambda *values: ContinuousModel.semantic(state, *values)
result = {'status': 'INCONCLUSIVE', 'scope': 'Real output-head weights and synthetic hidden states; not service throughput or audio quality', 'cases': []}
graphs = []
try:
    with torch.inference_mode():
        for batch in [1, 17, 32, 64]:
            control = SamplingGraph(sample, batch, weight.shape[0], 'npu:0')
            candidate = HeadGraph(sample, batch, head)
            graphs = [control, candidate]
            for graph in graphs:
                graph.capture()
            for replay in range(4):
                hidden = torch.randn(batch, 1, weight.shape[1], device='npu:0', dtype=weight.dtype)
                uniforms = torch.rand(batch, device='npu:0')
                seen = torch.rand(batch, weight.shape[0], device='npu:0') > .9
                steps = torch.full((batch,), replay, device='npu:0', dtype=torch.long)
                expected = control.run(head(hidden)[:, -1], uniforms, seen, steps)
                actual = candidate.run(hidden, uniforms, seen, steps)
                for first, second in zip(expected, actual):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
            timings = []
            for name in ['external_head', 'captured_head', 'captured_head', 'external_head']:
                operation = (lambda: control.run(head(hidden)[:, -1], uniforms, seen, steps)) if name == 'external_head' else (lambda: candidate.run(hidden, uniforms, seen, steps))
                for _ in range(10):
                    operation()
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    operation()
                torch.npu.synchronize()
                timings.append({'arm': name, 'mean_ms': (time.perf_counter() - start) * 10})
            result['cases'].append({'batch': batch, 'exact_tokens_and_seen': True, 'replays': 4, 'measurements': timings})
            for graph in graphs:
                graph.close()
            graphs = []
        result['status'] = 'DIAGNOSTIC_ONLY'
finally:
    for graph in graphs:
        graph.close()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
