"""Compare TorchAir whole-talker decode with the existing graph on identical KV state."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
import torch_npu
import torchair
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.continuous import SlotArena, SlotGraph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--capacity', type=int, default=512)
args = parser.parse_args()
if not 1 <= args.batch <= 64 or args.capacity < args.batch + 65:
    parser.error('Expected batch 1..64 and capacity covering all probe positions')
args.output.mkdir(parents=True, exist_ok=False)
result = {'status': 'INCONCLUSIVE', 'scope': 'Synthetic hidden-state backend diagnostic; not speech quality or service performance',
          'batch': args.batch, 'capacity': args.capacity, 'samples': [],
          'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'torchair': torchair.__file__,
          'fullgraph': True, 'dynamic': False,
          'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((Path(__file__).absolute().parent.parent / 'qwen3_tts_engine').glob('*.py'))}}
model = None
oracle = None
try:
    model = CustomVoiceModel(args.model, 'npu:0', rms_norm='native', pack_qkv=True,
                             pack_mlp=True, fused_swiglu=True, native_gqa=True, native_rope=True)
    reference = model.wrapper.model.talker.model
    with torch.inference_mode():
        first = SlotArena(reference, args.batch, args.capacity, native_kv_update=True)
        second = SlotArena(reference, args.batch, args.capacity, native_kv_update=True)
        oracle, candidate = SlotGraph(first, args.batch), SlotGraph(second, args.batch)
        oracle.capture()
        torch.manual_seed(42)
        for left, right in zip(first.keys + first.values, second.keys + second.values):
            left.normal_(0, 0.02)
            right.copy_(left)
        inputs = torch.randn_like(oracle.inputs) * 0.02
        positions = torch.arange(args.batch, device='npu:0') + 64
        rotary = positions.view(1, args.batch, 1).expand(3, -1, -1)
        expected = oracle.run(inputs, positions, rotary)
        candidate.inputs.copy_(inputs)
        candidate.positions.copy_(positions)
        candidate.rotary.copy_(rotary)
        config = torchair.CompilerConfig()
        backend = torchair.get_npu_backend(compiler_config=config)
        compiled = torch.compile(candidate.compute, backend=backend, dynamic=False, fullgraph=True)
        torch.npu.synchronize()
        start = time.perf_counter()
        actual = compiled().clone()
        torch.npu.synchronize()
        result['compile_first_call_seconds'] = time.perf_counter() - start
        error = (actual.float() - expected.float()).abs()
        result['max_abs_error'] = error.max().item()
        result['rms_error'] = error.square().mean().sqrt().item()
        result['cosine'] = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()
        result['finite'] = bool(torch.isfinite(actual).all())
        if not result['finite']:
            raise ValueError('Compiled decoder returned nonfinite values')
        def run_compiled():
            candidate.inputs.copy_(inputs)
            candidate.positions.copy_(positions)
            candidate.rotary.copy_(rotary)
            return compiled().clone()
        arms = {'npugraph': lambda: oracle.run(inputs, positions, rotary), 'torchair': run_compiled}
        for arm in ['npugraph', 'torchair', 'torchair', 'npugraph']:
            for _ in range(10): arms[arm]()
            durations = []
            for _ in range(100):
                torch.npu.synchronize()
                start = time.perf_counter()
                arms[arm]()
                torch.npu.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            result['samples'].append({'arm': arm, 'durations_ms': durations})
            print(arm, sum(durations) / len(durations), flush=True)
        result['status'] = 'DIAGNOSTIC_ONLY'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = f'{type(error).__name__}: {error}'
    (args.output / 'compile.err.txt').write_text(traceback.format_exc())
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if oracle is not None: oracle.close()
    if model is not None: model.close()
