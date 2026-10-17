"""Measure real codec 1x1 convolution weights with Conv1d versus transposed linear."""
import argparse
import json
from pathlib import Path
import time

import torch
from qwen3_tts_engine.model import CustomVoiceModel


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
result = {'status': 'INCONCLUSIVE', 'scope': 'Real weights, synthetic activations; numerical and operator timing diagnostic only', 'cases': []}
model = CustomVoiceModel(args.model, 'npu:0')
graphs = {}
try:
    decoder = model.wrapper.model.speech_tokenizer.model.decoder
    unique = {}
    for name, module in decoder.named_modules():
        if (isinstance(module, torch.nn.Conv1d) and module.kernel_size == (1,)
                and module.stride == (1,) and module.padding == (0,) and module.groups == 1):
            unique.setdefault((module.in_channels, module.out_channels), (name, module))
    shapes = sorted(unique, key=lambda s: s[0] * s[1])
    if not shapes:
        raise ValueError('No supported codec pointwise convolutions')
    selected = sorted({0, len(shapes) // 2, len(shapes) - 1})
    torch.manual_seed(42)
    with torch.inference_mode():
        for index in selected:
            name, conv = unique[shapes[index]]
            for length in (8, 128, 1024):
                hidden = torch.randn(4, conv.in_channels, length, device='npu:0', dtype=torch.bfloat16) * .02
                def operation(linear):
                    if linear:
                        return torch.nn.functional.linear(hidden.transpose(1, 2), conv.weight[..., 0], conv.bias).transpose(1, 2).contiguous()
                    return conv(hidden).contiguous()
                for linear in (False, True):
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        for _ in range(3): operation(linear)
                    torch.npu.current_stream().wait_stream(stream)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream): output = operation(linear)
                    graphs[linear] = (graph, output)
                for graph, _ in graphs.values(): graph.replay()
                reference, candidate = graphs[False][1], graphs[True][1]
                assert bool(torch.isfinite(candidate).all())
                error = (reference.float() - candidate.float()).abs()
                row = {'module': name, 'input_channels': conv.in_channels, 'output_channels': conv.out_channels,
                       'batch': 4, 'length': length, 'exact': torch.equal(reference, candidate),
                       'max_abs_error': error.max().item(), 'rms_error': error.square().mean().sqrt().item(), 'measurements': []}
                for linear in (False, True, True, False):
                    graph = graphs[linear][0]
                    for _ in range(10): graph.replay()
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(100): graph.replay()
                    torch.npu.synchronize()
                    row['measurements'].append({'arm': 'linear' if linear else 'conv1d', 'mean_ms': (time.perf_counter() - start) * 10})
                result['cases'].append(row)
                print(row, flush=True)
                for graph, _ in graphs.values(): graph.reset()
                graphs.clear()
    result['status'] = 'DIAGNOSTIC_ONLY'
except Exception as error:
    result.update(status='FAIL', error=repr(error))
    raise
finally:
    for graph, _ in graphs.values(): graph.reset()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    model.close()
