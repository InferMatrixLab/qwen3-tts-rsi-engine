"""Compare complete reference/graph PCM, EOS and frame counts on fixed prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import wave

from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.predictor import PredictorGraph

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 64, 65])
parser.add_argument('--whole', action='store_true')
parser.add_argument('--talker-graph', action='store_true')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--rms-norm', choices=['native', 'reference-rounding'])
parser.add_argument('--pack-qkv', action='store_true')
parser.add_argument('--native-gqa', action='store_true')
args = parser.parse_args()
if args.batch_size <= 0:
    parser.error('Batch size must be positive')
args.output.mkdir(parents=True, exist_ok=False)
data = args.workload.read_bytes()
payloads = [json.loads(line) for line in data.splitlines() if line.strip()]
model = CustomVoiceModel(args.model, 'npu:0')
reference = model.wrapper.model.talker.code_predictor
graph = PredictorGraph(reference)
if args.whole:
    from qwen3_tts_engine.whole_predictor import WholePredictorGraph
    graph = WholePredictorGraph(reference)
result = {'status': 'INCONCLUSIVE', 'scope': 'complete waveform parity; diagnostic timings only',
          'workload_sha256': hashlib.sha256(data).hexdigest(), 'arms': {}}
result['implementation'] = type(graph).__name__
talker_reference = model.wrapper.model.talker.model
talker_graph = None
if args.talker_graph:
    from qwen3_tts_engine.talker_graph import TalkerGraph
    talker_graph = TalkerGraph(talker_reference)
result['talker_graph'] = args.talker_graph
result['batch_size'] = args.batch_size
result['rms_norm'] = args.rms_norm
result['pack_qkv'] = args.pack_qkv
result['native_gqa'] = args.native_gqa
replacements = []
try:
    for name, predictor in [('reference', reference), ('graph', graph)]:
        model.wrapper.model.talker.code_predictor = predictor
        if name == 'graph' and args.rms_norm:
            from qwen3_tts_engine.operators import install_rms_norms
            replacements = install_rms_norms(model.wrapper.model.talker, args.rms_norm)
        if name == 'graph' and (args.pack_qkv or args.native_gqa):
            from qwen3_tts_engine.packed_attention import install_packed_attention
            replacements.extend(install_packed_attention(model.wrapper.model.talker, native_gqa=args.native_gqa))
        if name == 'graph' and talker_graph is not None:
            model.wrapper.model.talker.model = talker_graph
        rows = []
        for start in range(0, len(args.indices), args.batch_size):
            indices = args.indices[start:start + args.batch_size]
            batch = [dict(payloads[index], instructions='') for index in indices]
            started = time.perf_counter()
            outputs = model.generate(batch)
            elapsed = time.perf_counter() - started
            for index, output in zip(indices, outputs):
                pcm = output.pop('pcm')
                with wave.open(str(args.output / f'{name}-{index:05d}.wav'), 'wb') as audio:
                    audio.setparams((1, 2, output['sample_rate'], 0, 'NONE', 'not compressed'))
                    audio.writeframes(pcm)
                rows.append(dict(output, index=index, pcm_sha256=hashlib.sha256(pcm).hexdigest(), elapsed_s=elapsed))
                result['arms'][name] = rows
                (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                print(name, index, output, elapsed, flush=True)
                if output['finish_reason'] != 'stop':
                    raise ValueError('Generation hit length limit')
    comparable = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_s'} for row in rows]
    result['exact'] = comparable(result['arms']['reference']) == comparable(result['arms']['graph'])
    if not result['exact']:
        raise ValueError('Complete waveform parity failed')
    result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    graph.close()
    if replacements:
        from qwen3_tts_engine.operators import restore_modules
        restore_modules(replacements)
    if talker_graph is not None:
        talker_graph.close()
        model.wrapper.model.talker.model = talker_reference
    model.wrapper.model.talker.code_predictor = reference
    model.close()
