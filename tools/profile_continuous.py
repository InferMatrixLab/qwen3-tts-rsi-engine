"""Collect continuous-engine NPU traces; never use profiled timing as a benchmark."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import time
from contextlib import nullcontext

import torch
import torch_npu

from qwen3_tts_engine.continuous import ContinuousModel, SlotGraph


timings = []
measuring = False
current_step = -1
synchronized = False


def annotate(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if synchronized and measuring:
            torch.npu.synchronize()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            torch.npu.synchronize()
            timings.append({"label": label, "step": current_step,
                            "seconds": time.perf_counter() - start})
            return result
        with nullcontext() if synchronized else torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(owner, name, wrapped)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--slots', type=int, default=32)
parser.add_argument('--warmup', type=int, default=32)
parser.add_argument('--steps', type=int, default=16)
parser.add_argument('--incremental-codec', action='store_true')
parser.add_argument('--batch-prefill', action='store_true')
parser.add_argument('--native-kv-update', action='store_true')
parser.add_argument('--synchronized-timings', action='store_true')
parser.add_argument('--rms-norm', choices=['native', 'reference-rounding', 'reference'], default='native')
parser.add_argument('--pack-mlp', action='store_true')
parser.add_argument('--text-only-rope', action='store_true')
parser.add_argument('--prewarm-codec', action='store_true')
parser.add_argument('--bucket-kv', action='store_true')
parser.add_argument('--graph-embeddings', action='store_true')
parser.add_argument('--bucket-prefill', action='store_true')
parser.add_argument('--decode-bucket-step', type=int, default=0)
parser.add_argument('--batch-audio-transfer', action='store_true')
parser.add_argument('--native-gqa', action='store_true')
parser.add_argument('--codec-batch-size', type=int, default=4)
parser.add_argument('--graph-prefill', action='store_true')
parser.add_argument('--prefill-per-tick', type=int, default=4)
args = parser.parse_args()
synchronized = args.synchronized_timings
if args.prefill_per_tick < 1 or args.codec_batch_size < 1:
    parser.error('prefill-per-tick and codec-batch-size must be positive')
if args.slots < 1 or args.slots & (args.slots - 1) or args.steps < 1 or args.warmup < 1:
    parser.error('slots must be a positive power of two; warmup and steps must be positive')
args.output.mkdir(parents=True, exist_ok=False)
repository = Path(__file__).absolute().parent.parent
revision = subprocess.check_output(
    ['git', '-c', f'safe.directory={repository}', '-C', str(repository), 'rev-parse', 'HEAD'],
    text=True).strip()
data = args.workload.read_bytes()
payloads = [dict(json.loads(line), instructions='') for line in data.splitlines() if line.strip()]
if len(payloads) < args.slots + args.warmup + args.steps:
    parser.error('Use a corpus large enough to avoid recycling requests within the trace')
model = ContinuousModel(args.model, 'npu:0', max_slots=args.slots,
                        rms_norm=None if args.rms_norm == 'reference' else args.rms_norm, pack_qkv=True, codec_left_context=72, codec_batch_size=args.codec_batch_size,
                        incremental_codec=args.incremental_codec, batch_prefill=args.batch_prefill,
                        native_kv_update=args.native_kv_update, pack_mlp=args.pack_mlp,
                        graph_prefill=args.graph_prefill, text_only_rope=args.text_only_rope, prewarm_codec=args.prewarm_codec, bucket_kv=args.bucket_kv, graph_embeddings=args.graph_embeddings, native_gqa=args.native_gqa, bucket_prefill=args.bucket_prefill, decode_bucket_step=args.decode_bucket_step, batch_audio_transfer=args.batch_audio_transfer)
for owner, name, label in [(model, 'admit', 'continuous_prefill'),
                            (model, 'admit_batch', 'continuous_batch_prefill'),
                            (model.predictor, 'run', 'continuous_predictor'),
                            (model, 'semantic', 'continuous_sampling'),
                            (model, 'emit_audio', 'continuous_audio'),
                            (model.codec, 'decode', 'continuous_codec'),
                            (SlotGraph, 'run', 'continuous_talker')]:
    annotate(owner, name, label)
next_index = 0
events = []


def tick(phase, step):
    global next_index, current_step
    current_step = step
    count = min(args.prefill_per_tick, args.slots - len(model.states), len(payloads) - next_index)
    submissions = [(i, payloads[i]) for i in range(next_index, next_index + count)]
    next_index += count
    before = len(model.states)
    if synchronized:
        torch.npu.synchronize()
        start = time.perf_counter()
        completed = model.tick(submissions)
        torch.npu.synchronize()
        if measuring:
            timings.append({'label': 'continuous_tick', 'step': step,
                            'seconds': time.perf_counter() - start})
    else:
        with torch.profiler.record_function('continuous_tick'):
            completed = model.tick(submissions)
    events.append({'phase': phase, 'step': step, 'active_before': before,
                   'admitted': count, 'active_after': len(model.states),
                   'completed': [i for i, _ in completed], 'capacity': model.arena.capacity})
    if not model.states:
        raise RuntimeError('Corpus exhausted during profiling')


summary = {'status': 'INCONCLUSIVE', 'scope': 'diagnostic trace, not throughput or quality evidence',
           'revision': revision,
           'workload_sha256': hashlib.sha256(data).hexdigest(), 'slots': args.slots,
           'warmup': args.warmup, 'steps': args.steps, 'events': events,
           'incremental_codec': args.incremental_codec, 'batch_prefill': args.batch_prefill,
           'native_kv_update': args.native_kv_update,
           'configuration': {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
           'source_sha256': {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted((repository / 'qwen3_tts_engine').glob('*.py'))},
           'timings': timings,
           'timing_caveat': 'Synchronized inclusive ranges overlap and perturb execution; do not add nested ranges or interpret as service throughput.'}
try:
    for step in range(args.warmup):
        tick('warmup', step)
    torch.npu.synchronize()
    measuring = True
    with nullcontext() if synchronized else torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
    ):
        for step in range(args.steps):
            tick('profile', step)
        torch.npu.synchronize()
    summary['status'] = 'DIAGNOSTIC_ONLY'
finally:
    summary['captured_shapes'] = {'prefill': sorted(model.prefills), 'talker': sorted(model.graphs),
                                  'codec': sorted(model.codec.graphs) if model.incremental_codec else [str(key) for key in model.codec.captures]}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    model.close()
