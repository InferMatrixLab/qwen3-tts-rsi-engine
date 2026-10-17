"""Collect continuous-engine NPU traces; never use profiled timing as a benchmark."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import torch_npu

from qwen3_tts_engine.continuous import ContinuousModel, SlotGraph


def annotate(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(owner, name, wrapped)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--slots', type=int, default=32)
parser.add_argument('--warmup', type=int, default=32)
parser.add_argument('--steps', type=int, default=16)
args = parser.parse_args()
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
                        rms_norm='native', pack_qkv=True, codec_left_context=72, codec_batch_size=4)
for owner, name, label in [(model, 'admit', 'continuous_prefill'),
                            (model.predictor, 'run', 'continuous_predictor'),
                            (model, 'semantic', 'continuous_sampling'),
                            (model, 'emit_audio', 'continuous_audio'),
                            (model.codec, 'decode', 'continuous_codec'),
                            (SlotGraph, 'run', 'continuous_talker')]:
    annotate(owner, name, label)
next_index = 0
events = []


def tick(phase, step):
    global next_index
    count = min(4, args.slots - len(model.states), len(payloads) - next_index)
    submissions = [(i, payloads[i]) for i in range(next_index, next_index + count)]
    next_index += count
    before = len(model.states)
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
           'warmup': args.warmup, 'steps': args.steps, 'events': events}
try:
    for step in range(args.warmup):
        tick('warmup', step)
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
    ):
        for step in range(args.steps):
            tick('profile', step)
        torch.npu.synchronize()
    summary['status'] = 'DIAGNOSTIC_ONLY'
finally:
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    model.close()
