"""Exercise native request-slot reuse, capacity growth and complete streaming PCM."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import wave
from qwen3_tts_engine.continuous import ContinuousModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--workload', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--slots', type=int, default=2)
parser.add_argument('--capacity', type=int, default=64)
parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 64, 65])
parser.add_argument('--cancel-first-at', type=int)
parser.add_argument('--incremental-codec', action='store_true')
parser.add_argument('--batch-prefill', action='store_true')
parser.add_argument('--native-kv-update', action='store_true')
parser.add_argument('--triton-kv-update', action='store_true')
parser.add_argument('--triton-talker-rope', action='store_true')
parser.add_argument('--codec-chunk-frames', type=int, choices=[8, 16], default=8)
parser.add_argument('--native-gqa', action='store_true')
parser.add_argument('--graph-prefill', action='store_true')
parser.add_argument('--text-only-rope', action='store_true')
parser.add_argument('--prewarm-codec', action='store_true')
parser.add_argument('--bucket-kv', action='store_true')
parser.add_argument('--graph-embeddings', action='store_true')
parser.add_argument('--bucket-prefill', action='store_true')
parser.add_argument('--decode-bucket-step', type=int, default=0)
parser.add_argument('--batch-audio-transfer', action='store_true')
parser.add_argument('--reference-single', action='store_true')
parser.add_argument('--graph-sampling', action='store_true')
parser.add_argument('--native-rope', action='store_true')
parser.add_argument('--codec-code-buffer', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
data = args.workload.read_bytes()
payloads = [dict(json.loads(line), instructions='') for line in data.splitlines()]
model = ContinuousModel(args.model, 'npu:0', max_slots=args.slots, initial_capacity=args.capacity,
                        incremental_codec=args.incremental_codec, batch_prefill=args.batch_prefill,
                        native_kv_update=args.native_kv_update, triton_kv_update=args.triton_kv_update, triton_talker_rope=args.triton_talker_rope, codec_chunk_frames=args.codec_chunk_frames, native_gqa=args.native_gqa,
                        graph_prefill=args.graph_prefill, text_only_rope=args.text_only_rope,
                        prewarm_codec=args.prewarm_codec, bucket_kv=args.bucket_kv, graph_embeddings=args.graph_embeddings, bucket_prefill=args.bucket_prefill, decode_bucket_step=args.decode_bucket_step, batch_audio_transfer=args.batch_audio_transfer, reference_single=args.reference_single, graph_sampling=args.graph_sampling, native_rope=args.native_rope, codec_code_buffer=args.codec_code_buffer)
result = {'status': 'INCONCLUSIVE', 'scope': 'native scheduling correctness probe; not service performance',
          'requests': [], 'events': [],
          'configuration': {'slots': args.slots, 'capacity': args.capacity,
                            'incremental_codec': args.incremental_codec, 'batch_prefill': args.batch_prefill,
                            'native_kv_update': args.native_kv_update, 'triton_kv_update': args.triton_kv_update, 'triton_talker_rope': args.triton_talker_rope, 'codec_chunk_frames': args.codec_chunk_frames,
                            'native_gqa': args.native_gqa,
                            'graph_prefill': args.graph_prefill,
                            'text_only_rope': args.text_only_rope, 'prewarm_codec': args.prewarm_codec, 'bucket_kv': args.bucket_kv, 'graph_embeddings': args.graph_embeddings, 'bucket_prefill': args.bucket_prefill, 'decode_bucket_step': args.decode_bucket_step, 'batch_audio_transfer': args.batch_audio_transfer, 'reference_single': args.reference_single, 'graph_sampling': args.graph_sampling, 'native_rope': args.native_rope, 'codec_code_buffer': args.codec_code_buffer,
                            'cancel_first_at': args.cancel_first_at},
          'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((Path(__file__).absolute().parent.parent / 'qwen3_tts_engine').glob('*.py'))}}
(args.output / 'runspec.json').write_text(json.dumps({'workload': str(args.workload),
    'workload_sha256': hashlib.sha256(data).hexdigest(), 'sample_rate': 24000, 'frame_rate': 12.5}, indent=2)+'\n')
starts, pieces, chunks = {}, {}, {}
cancelled_ids = set()
def emit(identifier, pcm, rate):
    assert identifier not in cancelled_ids
    assert rate == 24000 and len(pcm) > 0 and len(pcm) % 2 == 0
    pieces[identifier].append(pcm)
    chunks[identifier].append({'seconds': time.perf_counter() - starts[identifier], 'bytes': len(pcm)})
try:
    pending = list(args.indices)
    completed = set()
    while pending or model.states:
        cancelled = []
        if args.cancel_first_at is not None and args.indices[0] not in cancelled_ids:
            for state in model.states:
                if state.identifier == args.indices[0] and state.frames >= args.cancel_first_at:
                    cancelled = [state.identifier]
                    cancelled_ids.add(state.identifier)
                    break
        submissions = []
        while pending and len(model.states) - len(cancelled) + len(submissions) < args.slots:
            index = pending.pop(0)
            starts[index] = time.perf_counter()
            pieces[index], chunks[index] = [], []
            submissions.append((index, payloads[index]))
        outputs = model.tick(submissions, emit, cancelled)
        if submissions or outputs or cancelled:
            result['events'].append({'admitted': [i for i, _ in submissions], 'finished': [i for i, _ in outputs],
                                     'cancelled': cancelled, 'active': [state.identifier for state in model.states],
                                     'capacity': model.arena.capacity})
            print(result['events'][-1], flush=True)
        for index, output in outputs:
            assert index not in completed
            assert b''.join(pieces[index]) == output['pcm']
            assert len(output['pcm']) == output['codec_frames'] * 1920 * 2
            assert output['finish_reason'] == 'stop'
            completed.add(index)
            with wave.open(str(args.output / f'{index:05d}.wav'), 'wb') as audio:
                audio.setparams((1, 2, output['sample_rate'], 0, 'NONE', 'not compressed'))
                audio.writeframes(output['pcm'])
            result['requests'].append({'index': index, 'success': True,
                'pcm_sha256': hashlib.sha256(output['pcm']).hexdigest(), 'pcm_bytes': len(output['pcm']),
                'codec_frames': output['codec_frames'], 'finish_reason': output['finish_reason'],
                'audio_seconds': len(output['pcm']) / 48000, 'elapsed_seconds': time.perf_counter() - starts[index],
                'chunks': chunks[index]})
            (args.output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert completed == set(args.indices) - cancelled_ids
    if args.cancel_first_at is not None:
        assert cancelled_ids == {args.indices[0]}
    result['cancelled'] = sorted(cancelled_ids)
    result['status'] = 'PASS'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    result['captured_shapes'] = {'prefill': sorted(model.prefills), 'talker': sorted(model.graphs),
                                 'predictor': sorted(model.predictor.captures)}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    model.close()
