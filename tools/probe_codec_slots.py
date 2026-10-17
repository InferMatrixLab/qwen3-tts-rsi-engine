"""Check graph codec history across interleaved streams, capture, move and reset."""
import argparse
import json
from pathlib import Path

import torch
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.incremental_codec import CodecState, IncrementalCodec
from qwen3_tts_engine.codec_slots import CodecSlots

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--strict-fp32', action='store_true')
parser.add_argument('--prewarm', action='store_true')
parser.add_argument('--cache-constants', action='store_true')
parser.add_argument('--graph-control', action='store_true')
parser.add_argument('--triton-snake', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
if args.strict_fp32:
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
model = CustomVoiceModel(args.model, 'npu:0')
decoder = model.wrapper.model.speech_tokenizer.model.decoder
if args.strict_fp32:
    decoder.float()
engine = CodecSlots(decoder, 4, cache_constants=args.cache_constants, triton_snake=args.triton_snake)
eager = IncrementalCodec(decoder)
control = CodecSlots(decoder, 4) if args.graph_control else None
triton_calls = []
if args.triton_snake:
    original_snake = engine.eager.triton_snake
    def observed_snake(hidden, *arguments, **keywords):
        triton_calls.append(tuple(hidden.shape))
        return original_snake(hidden, *arguments, **keywords)
    engine.eager.triton_snake = observed_snake
result = {'status': 'INCONCLUSIVE', 'scope': 'synthetic slot graph diagnostic, not speech quality or speed',
          'triton_snake': args.triton_snake, 'graph_control': args.graph_control, 'strict_fp32': args.strict_fp32, 'cache_constants': args.cache_constants, 'prewarm': args.prewarm, 'cases': []}
try:
    with torch.inference_mode():
        codes = torch.randint(0, 2048, (3, 16, 101), generator=torch.Generator().manual_seed(42)).to('npu:0')
        expected_states = [CodecState(), CodecState()]
        expected, actual = [[], []], [[], []]
        slots = [3, 1]
        start = 0
        for length in [2, 2, 4] + [8] * 11 + [5]:
            chunk = codes[..., start:start + length]
            expected[0].append(eager.decode(chunk[:2], expected_states[0]))
            expected[1].append(eager.decode(chunk[2:], expected_states[1]))
            actual[0].append(engine.decode(slots, chunk[:2]))
            actual[1].append(engine.decode([2], chunk[2:]))
            if control is not None:
                torch.testing.assert_close(actual[0][-1], control.decode(slots, chunk[:2]), atol=0, rtol=0)
                torch.testing.assert_close(actual[1][-1], control.decode([2], chunk[2:]), atol=0, rtol=0)
            start += length
            if start == 8:
                if args.prewarm:
                    saved = [buffer.clone() for buffer in engine.buffers]
                    engine.prewarm(4)
                    assert len(engine.graphs) == 32
                    for buffer, original in zip(engine.buffers, saved):
                        torch.testing.assert_close(buffer, original, atol=0, rtol=0)
                    result['prewarm_live_state_exact'] = True
                if control is not None:
                    control.move(3, 0)
                    control.reset(3)
                engine.move(3, 0)
                engine.reset(3)
                slots[0] = 0
        assert engine.frames.tolist() == [101, 101, 101, 0]
        for i in range(2):
            reference, candidate = torch.cat(expected[i], -1), torch.cat(actual[i], -1)
            error = (candidate.float() - reference.float()).abs()
            row = {'batch': candidate.shape[0], 'frames': start, 'max_abs_error': error.max().item(),
                   'rms_error': error.square().mean().sqrt().item(), 'finite': bool(torch.isfinite(candidate).all())}
            result['cases'].append(row)
            print(row, flush=True)
            assert row['finite'] and candidate.shape == reference.shape
            if args.strict_fp32:
                torch.testing.assert_close(candidate, reference, atol=1e-4, rtol=1e-4)
        engine.reset(0)
        fresh = engine.decode([0], codes[0:1, :, :2])
        if control is not None:
            control.reset(0)
            torch.testing.assert_close(fresh, control.decode([0], codes[0:1, :, :2]), atol=0, rtol=0)
            result['graph_control_exact'] = True
        reference = eager.decode(codes[0:1, :, :2], CodecState())
        assert engine.frames.tolist() == [2, 101, 101, 0]
        result['reset_max_abs_error'] = (fresh.float() - reference.float()).abs().max().item()
        if args.strict_fp32:
            torch.testing.assert_close(fresh, reference, atol=1e-4, rtol=1e-4)
        if args.triton_snake and not args.strict_fp32:
            assert triton_calls, 'Triton path was never executed'
        result['status'] = 'PASS' if args.strict_fp32 else 'DIAGNOSTIC_ONLY'
except Exception as error:
    result['status'] = 'FAIL'
    result['error'] = str(error)
    raise
finally:
    result['triton_capture_calls'] = len(triton_calls)
    result['triton_shapes'] = sorted(set(triton_calls))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if control is not None: control.close()
    engine.close()
    model.close()
