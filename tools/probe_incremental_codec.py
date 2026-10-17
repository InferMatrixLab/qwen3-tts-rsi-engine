"""Compare complete and incremental codec output on fixed synthetic code sequences."""
import argparse
import json
from pathlib import Path

import torch
from qwen3_tts_engine.model import CustomVoiceModel
from qwen3_tts_engine.incremental_codec import CodecState, IncrementalCodec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--float32', action='store_true', help='Separate arithmetic diagnostic; not serving precision')
parser.add_argument('--strict-fp32', action='store_true')
args = parser.parse_args()
if args.strict_fp32 and not args.float32:
    parser.error('--strict-fp32 requires --float32')
if args.strict_fp32:
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
args.output.mkdir(parents=True, exist_ok=False)
model = CustomVoiceModel(args.model, 'npu:0')
decoder = model.wrapper.model.speech_tokenizer.model.decoder
if args.float32:
    decoder.float()
incremental = IncrementalCodec(decoder)
result = {'status': 'INCONCLUSIVE', 'scope': 'synthetic codec numerical diagnostic; not speech quality or speed', 'cases': []}
result.update(conv_hf32=torch.npu.conv.allow_hf32, matmul_hf32=torch.npu.matmul.allow_hf32)
try:
    with torch.inference_mode():
        for batch in [1, 2]:
            generator = torch.Generator().manual_seed(42)
            codes = torch.randint(0, 2048, (batch, 16, 101), generator=generator).to('npu:0')
            expected = decoder(codes)
            state = CodecState()
            outputs = []
            start = 0
            for size in [2, 2, 4] + [8] * 11 + [5]:
                outputs.append(incremental.decode(codes[..., start:start + size], state))
                start += size
            actual = torch.cat(outputs, dim=-1)
            assert start == 101 and actual.shape == expected.shape
            error = (actual.float() - expected.float()).abs()
            result['cases'].append({'batch': batch, 'frames': start,
                'max_abs_error': error.max().item(), 'mean_abs_error': error.mean().item(),
                'rms_error': error.square().mean().sqrt().item(),
                'reference_rms': expected.float().square().mean().sqrt().item(),
                'finite': bool(torch.isfinite(actual).all()), 'dtype': str(expected.dtype)})
            print(result['cases'][-1], flush=True)
            if not torch.isfinite(actual).all():
                raise ValueError('Nonfinite incremental waveform')
        result['status'] = 'DIAGNOSTIC_ONLY'
finally:
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    model.close()
