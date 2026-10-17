"""Compare BF16-to-FP32 conversion before and after codec audio D2H transfer."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'INCONCLUSIVE', 'scope': 'Synthetic transfer diagnostic, not service throughput', 'cases': []}
    torch.npu.set_device('npu:0')
    torch.manual_seed(42)
    try:
        for batch in (1, 4, 8):
            for frames in (2, 8):
                wave = torch.randn(batch, frames * 1920, device='npu:0', dtype=torch.bfloat16).clamp(-1, 1)
                def convert(cpu):
                    return wave.cpu().float().numpy() if cpu else wave.float().cpu().numpy()
                expected, actual = convert(False), convert(True)
                assert np.array_equal(expected, actual)
                def pcm(x):
                    return (np.clip(x, -1, 1) * 32767).astype('<i2').tobytes()
                assert pcm(expected) == pcm(actual)
                row = {'batch': batch, 'frames': frames, 'float_exact': True, 'pcm_exact': True, 'measurements': []}
                for cpu in (False, True, True, False):
                    for _ in range(10):
                        convert(cpu)
                    durations = []
                    for _ in range(100):
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        convert(cpu)
                        durations.append((time.perf_counter() - start) * 1000)
                    row['measurements'].append({'arm': 'cpu_conversion' if cpu else 'npu_conversion', 'durations_ms': durations})
                result['cases'].append(row)
        result['status'] = 'PASS'
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
