"""Explicit >1 GB FP32-equivalent row-boundary experiment (about 5.4 GB workspace).

Use separately from normal/sanitizer tests: python -m tests.invariance.large_gpu
Requires the actual CUDA library and GPU. It is never silently skipped.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from exactsample.cuda import CudaRequestParams, CudaRngKey, plan, verify


def run(rows, begin=0):
    vocab = 151936
    count = len(rows)
    work = plan(count, count, 0, vocab)
    params = CudaRequestParams(torch.ones(count, dtype=torch.float64, device='cuda'),
                               torch.full((count,), 20, dtype=torch.int64, device='cuda'),
                               torch.ones(count, dtype=torch.float64, device='cuda'),
                               torch.zeros(count, dtype=torch.float64, device='cuda'),
                               torch.zeros(count, dtype=torch.int64, device='cuda'))
    rng = CudaRngKey(torch.full((count,), 123, dtype=torch.int64, device='cuda'),
                     torch.zeros(count, dtype=torch.int64, device='cuda'),
                     torch.arange(begin, begin + count, dtype=torch.int64, device='cuda'))
    result = verify(rows, torch.arange(count + 1, dtype=torch.int64, device='cuda'),
                    torch.empty((count, 0), dtype=torch.int64, device='cuda'),
                    params=params, rng=rng, plan=work, out=work.allocate_output(), dispatch='fused')
    work.synchronize()
    if result.status.count_nonzero().item():
        raise RuntimeError('non-OK status')
    return result.tokens.cpu().numpy()


def main():
    count, vocab, split = 1770, 151936, 1766
    # Distinct real allocated bf16 rows exceed 1 GB in vLLM's FP32 copy.
    generator = torch.Generator(device='cuda').manual_seed(910)
    rows = torch.randn((count, vocab), generator=generator, device='cuda', dtype=torch.bfloat16)
    whole = run(rows)
    torch.cuda.empty_cache()
    parts = np.concatenate([run(rows[:split]), run(rows[split:], split)])
    np.testing.assert_array_equal(whole, parts)
    report = {'status': 'passed', 'rows': count, 'vocab': vocab, 'split_after': split,
              'fp32_equivalent_bytes': count * vocab * 4, 'actual_logits_dtype': 'bfloat16',
              'data_seed': 910, 'rng': {'seed': 123, 'stream_id': 0, 'positions': [0, count - 1]},
              'scope': 'exactsample CUDA whole vs split; no vLLM engine execution',
              'tokens_sha256': hashlib.sha256(whole.tobytes()).hexdigest()}
    path = Path('results/large-gpu-invariance.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
