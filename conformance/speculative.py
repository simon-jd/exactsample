"""Real speculative-kernel observations against the same hand-computed p/q.

This finite smoke experiment records marginal and acceptance counts. It is not
a replacement for the statistical plan or engine integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from conformance.common import Unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['flashinfer', 'vllm_mrv2', 'exactsample_cuda'], required=True)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--flashinfer-commit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    count = args.samples
    random = np.random.Generator(np.random.PCG64(20261013))
    drafts = (random.random(count) >= .25).astype(np.int64).reshape(count, 1)
    p = np.tile(np.array([[.75, .25], [.5, .5]], np.float32), (count, 1, 1))
    q = np.tile(np.array([[[.25, .75]]], np.float32), (count, 1, 1))
    report = {'backend': args.backend, 'kind': 'speculative_observation', 'qualification': False,
              'samples': count, 'draft_generator': {'algorithm': 'NumPy PCG64', 'seed': 20261013},
              'seed_start': 65536, 'positions': [100, 101], 'p': [[.75, .25], [.5, .5]],
              'q': [[.25, .75]], 'expected_first_token': [.75, .25],
              'expected_prefix_acceptance': .5,
              'input_hashes': {name: hashlib.sha256(array.tobytes()).hexdigest()
                               for name, array in [('p', p), ('q', q), ('draft_ids', drafts)]}}
    try:
        if args.backend == 'flashinfer':
            from conformance.flashinfer import FlashInferAdapter
            adapter = FlashInferAdapter(expected_commit=args.flashinfer_commit)
            report['version'] = adapter.version
            result = adapter.verify_probabilities(p, q, drafts, seed=65536, offset=100)
            tokens = result['tokens']
            accepted = result['emitted_draft']
            report['source_commit'] = adapter.commit
            report['runtime_evidence'] = adapter.evidence()
            report['rng_mapping'] = 'native scalar seed=65536, offset=100; native batch-lane streams'
        elif args.backend == 'vllm_mrv2':
            from conformance.vllm_mrv2 import VllmMRv2Adapter
            adapter = VllmMRv2Adapter()
            torch = adapter.torch
            report['version'] = adapter.version
            def tensor(array, dtype):
                return torch.as_tensor(array, dtype=dtype, device='cuda')
            sampled = np.full((count, 2), -1, np.int64)
            sampled[:, 1] = drafts[:, 0]
            tokens, num_sampled = adapter.verify_processed(
                target_logits=tensor(np.log(p).reshape(-1, 2), torch.float32),
                draft_logits=tensor(np.log(q), torch.float32),
                draft_sampled=tensor(sampled.reshape(-1), torch.int64),
                cu_rows=tensor(np.arange(count + 1) * 2, torch.int32),
                positions=tensor(np.tile([100, 101], count), torch.int64),
                request_mapping=tensor(np.arange(count), torch.int32),
                row_mapping=tensor(np.repeat(np.arange(count), 2), torch.int32),
                local_positions=tensor(np.tile([0, 1], count), torch.int32),
                temperature=tensor(np.ones(count), torch.float32),
                seed=tensor(np.arange(65536, 65536 + count), torch.int64), max_draft=1)
            accepted = num_sampled - 1
            report['rng_mapping'] = 'native per-request seeds starting at 65536 and positions [100,101]'
        else:
            from conformance.common import cuda_runtime
            cuda_runtime()
            from exactsample import RequestParams, RngKey
            from exactsample.cuda import verify
            from tests.cuda_helpers import prepared_cuda
            inputs = prepared_cuda(np.log(p).reshape(-1, 2), np.arange(count + 1) * 2, drafts,
                                    draft_logits=np.repeat(np.log(q), 2, axis=1).reshape(-1, 2),
                                    params=RequestParams(draft_kind='full'),
                                    rng=RngKey(np.arange(65536, 65536 + count), 0,
                                               np.tile([100, 101], count)))
            output = verify(**inputs)
            inputs['plan'].synchronize()
            if np.any(output.status.cpu().numpy()):
                raise RuntimeError('exactsample returned a non-OK status')
            tokens, accepted = output.tokens.cpu().numpy(), output.accepted_len.cpu().numpy()
            report['version'] = 'contract-v1.0/cuda'
            report['rng_mapping'] = 'exactsample per-request seeds starting at 65536, stream_id=0, positions [100,101]'
        if np.any(tokens[:, 0] < 0) or np.any(tokens[:, 0] > 1):
            raise RuntimeError('out-of-vocabulary first token')
        report.update(status='observed', first_token_counts=np.bincount(tokens[:, 0], minlength=2).tolist(),
                      prefix_accepted_counts=np.bincount(accepted, minlength=2).tolist())
    except Unavailable as exc:
        report.update(status='skipped', reason=str(exc))
    except Exception as exc:
        report.update(status='error', reason=f'{type(exc).__name__}: {exc}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'status': report['status']}))
    return 77 if report['status'] == 'skipped' else int(report['status'] == 'error')


if __name__ == '__main__':
    raise SystemExit(main())
