"""Observe native request identity under batch reorder, insertion, and replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from conformance.common import Unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['flashinfer', 'vllm_mrv2', 'exactsample_cuda'], required=True)
    parser.add_argument('--flashinfer-commit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'kind': 'native_rng_identity', 'backend': args.backend, 'qualification': False,
              'mapping': 'same native seed and position/offset for each request; no cross-library token equality'}
    try:
        adapter = None
        if args.backend == 'flashinfer':
            from conformance.flashinfer import FlashInferAdapter
            adapter = FlashInferAdapter(expected_commit=args.flashinfer_commit)
        elif args.backend == 'vllm_mrv2':
            from conformance.vllm_mrv2 import VllmMRv2Adapter
            adapter = VllmMRv2Adapter()
        rows = np.random.default_rng(901).normal(size=(32, 128)).astype(np.float32)
        seeds = np.arange(65536, 65568, dtype=np.int64)
        positions = np.arange(100, 132, dtype=np.int64)
        report['inputs'] = {'logits_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
                            'data_rng': 'NumPy default_rng(901)', 'seeds': seeds.tolist(),
                            'positions': positions.tolist()}

        def sample(data, keys, pos):
            if adapter is not None:
                return adapter.sample(data, seed=keys,
                                      **({'offset': pos} if args.backend == 'flashinfer' else {'position': pos}))
            from exactsample import RngKey
            from exactsample.cuda import verify
            from tests.cuda_helpers import prepared_cuda
            inputs = prepared_cuda(data, np.arange(len(data) + 1), np.empty((len(data), 0), np.int64),
                                    rng=RngKey(keys, 0, pos))
            result = verify(**inputs)
            inputs['plan'].synchronize()
            if np.any(result.status.cpu().numpy()):
                raise RuntimeError('non-OK exactsample status')
            return result.tokens[:, 0].cpu().numpy()

        original = sample(rows, seeds, positions)
        order = np.arange(len(rows))[::-1]
        reordered = sample(rows[order].copy(), seeds[order].copy(), positions[order].copy())[::-1]
        replayed = sample(rows, seeds, positions)
        individual = np.concatenate([sample(rows[i:i + 1], seeds[i:i + 1], positions[i:i + 1])
                                     for i in range(len(rows))])
        report.update(status='observed', version=adapter.version if adapter else 'contract-v1.0/cuda',
                      original=original.tolist(), reordered=reordered.tolist(), individual=individual.tolist(),
                      reordered_differences=int(np.count_nonzero(original != reordered)),
                      single_request_differences=int(np.count_nonzero(original != individual)),
                      replay_differences=int(np.count_nonzero(original != replayed)))
        if adapter is not None and hasattr(adapter, 'evidence'):
            report['runtime_evidence'] = adapter.evidence()
            report['source_commit'] = adapter.commit
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
