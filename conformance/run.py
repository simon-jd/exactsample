"""Small shared-input observations, not the statistical plan's distribution test.

Missing implementations are explicit skips. Runtime failures are errors. A
third-party token mismatch is not itself a bug because its RNG is different.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone
import numpy as np
from conformance.common import Unavailable, UnsupportedCase
from conformance.exactsample import ExactSampleReferenceAdapter, ExactSampleCudaAdapter
from conformance.flashinfer import FlashInferAdapter
from conformance.vllm_mrv2 import VllmMRv2Adapter

CASES = [
    ('equal', [0., 0., 0., 0.], {}),
    ('dominant', [10., 0., 0., 0.], {}),
    ('topk_tie', [2., 1., 1., 0.], {'top_k': 2}),
    ('topp_boundary', [0., 0., 0., 0.], {'top_p': .5}),
    ('minp', [2., 1., 0., -1.], {'min_p': .4}),
    ('temperature', [2., 1., 0., -1.], {'temperature': 2.}),
    ('all_mask', [-float('inf')] * 4, {}),
    ('nan', [0., float('nan'), 0., 0.], {}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--case', choices=[item[0] for item in CASES], help='run one small reproducer')
    parser.add_argument('--flashinfer-version', choices=['0.7.0.post1', '0.7.1'])
    parser.add_argument('--flashinfer-commit', help='require an imported clean source checkout at this full SHA')
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    report = {'kind': 'observation_smoke', 'qualification': False,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'seed_start': 65536, 'samples': args.samples, 'implementations': []}
    for name, factory in [
            ('exactsample_reference', ExactSampleReferenceAdapter),
            ('exactsample_cuda', ExactSampleCudaAdapter),
            ('flashinfer', lambda: FlashInferAdapter(args.flashinfer_version, args.flashinfer_commit)),
            ('vllm_mrv2', VllmMRv2Adapter)]:
        item = {'name': name, 'cases': []}
        report['implementations'].append(item)
        try:
            adapter = factory()
        except Unavailable as exc:
            item.update(status='skipped', reason=str(exc))
            continue
        except Exception as exc:
            item.update(status='error', reason=f'{type(exc).__name__}: {exc}')
            continue
        item.update(status='observed', version=adapter.version)
        if getattr(adapter, 'commit', None):
            item['source_commit'] = adapter.commit
        for label, row, params in CASES:
            if args.case is not None and label != args.case:
                continue
            logits = np.array([row], dtype=np.float32)
            case = {'name': label, 'params': params, 'input_sha256': hashlib.sha256(logits.tobytes()).hexdigest(),
                    'dtype': str(logits.dtype), 'shape': list(logits.shape)}
            item['cases'].append(case)
            try:
                counts, statuses = {}, {}
                for trial in range(args.samples):
                    result = adapter.sample(logits, seed=65536 + trial, **params)
                    if name.startswith('exactsample_'):
                        status = str(int(result.status[0]))
                        statuses[status] = statuses.get(status, 0) + 1
                        token = int(result.tokens[0, 0])
                    else:
                        token = int(np.asarray(result).reshape(-1)[0])
                    counts[str(token)] = counts.get(str(token), 0) + 1
                case.update(status='observed', token_counts=counts, row_status_counts=statuses,
                            conclusion='Raw observation only; RNG differs, no exact-token equality claim.')
            except UnsupportedCase as exc:
                case.update(status='not_comparable', reason=str(exc))
            except Exception as exc:
                case.update(status='error', reason=f'{type(exc).__name__}: {exc}')
        if hasattr(adapter, 'evidence'):
            item['runtime_evidence'] = adapter.evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    errors = any(item['status'] == 'error' or any(c['status'] == 'error' for c in item['cases'])
                 for item in report['implementations'])
    incomplete = any(item['status'] == 'skipped' for item in report['implementations'])
    print(json.dumps({'output': str(args.output), 'errors': errors, 'missing_implementations': incomplete}))
    return int(errors or (args.require_all and incomplete))


if __name__ == '__main__':
    raise SystemExit(main())
