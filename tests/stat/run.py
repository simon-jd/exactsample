"""Run declared cells; --smoke is deliberately not qualification evidence.

python -m tests.stat.run --smoke 1000 --output results/stat-smoke.json
python -m tests.stat.run --output results/stat-reference-qualified.json
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import os
import platform
import numpy as np
from exactsample import RequestParams, RngKey, Status, verify_reference
from tests.stat.analysis import g_test, holm, power_bound


def reference_counts(cell, samples, seed_start, stream, position):
    weights = cell['weights']
    kind = cell['kind']
    count = 3 if kind == 'accepted' else (0 if kind == 'sample' else 1)
    rows = np.tile(np.log(weights), (count + 1, 1))
    params = RequestParams(**cell.get('params', {}), draft_kind='full' if kind == 'full' else 'one_hot')
    counter = Counter()
    proposal_rng = np.random.Generator(np.random.PCG64(seed_start ^ 0x51A7))
    for trial in range(samples):
        # Proposal draws use an independent algorithm and fixed recorded key.
        draft = 0 if kind != 'full' or proposal_rng.random() < .25 else 1
        out = verify_reference(rows, [0, count + 1], np.full((1, count), draft, np.int64),
                               params=params,
                               rng=RngKey(seed_start + trial, stream, np.arange(position, position + count + 1)),
                               draft_logits=np.tile(np.log([1, 3]), (2, 1)) if kind == 'full' else None,
                               mode='coupled' if kind == 'coupled' else 'ratio')
        if out.status[0] != Status.OK:
            raise RuntimeError(f"unexpected status {out.status[0]} in {cell['name']}")
        counter[int(out.accepted_len[0] if kind == 'accepted' else out.tokens[0, 0])] += 1
    return [counter[i] for i in range(len(cell['expected']))]


def cuda_counts(cell, samples, seed_start, stream, position, batch_size, list_max=256):
    import torch
    from exactsample.cuda import verify
    from tests.cuda_helpers import prepared_cuda
    kind = cell['kind']
    drafts = 3 if kind == 'accepted' else (0 if kind == 'sample' else 1)
    params = RequestParams(**cell.get('params', {}), draft_kind='full' if kind == 'full' else 'one_hot')
    proposal_rng = np.random.Generator(np.random.PCG64(seed_start ^ 0x51A7))
    counts = np.zeros(len(cell['expected']), dtype=np.int64)
    for begin in range(0, samples, batch_size):
        size = min(batch_size, samples - begin)
        ids = np.zeros((size, drafts), np.int64)
        if kind == 'full':
            ids[:, 0] = (proposal_rng.random(size) >= .25).astype(np.int64)
        rows = np.tile(np.log(cell['weights']), (size * (drafts + 1), 1))
        offsets = np.arange(size + 1, dtype=np.int64) * (drafts + 1)
        rng = RngKey(np.arange(seed_start + begin, seed_start + begin + size, dtype=np.uint64),
                     stream, np.tile(np.arange(position, position + drafts + 1), size))
        q = np.tile(np.log([1., 3.]), (size * 2, 1)) if kind == 'full' else None
        args = prepared_cuda(rows, offsets, ids, params=params, rng=rng, draft_logits=q, dtype=torch.float64)
        args['plan'].list_max = list_max
        out = verify(**args, mode='coupled' if kind == 'coupled' else 'ratio', dispatch='auto')
        args['plan'].synchronize()
        if np.any(out.status.cpu().numpy() != int(Status.OK)):
            raise RuntimeError(f"CUDA status failure in {cell['name']}")
        observations = (out.accepted_len if kind == 'accepted' else out.tokens[:, 0]).cpu().numpy()
        if np.any(observations < 0) or np.any(observations >= len(counts)):
            raise RuntimeError(f"CUDA emitted out-of-range category in {cell['name']}")
        counts += np.bincount(observations, minlength=len(counts))
    return counts.tolist()


def run_family(plan, samples, seed_start, backend='reference', batch_size=8192, adapter=None, list_max=256):
    results = []
    for cell in plan['cells']:
        if backend == 'reference':
            counts = reference_counts(cell, samples, seed_start, plan['stream_id'], plan['position'])
        elif backend == 'cuda':
            counts = cuda_counts(cell, samples, seed_start, plan['stream_id'], plan['position'], batch_size, list_max)
        else:
            from tests.stat.external import counts as external_counts
            from conformance.common import UnsupportedCase
            try:
                counts = external_counts(adapter, backend, cell, samples, seed_start,
                                         plan['stream_id'], plan['position'], batch_size)
            except UnsupportedCase as exc:
                results.append({'cell': cell['name'], 'status': 'not_comparable', 'reason': str(exc),
                                'p_value': 1., 'rejected': False})
                continue
        results.append({"cell": cell['name'], "counts": counts, "expected": cell['expected'],
                        "target_row_float64_sha256": hashlib.sha256(np.log(cell['weights']).astype(np.float64).tobytes()).hexdigest(),
                        **g_test(counts, cell['expected'])})
        print(json.dumps({'cell': cell['name'], 'samples': samples,
                          'p_value': results[-1]['p_value']}), flush=True)
    for result, rejected in zip(results, holm([r['p_value'] for r in results], plan['family_error_rate'])):
        result['rejected'] = rejected
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=int, help='smaller non-qualifying sample count')
    parser.add_argument('--backend', choices=['reference', 'cuda', 'flashinfer', 'vllm_mrv2'], default='reference')
    parser.add_argument('--flashinfer-commit')
    parser.add_argument('--flashinfer-version', choices=['0.7.0.post1', '0.7.1'])
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--list-max', type=int, default=256,
                        help='CUDA fused path: 0 sends every row through the FP32 full-row path')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    raw_plan = (Path(__file__).parents[1] / 'stat_plan.json').read_bytes()
    plan = json.loads(raw_plan)
    samples = plan['samples_per_cell'] if args.smoke is None else args.smoke
    if samples <= 0 or args.batch_size <= 0:
        parser.error('samples and batch-size must be positive')
    started = time.time()
    adapter = None
    if args.backend == 'flashinfer':
        from conformance.flashinfer import FlashInferAdapter
        adapter = FlashInferAdapter(expected_version=args.flashinfer_version,
                                    expected_commit=args.flashinfer_commit)
    elif args.backend == 'vllm_mrv2':
        from conformance.vllm_mrv2 import VllmMRv2Adapter
        adapter = VllmMRv2Adapter()
    results = run_family(plan, samples, plan['seed_start'], args.backend, args.batch_size, adapter, args.list_max)
    failed = any(r['rejected'] for r in results)
    scope = {'reference': 'CPU float64 reference',
             'cuda': f'exactsample CUDA fused path, float64 inputs, list_max={args.list_max}',
             'flashinfer': 'FlashInfer native float32 APIs, supported cells only',
             'vllm_mrv2': 'vLLM MRv2 native float32 processing/verification, supported cells only'}[args.backend]
    report = {"scope": scope, "qualification": args.smoke is None and not failed and not any(r.get('status') == 'not_comparable' for r in results),
              "preregistered_sample_count_used": args.smoke is None,
              "stat_plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
              "samples_per_cell": samples, "rng": {k: plan[k] for k in ['seed_start', 'stream_id', 'position']},
              "power": power_bound(plan), "cells": results, "initial_failure": failed}
    report['environment'] = {'python': platform.python_version(), 'numpy': np.__version__}
    import scipy
    report['environment']['scipy'] = scipy.__version__
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [Path('python/exactsample/reference.py'), Path('python/exactsample/rng.py'),
                                           Path('csrc/verify/verify.cu'), Path('csrc/verify/fused.cuh'),
                                           Path('include/exactsample/philox.h')]}
    if args.backend != 'reference':
        import torch
        report['environment'].update(torch=torch.__version__, cuda=torch.version.cuda,
                                     gpu=torch.cuda.get_device_name(),
                                     capability=list(torch.cuda.get_device_capability()))
        library = os.environ.get('EXACTSAMPLE_LIBRARY')
        if library and args.backend == 'cuda':
            report['library_sha256'] = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    if failed and args.smoke is None:
        report['diagnostic_retry'] = run_family(plan, samples, plan['retry']['seed_start'], args.backend,
                                                args.batch_size, adapter, args.list_max)
    report['elapsed_seconds'] = time.time() - started
    if adapter is not None:
        report['baseline_version'] = adapter.version
        report['baseline_source_commit'] = getattr(adapter, 'commit', None)
        report['baseline_rng_mapping'] = ('native per-row seed/offset tensors; stream_id not representable'
                                           if args.backend == 'flashinfer' else
                                           'native per-request seeds and absolute positions; stream_id not representable')
        if hasattr(adapter, 'evidence'):
            report['baseline_runtime_evidence'] = adapter.evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({"output": str(args.output), "qualification": report['qualification'], "failed": failed}))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
