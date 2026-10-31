"""Native baseline statistics; unsupported coupled semantics stay explicit."""
import numpy as np
from conformance.common import UnsupportedCase


def counts(adapter, backend, cell, samples, seed_start, stream, position, batch_size):
    import torch
    kind = cell['kind']
    if kind == 'coupled':
        raise UnsupportedCase('baseline verification API has no exactsample same-key coupled mode')
    counter = np.zeros(len(cell['expected']), dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(seed_start ^ 0x51A7))
    for begin in range(0, samples, batch_size):
        size = min(batch_size, samples - begin)
        seeds = np.arange(seed_start + begin, seed_start + begin + size, dtype=np.int64)
        if kind == 'sample':
            kwargs = dict(seed=seeds, **cell.get('params', {}))
            kwargs['offset' if backend == 'flashinfer' else 'position'] = position
            observed = adapter.sample(np.tile(np.log(cell['weights']), (size, 1)), **kwargs)
        else:
            count = 3 if kind == 'accepted' else 1
            weights = np.asarray(cell['weights'], dtype=np.float64)
            probabilities = weights / weights.sum()
            p = np.tile(probabilities, (size, count + 1, 1)).astype(np.float32)
            ids = np.zeros((size, count), dtype=np.int64)
            if kind == 'full':
                ids[:, 0] = (generator.random(size) >= .25).astype(np.int64)
                q = np.tile([.25, .75], (size, count, 1)).astype(np.float32)
            else:
                q = np.zeros((size, count, len(weights)), np.float32)
                q[:, :, 0] = 1.
            if backend == 'flashinfer':
                result = adapter.verify_probabilities(p, q, ids, seed=seeds, offset=position)
                observed = result['emitted_draft'] if kind == 'accepted' else result['tokens'][:, 0]
            else:
                def tensor(value, dtype):
                    return torch.as_tensor(value, dtype=dtype, device='cuda')
                gathered = np.full((size, count + 1), -1, np.int64)
                gathered[:, 1:] = ids
                logits = np.log(p).reshape(-1, len(weights))
                q_logits = tensor(np.log(q), torch.float32) if kind == 'full' else None
                tokens, num_sampled = adapter.verify_processed(
                    target_logits=tensor(logits, torch.float32), draft_logits=q_logits,
                    draft_sampled=tensor(gathered.reshape(-1), torch.int64),
                    cu_rows=tensor(np.arange(size + 1) * (count + 1), torch.int32),
                    positions=tensor(np.tile(np.arange(position, position + count + 1), size), torch.int64),
                    request_mapping=tensor(np.arange(size), torch.int32),
                    row_mapping=tensor(np.repeat(np.arange(size), count + 1), torch.int32),
                    local_positions=tensor(np.tile(np.arange(count + 1), size), torch.int32),
                    temperature=tensor(np.ones(size), torch.float32), seed=tensor(seeds, torch.int64),
                    max_draft=count)
                observed = num_sampled - 1 if kind == 'accepted' else tokens[:, 0]
        observed = np.asarray(observed, dtype=np.int64)
        if np.any(observed < 0) or np.any(observed >= len(counter)):
            raise RuntimeError(f'out-of-range category in {cell["name"]}')
        counter += np.bincount(observed, minlength=len(counter))
    return counter.tolist()
