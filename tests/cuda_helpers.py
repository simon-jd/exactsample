"""Tensor setup for CUDA tests (allocation is intentionally outside verify)."""
import numpy as np
from exactsample import RequestParams, RngKey


def prepared_cuda(logits, cu_rows, draft_ids, *, params=None, rng=None, mask=None,
                  draft_logits=None, dtype=None, strided=False):
    import torch
    from exactsample.cuda import CudaRequestParams, CudaRngKey, plan
    params = params or RequestParams()
    data = np.asarray(logits)
    offsets = np.asarray(cu_rows)
    ids = np.asarray(draft_ids, dtype=np.int64)
    rows, vocab = data.shape
    requests = len(offsets) - 1
    rng = rng or RngKey(1234, 3, np.arange(rows))
    dtype = dtype or torch.float64

    def tensor(value, kind):
        result = torch.as_tensor(np.array(value, copy=True), dtype=kind, device='cuda')
        if strided and result.ndim and result.shape[-1]:
            backing = torch.empty((*result.shape[:-1], result.shape[-1] * 2), dtype=kind, device='cuda')
            backing[..., ::2] = result
            result = backing[..., ::2]
        return result

    def per(value):
        array = np.asarray(value, dtype=object)
        return [array.item()] * requests if array.ndim == 0 else list(array)

    seeds = [int(s) if int(s) < 2**63 else int(s) - 2**64 for s in per(rng.seed)]
    gpu_params = CudaRequestParams(
        tensor(per(params.temperature), torch.float64), tensor(per(params.top_k), torch.int64),
        tensor(per(params.top_p), torch.float64), tensor(per(params.min_p), torch.float64),
        tensor([0 if k == 'one_hot' else 1 for k in per(params.draft_kind)], torch.int64))
    gpu_rng = CudaRngKey(tensor(seeds, torch.int64), tensor(per(rng.stream_id), torch.int64),
                         tensor(rng.positions, torch.int64))
    work = plan(rows, requests, ids.shape[1], vocab, reference=True)
    args = dict(target_logits=tensor(data, dtype), cu_rows=tensor(offsets, torch.int64),
                draft_ids=tensor(ids, torch.int64), params=gpu_params, rng=gpu_rng,
                plan=work, out=work.allocate_output())
    if mask is not None:
        args['mask'] = tensor(mask, torch.int32)
    if draft_logits is not None:
        args['draft_logits'] = tensor(draft_logits, dtype)
    return args
