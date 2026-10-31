import numpy as np
from exactsample import RequestParams, RngKey, verify_reference


class ExactSampleReferenceAdapter:
    version = 'contract-v1.0/reference'

    def sample(self, logits, *, seed, position=0, **params):
        rows = np.asarray(logits)
        batch = len(rows)
        result = verify_reference(rows, np.arange(batch + 1), np.empty((batch, 0), np.int64),
                                  params=RequestParams(**params),
                                  rng=RngKey(seed, 0, np.arange(position, position + batch)))
        return result


class ExactSampleCudaAdapter:
    version = 'contract-v1.0/cuda'

    def __init__(self, dispatch='auto'):
        from conformance.common import cuda_runtime
        from exactsample.cuda import load_library
        self.torch = cuda_runtime()
        load_library()
        self.dispatch = dispatch

    def sample(self, logits, *, seed, position=0, temperature=1., top_k=0, top_p=1., min_p=0.):
        from exactsample.cuda import CudaRequestParams, CudaRngKey, plan, verify
        torch = self.torch
        rows = torch.as_tensor(logits, dtype=torch.float32, device='cuda')
        count, vocab = rows.shape
        work = plan(count, count, 0, vocab)
        full = lambda value, kind: torch.full((count,), value, dtype=kind, device='cuda')
        params = CudaRequestParams(full(temperature, torch.float64), full(top_k, torch.int64),
                                   full(top_p, torch.float64), full(min_p, torch.float64),
                                   full(0, torch.int64))
        signed_seed = seed if seed < 2**63 else seed - 2**64
        rng = CudaRngKey(full(signed_seed, torch.int64), full(0, torch.int64),
                         torch.arange(position, position + count, dtype=torch.int64, device='cuda'))
        result = verify(rows, torch.arange(count + 1, dtype=torch.int64, device='cuda'),
                        torch.empty((count, 0), dtype=torch.int64, device='cuda'),
                        params=params, rng=rng, plan=work, out=work.allocate_output(), dispatch=self.dispatch)
        work.synchronize()
        from exactsample import VerifyOutput
        return VerifyOutput(*(getattr(result, field).cpu().numpy() for field in
                              ['tokens', 'accepted_len', 'status', 'path_used']))
