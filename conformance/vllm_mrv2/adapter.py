"""Direct locked vLLM MRv2 kernel calls, not a reimplementation of vLLM."""
from conformance.common import Unavailable, UnsupportedCase, cuda_runtime, package_version


class VllmMRv2Adapter:
    def __init__(self):
        self.version = package_version('vllm')
        if self.version != '0.31.0':
            raise Unavailable(f'vLLM 0.31.0 required; found {self.version}')
        self.torch = cuda_runtime()
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample, apply_temperature
        from vllm.v1.worker.gpu.sample.min_p import apply_min_p
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
        self.gumbel_sample = gumbel_sample
        self.rejection_sample = rejection_sample
        self.apply_temperature = apply_temperature
        self.apply_min_p = apply_min_p
        self.apply_top_k_top_p = apply_top_k_top_p

    def sample(self, logits, *, seed, position=0, temperature=1., top_k=0, top_p=1., min_p=0.):
        torch = self.torch
        tensor = torch.as_tensor(logits, dtype=torch.float32, device='cuda')
        batch = tensor.shape[0]
        mapping = torch.arange(batch, device='cuda', dtype=torch.int32)
        temperatures = torch.full((batch,), temperature, device='cuda', dtype=torch.float32)
        seeds = torch.as_tensor(seed, device='cuda', dtype=torch.int64).expand(batch).contiguous()
        positions = torch.as_tensor(position, device='cuda', dtype=torch.int64).expand(batch).contiguous()
        # These are vLLM's actual processing kernels in SamplingStates order.
        # No exactsample filtering is used, and no whole-engine integration is claimed.
        if temperature not in (0, 1):
            self.apply_temperature(tensor, mapping, temperatures)
        if min_p:
            self.apply_min_p(tensor, mapping, torch.full((batch,), min_p, device='cuda', dtype=torch.float32))
        k = torch.full((batch,), top_k, device='cuda', dtype=torch.int32) if top_k else None
        p = torch.full((batch,), top_p, device='cuda', dtype=torch.float32) if top_p < 1 else None
        tensor = self.apply_top_k_top_p(tensor, k, p)
        tokens = self.gumbel_sample(tensor, mapping, temperatures, seeds, positions,
                                   apply_temperature=False, is_drafting=False, use_fp64=True)
        torch.cuda.synchronize()
        return tokens.cpu().numpy()

    def verify_processed(self, *, target_logits, draft_logits, draft_sampled, cu_rows, positions,
                         request_mapping, row_mapping, local_positions, temperature, seed, max_draft):
        """Native layout; caller must provide actual processed distributions.

        draft_sampled is [M]: at start+i+1 it contains draft i, matching vLLM's
        gather_draft_sampled layout. q logits are [max_requests,K,V] before
        temperature, unlike exactsample's actual q log weights. No implicit conversion.
        """
        result = self.rejection_sample(
            target_logits, draft_logits, draft_sampled, cu_rows, positions,
            request_mapping, row_mapping, local_positions, temperature, seed,
            max_draft, use_fp64=True, use_block_verification=False)
        self.torch.cuda.synchronize()
        return tuple(item.cpu().numpy() for item in result)
