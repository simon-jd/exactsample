"""Real public API calls; no exactsample filtering, RNG, or sampling in this adapter."""
from conformance.common import Unavailable, UnsupportedCase, cuda_runtime, package_version


class FlashInferAdapter:
    def __init__(self, expected_version=None, expected_commit=None):
        if expected_commit is not None:
            import os
            if not os.environ.get('FLASHINFER_WORKSPACE_BASE'):
                raise Unavailable('fixed-commit FlashInfer requires a separate FLASHINFER_WORKSPACE_BASE to avoid release JIT cache reuse')
        self.version = package_version('flashinfer-python')
        if expected_version is not None and self.version != expected_version:
            raise Unavailable(f'FlashInfer {self.version} != requested {expected_version}')
        self.torch = cuda_runtime()
        from flashinfer import sampling
        from flashinfer.jit import env as jit_env
        self.sampling = sampling
        self.jit_cache = str(jit_env.FLASHINFER_JIT_DIR)
        self.commit = None
        if expected_commit is not None:
            # A source checkout proves the requested SHA; package versions and
            # a user-supplied label alone do not prove which main was installed.
            from pathlib import Path
            import subprocess
            origin = Path(sampling.__file__).resolve()
            checkout = next((parent for parent in origin.parents if (parent / '.git').exists()), None)
            if checkout is None:
                raise Unavailable('fixed-commit baseline needs an importable Git checkout')
            actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                            cwd=checkout, text=True).strip()
            if actual != expected_commit or dirty:
                raise Unavailable(f'FlashInfer checkout must be clean at {expected_commit}; found {actual}, dirty={bool(dirty)}')
            self.commit = actual

    def evidence(self):
        import hashlib
        from pathlib import Path
        directory = Path(self.jit_cache) / 'sampling'
        return {'jit_cache': self.jit_cache,
                'jit_artifact_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in [directory / 'sampling.so', directory / 'build.ninja']
                                        if path.is_file()}}

    def sample(self, logits, *, seed, offset=0, temperature=1., top_k=0, top_p=1., min_p=0.):
        if temperature <= 0 or (min_p and (top_k or top_p != 1)):
            raise UnsupportedCase('public logits API does not combine greedy/min-p/top-k/top-p')
        torch = self.torch
        tensor = torch.as_tensor(logits, dtype=torch.float32, device='cuda') / temperature
        if not isinstance(seed, int):
            seed = torch.as_tensor(seed, dtype=torch.int64, device='cuda')
            offset = torch.as_tensor(offset, dtype=torch.int64, device='cuda').expand(tensor.shape[0]).contiguous()
        if min_p:
            result = self.sampling.min_p_sampling_from_probs(torch.softmax(tensor, dim=-1),
                                                             min_p=min_p, seed=seed, offset=offset)
        else:
            result = self.sampling.top_k_top_p_sampling_from_logits(
                tensor, top_k=top_k or tensor.shape[-1], top_p=top_p,
                filter_apply_order='top_k_first', deterministic=True, seed=seed, offset=offset)
        torch.cuda.synchronize()
        return result.cpu().numpy()

    def verify_probabilities(self, target_probs, draft_probs, draft_ids, *, seed, offset=0):
        """Padded [B,K+1,V]/[B,K,V], already processed actual p and q.

        Return native output and both native counters. FlashInfer's independent
        accepted counter is NOT the prefix accepted length used by exactsample.
        """
        torch = self.torch
        p = torch.as_tensor(target_probs, dtype=torch.float32, device='cuda')
        q = torch.as_tensor(draft_probs, dtype=torch.float32, device='cuda')
        ids = torch.as_tensor(draft_ids, dtype=torch.int32, device='cuda')
        accepted = torch.zeros(ids.shape[0], dtype=torch.int32, device='cuda')
        emitted = torch.zeros_like(accepted)
        if not isinstance(seed, int):
            seed = torch.as_tensor(seed, dtype=torch.int64, device='cuda')
            offset = torch.as_tensor(offset, dtype=torch.int64, device='cuda').expand(ids.shape[0]).contiguous()
        tokens, accepted, emitted = self.sampling.chain_speculative_sampling(
            q, ids, p, maybe_output_accepted_token_num=accepted,
            maybe_output_emitted_draft_token_num=emitted, seed=seed, offset=offset)
        torch.cuda.synchronize()
        return {"tokens": tokens.cpu().numpy(), "independently_accepted": accepted.cpu().numpy(),
                "emitted_draft": emitted.cpu().numpy()}
