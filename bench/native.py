"""Timed native baselines, starting from the same raw bf16 logits as exactsample.

Each library is configured the way its own engine integration calls it, and
everything a caller can keep across decode steps is prepared outside the
timed call, as exactsample's plan is: index tensors, parameter tensors, and for
FlashInfer verification the one-hot draft buffer.

vllm_mrv2 follows vLLM 0.31's model-runner-v2 sampler for requests with
explicit seeds (vllm/v1/worker/gpu/sample/sampler.py): an FP32 copy of the
logits (sampler.py:239), temperature and min-p in place, a per-step gather
of each row's top-k/top-p (states.py:110-111), the Triton top-k/top-p
kernel (sample/ops/topk_topp_sampler.py:455), then the Triton Gumbel
sampler, or the Triton rejection sampler for drafts.

FlashInfer has two public entry points for sampling, and both are timed:
  flashinfer:logits  scale the bf16 logits by 1/T, then sampling_from_logits
                     or top_k_top_p_sampling_from_logits (k = V for top-p only);
  flashinfer:probs   sampling.softmax with per-row temperature, then
                     sampling_from_probs, top_p_ or top_k_top_p_sampling_from_probs.
Verification has one path: softmax, top-k/top-p renormalization, and
chain_speculative_sampling on the [requests, drafts + 1, V] view of the rows
(no copy when every request has the same number of drafts, one gather
otherwise) against a persistent one-hot draft buffer that each call marks
and clears with two scatters.

These implementations use their own RNGs and tie/threshold rules, so only
their speed is compared, never their tokens.
"""
from types import SimpleNamespace
import importlib.metadata
import numpy as np

FLASHINFER_VERSIONS = ("0.7.0.post1", "0.7.1")


def prepare(name, logits, offsets, drafts, params, rng, *, mask=None, draft_logits=None):
    """(call, metadata) for one cell, or None if the implementation has no path for it."""
    import torch

    if draft_logits is not None or bool((params.draft_kind != 0).any().item()):
        raise ValueError("native timing currently supports actual one-hot proposals only")
    if bool((params.temperature <= 0).any().item()):
        raise ValueError("native timing currently requires positive temperatures")
    bounds = offsets.cpu().numpy()
    counts = np.diff(bounds) - 1
    requests, vocab = len(counts), logits.shape[1]
    max_draft = int(counts.max(initial=0))
    device = logits.device
    mapping = torch.arange(requests, dtype=torch.int32, device=device)
    row_mapping = torch.repeat_interleave(mapping, torch.as_tensor(counts + 1, device=device))
    local = torch.cat([torch.arange(c + 1, device=device, dtype=torch.int32) for c in counts])
    cu_rows = offsets.to(torch.int32)
    temperatures = params.temperature.float()
    min_p = params.min_p.float()
    k = torch.where(params.top_k == 0, vocab, params.top_k).to(torch.int32)
    p = params.top_p.float()
    has_k = bool((params.top_k != 0).any().item())
    has_p = bool((params.top_p < 1).any().item())
    has_min_p = bool((params.min_p > 0).any().item())

    def copy_and_mask():
        # xgrammar modifies in place, so include the defensive raw-logit copy.
        if mask is None:
            return logits.to(torch.float32, copy=True)
        import xgrammar
        raw = logits.clone()
        xgrammar.apply_token_bitmask_inplace(raw, mask, vocab_size=vocab)
        return raw.float()

    if name == "vllm_mrv2":
        version = importlib.metadata.version("vllm")
        if version != "0.31.0":
            raise ValueError(f"locked vLLM 0.31.0 required, found {version}")
        from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
        from vllm.v1.worker.gpu.sample.min_p import apply_min_p
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

        # Native vLLM stores draft i at row start+i+1; row start is context.
        native_drafts = torch.zeros(logits.shape[0], dtype=torch.int64, device=device)
        for i, count in enumerate(counts):
            native_drafts[int(bounds[i]) + 1:int(bounds[i + 1])].copy_(drafts[i, :int(count)])

        def call():
            tensor = copy_and_mask()
            apply_temperature(tensor, row_mapping, temperatures)
            if has_min_p:
                apply_min_p(tensor, row_mapping, min_p)
            tensor = apply_top_k_top_p(tensor, k[row_mapping] if has_k else None,
                                      p[row_mapping] if has_p else None)
            if max_draft:
                tokens, num_sampled = rejection_sample(
                    tensor, None, native_drafts, cu_rows, rng.positions, mapping,
                    row_mapping, local, temperatures, rng.seed, max_draft,
                    use_fp64=False, use_block_verification=False)
                accepted = num_sampled - 1
            else:
                tokens = gumbel_sample(tensor, mapping, temperatures, rng.seed, rng.positions,
                                       apply_temperature=False, is_drafting=False, use_fp64=False)[:, None]
                accepted = torch.zeros(requests, dtype=torch.int32, device=device)
            return SimpleNamespace(tokens=tokens, accepted_len=accepted)

        metadata = {"version": version, "rng": "native Murmur3 / rejection Philox; fp32 native default",
                    "scope": "fp32 copy + temperature/min-p + per-step top-k/top-p gather + Triton "
                             "top-k/top-p + Triton sampling/verification, as MRv2 with explicit seeds",
                    "semantics": "ratio verification; native tie/threshold behavior; different RNG from exactsample"}
        return call, metadata

    api = name.partition(":")[2]
    if name.partition(":")[0] != "flashinfer" or api not in ("logits", "probs"):
        raise ValueError(name)
    from flashinfer import sampling
    version = importlib.metadata.version("flashinfer-python")
    if version not in FLASHINFER_VERSIONS:
        raise ValueError(f"unlocked FlashInfer version {version}")
    if mask is not None or has_min_p:
        raise ValueError("FlashInfer timing covers unmasked logits without min-p")
    if max_draft and api == "logits":
        return None  # chain_speculative_sampling takes probabilities only
    row_k, row_p = k[row_mapping], p[row_mapping]
    row_t = temperatures[row_mapping]
    positions = rng.positions[offsets[:-1].long()]

    def target_probs():
        probabilities = sampling.softmax(logits, temperature=row_t)
        if has_k:
            probabilities = sampling.top_k_renorm_probs(probabilities, row_k)
        if has_p:
            probabilities = sampling.top_p_renorm_probs(probabilities, row_p)
        return probabilities

    if max_draft:
        uniform = bool((counts == max_draft).all())
        # Rows of request i padded to max_draft + 1; padding repeats the
        # request's last (bonus) row and is never emitted.
        pad = torch.as_tensor(np.concatenate([np.minimum(np.arange(max_draft + 1), c) + b
                                              for b, c in zip(bounds[:-1], counts)]), device=device)
        ids = torch.zeros((requests, max_draft), dtype=torch.int32, device=device)
        for i, count in enumerate(counts):
            ids[i, :int(count)].copy_(drafts[i, :int(count)])
        slots = ids.long().unsqueeze(-1)
        q = torch.zeros((requests, max_draft, vocab), dtype=torch.float32, device=device)
        emitted = torch.zeros(requests, dtype=torch.int32, device=device)
        lengths = torch.as_tensor(counts, dtype=torch.int32, device=device)
        later = torch.arange(max_draft + 1, device=device)[None, :]

        def call():
            probabilities = target_probs()
            target = (probabilities.view(requests, max_draft + 1, vocab) if uniform
                      else probabilities.index_select(0, pad).view(requests, max_draft + 1, vocab))
            q.scatter_(2, slots, 1.)
            tokens = sampling.chain_speculative_sampling(q, ids, target,
                        maybe_output_emitted_draft_token_num=emitted,
                        deterministic=True, seed=rng.seed, offset=positions)
            q.scatter_(2, slots, 0.)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            accepted = torch.minimum(emitted, lengths)
            tokens.masked_fill_(later > accepted[:, None], -1)
            return SimpleNamespace(tokens=tokens, accepted_len=accepted)

        scope = "softmax(T) + top-k/top-p renorm + chain_speculative_sampling on a view, persistent one-hot q"
    else:
        zero = torch.zeros(requests, dtype=torch.int32, device=device)
        inverse_t = (1.0 / row_t)[:, None]

        def call():
            if api == "logits":
                scaled = logits * inverse_t  # bf16 * fp32 -> fp32, one kernel
                if has_k or has_p:
                    tokens = sampling.top_k_top_p_sampling_from_logits(
                        scaled, top_k=row_k, top_p=row_p, filter_apply_order="top_k_first",
                        deterministic=True, seed=rng.seed, offset=positions)
                else:
                    tokens = sampling.sampling_from_logits(scaled, deterministic=True,
                                                           seed=rng.seed, offset=positions)
            else:
                probabilities = sampling.softmax(logits, temperature=row_t)
                if has_k:
                    tokens = sampling.top_k_top_p_sampling_from_probs(
                        probabilities, top_k=row_k, top_p=row_p, filter_apply_order="top_k_first",
                        deterministic=True, seed=rng.seed, offset=positions)
                elif has_p:
                    tokens = sampling.top_p_sampling_from_probs(probabilities, row_p, deterministic=True,
                                                                seed=rng.seed, offset=positions)
                else:
                    tokens = sampling.sampling_from_probs(probabilities, deterministic=True,
                                                          seed=rng.seed, offset=positions)
            return SimpleNamespace(tokens=tokens[:, None], accepted_len=zero)

        scope = ("1/T scale + sampling_from_logits or top_k_top_p_sampling_from_logits" if api == "logits"
                 else "softmax(T) + sampling_from_probs or top_p_/top_k_top_p_sampling_from_probs")
    metadata = {"version": version, "rng": "native FlashInfer seed/offset API", "api": api, "scope": scope,
                "semantics": "ratio verification; native tie/threshold behavior; different RNG from exactsample"}
    return call, metadata
