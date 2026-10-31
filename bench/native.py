"""Timed native baselines, starting at the same raw device logits as exactsample.

Preparation below only constructs static layout/parameter metadata. Conversion,
masking, filtering, probability materialization and padding occur inside calls.
These implementations have different RNG and boundary semantics from exactsample.
"""
from types import SimpleNamespace
import importlib.metadata
import numpy as np


def prepare(name, logits, offsets, drafts, params, rng, *, mask=None, draft_logits=None):
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
    row_temperatures = temperatures[row_mapping.long(), None]
    k = torch.where(params.top_k == 0, vocab, params.top_k).to(torch.int32)
    p = params.top_p.float()
    has_k = bool((params.top_k != 0).any().item())
    has_p = bool((params.top_p < 1).any().item())
    has_min_p = bool((params.min_p > 0).any().item())
    # Native vLLM stores draft i at row start+i+1; row start is context.
    native_drafts = torch.zeros(logits.shape[0], dtype=torch.int64, device=device)
    for i, count in enumerate(counts):
        native_drafts[int(bounds[i]) + 1:int(bounds[i + 1])].copy_(drafts[i, :int(count)])

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

        def call():
            tensor = copy_and_mask()
            apply_temperature(tensor, row_mapping, temperatures)
            if has_min_p:
                apply_min_p(tensor, row_mapping, min_p)
            tensor = apply_top_k_top_p(tensor, k[row_mapping.long()] if has_k else None,
                                      p[row_mapping.long()] if has_p else None)
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
                    "scope": "raw copy/mask + fp32 conversion + temperature/min-p/top-k/top-p + native sampling/verification",
                    "semantics": "ratio verification; native tie/threshold behavior; different RNG from exactsample"}
    elif name == "flashinfer":
        from flashinfer import sampling
        version = importlib.metadata.version("flashinfer-python")
        if version not in ("0.7.0.post1", "0.7.1"):
            raise ValueError(f"unlocked FlashInfer version {version}")
        row_k, row_p = k[row_mapping.long()], p[row_mapping.long()]
        row_min_p = min_p[row_mapping.long(), None]
        positions = rng.positions[offsets[:-1].long()]

        def call():
            tensor = copy_and_mask() / row_temperatures
            if not max_draft and not has_min_p:
                tokens = sampling.top_k_top_p_sampling_from_logits(
                    tensor, top_k=row_k, top_p=row_p, filter_apply_order="top_k_first",
                    deterministic=True, seed=rng.seed, offset=positions)
                return SimpleNamespace(tokens=tokens[:, None],
                                       accepted_len=torch.zeros(requests, dtype=torch.int32, device=device))
            probabilities = torch.softmax(tensor, dim=-1)
            if has_min_p:
                # FlashInfer has no public combined min-p renormalization op.
                probabilities = torch.where(probabilities >= probabilities.amax(-1, keepdim=True) * row_min_p,
                                            probabilities, 0.)
                probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            if has_k:
                probabilities = sampling.top_k_renorm_probs(probabilities, row_k)
            if has_p:
                probabilities = sampling.top_p_renorm_probs(probabilities, row_p)
            if not max_draft:
                tokens = sampling.sampling_from_probs(probabilities, deterministic=True,
                                                      seed=rng.seed, offset=positions)
                return SimpleNamespace(tokens=tokens[:, None],
                                       accepted_len=torch.zeros(requests, dtype=torch.int32, device=device))
            # Public chain API is padded. Include all dense allocations/copies.
            target = torch.zeros((requests, max_draft + 1, vocab), dtype=torch.float32, device=device)
            target[:, :, 0] = 1.
            ids = torch.zeros((requests, max_draft), dtype=torch.int32, device=device)
            for i, count in enumerate(counts):
                target[i, :int(count) + 1].copy_(probabilities[int(bounds[i]):int(bounds[i + 1])])
                ids[i, :int(count)].copy_(drafts[i, :int(count)])
            q = torch.zeros((requests, max_draft, vocab), dtype=torch.float32, device=device)
            q.scatter_(2, ids.long().unsqueeze(-1), 1.)
            emitted = torch.zeros(requests, dtype=torch.int32, device=device)
            tokens = sampling.chain_speculative_sampling(q, ids, target,
                        maybe_output_emitted_draft_token_num=emitted,
                        deterministic=True, seed=rng.seed, offset=positions)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            # At the original bonus row padded one-hot verification still draws
            # from that row's target law; discard any later artificial positions.
            lengths = torch.as_tensor(counts, dtype=torch.int32, device=device)
            emitted = torch.minimum(emitted, lengths)
            tokens.masked_fill_(torch.arange(max_draft + 1, device=device)[None, :] > emitted[:, None], -1)
            return SimpleNamespace(tokens=tokens, accepted_len=emitted)

        metadata = {"version": version, "rng": "native FlashInfer seed/offset API",
                    "scope": "raw copy/mask + fp32 conversion + temperature/min-p + public filtering + softmax/padding/one-hot q + native chain",
                    "semantics": "ratio verification; native tie/threshold behavior; min-p preparation uses PyTorch; padding cost included"}
    else:
        raise ValueError(name)
    return call, metadata
