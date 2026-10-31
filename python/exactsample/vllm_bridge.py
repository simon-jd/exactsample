"""Opt-in bridge for the pinned vLLM v0.31.0 Model Runner V2 local fork.

The bounded upstream patch calls this before grammar mutates raw logits.
Numerical status travels through the existing asynchronous D2H completion event;
the scheduler terminates only the failed requests without committing their tokens.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np


class ExactSampleDecodeError(RuntimeError):
    """Malformed bridge metadata: the step cannot safely be submitted."""


# Host staging of per-step parameters (see initialize_runner).
_STAGED_FIELDS = ("temperature", "top_k", "top_p", "min_p", "seeds")
_STAGED_FLOATS = ("temperature", "top_p", "min_p")
_STAGING_DEPTH = 3


def unsupported_request(params):
    """Return a reason for whole-batch fallback, without inspecting GPU data."""
    defaults = {"n": 1, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "repetition_penalty": 1.0, "min_tokens": 0}
    for name, expected in defaults.items():
        if getattr(params, name, expected) != expected:
            return name
    for name in ("logit_bias", "bad_words", "_bad_words_token_ids", "allowed_token_ids",
                 "logprobs", "logprob_token_ids", "prompt_logprobs", "prompt_logprob_token_ids",
                 "thinking_token_budget", "trace_decode_token_ids", "extra_args"):
        value = getattr(params, name, None)
        if value is not None and (name in ("logprobs", "prompt_logprobs", "thinking_token_budget") or bool(value)):
            return name
    return None


def _trace(record):
    destination = os.environ.get("EXACTSAMPLE_TRACE_PATH")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema": "exactsample.vllm_trace.v1", "pid": os.getpid(),
                  "time_ns": time.time_ns(), **record}
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")


def initialize_runner(runner):
    """Preallocate before vLLM profiles its available KV-cache memory."""
    import torch
    from .cuda import plan

    if getattr(runner, "_exactsample_bridge", None) is not None:
        return
    max_draft = int(runner.num_speculative_steps)
    max_requests = int(runner.max_num_reqs)
    max_rows = max_requests * (max_draft + 1)
    workspace = plan(max_rows, max_requests, max_draft, int(runner.vocab_size), device=runner.device)
    # Per-step parameters and draft indices: 8-byte words packed as
    # [temperature, top_k, top_p, min_p, seed] x R, then R x D draft indices.
    staged_words = max_requests * (len(_STAGED_FIELDS) + max_draft)
    staging = [torch.empty(staged_words, dtype=torch.int64, pin_memory=True) for _ in range(_STAGING_DEPTH)]
    runner._exactsample_bridge = {
        "plan": workspace, "out": workspace.allocate_output(), "requests": {}, "request_seeds": {}, "step": 0,
        "drafts": torch.empty((max_requests, max_draft), device=runner.device, dtype=torch.int64),
        "offsets": torch.empty(max_requests + 1, device=runner.device, dtype=torch.int64),
        "mask": torch.empty((max_rows, (runner.vocab_size + 31) // 32), device=runner.device, dtype=torch.int32),
        # Pinned host slots, each reused only after the event recorded behind
        # its last non-blocking copy, so the host can run ahead of the GPU.
        "staging": staging, "staging_np": [slot.numpy() for slot in staging],
        "staging_events": [torch.cuda.Event() for _ in staging], "staging_next": 0,
        "staged": torch.empty(staged_words, device=runner.device, dtype=torch.int64),
        "streams": torch.zeros(max_requests, device=runner.device, dtype=torch.int64),
        "kind": torch.zeros(max_requests, device=runner.device, dtype=torch.int64),
        "provenance": {
            "target_model": str(runner.model_config.model),
            "target_revision": getattr(runner.model_config.hf_config, "_commit_hash", None),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(runner.device),
        },
    }
    _trace({"event": "initialized", "max_rows": max_rows, "max_requests": max_requests,
            "max_draft": max_draft, "workspace_bytes": workspace.workspace_bytes})


def register_request(runner, request_id, request_index, params):
    state = runner._exactsample_bridge
    request_id = str(request_id)
    # vLLM re-adds preempted requests and assigns seed=None a new random seed.
    # Preserve the admission key independently of persistent-slot recycling.
    state["request_seeds"].setdefault(
        request_id, int(runner.sampler.sampling_states.seeds.np[int(request_index)]))
    state["requests"][int(request_index)] = {
        "id": request_id, "reason": unsupported_request(params),
        # n=1 has sub-sequence 0. A scheduling slot never enters the RNG key.
        "stream_id": 0,
    }


def finish_requests(runner, request_ids):
    """Release true completions/cancellations; the hook excludes preemptions."""
    state = runner._exactsample_bridge
    finished = {str(request) for request in request_ids}
    for request in finished:
        state["request_seeds"].pop(request, None)
    for slot, request in list(state["requests"].items()):
        if request["id"] in finished:
            del state["requests"][slot]


def grammar_mask_rows(req_ids, offsets, grammar_req_ids, bitmask, vocab, out=None):
    """Convert vLLM's constrained-request rows to dense exactsample packed rows.

    Adaptive verification is excluded before this function: offsets are the
    authoritative CPU layout, and each constrained request supplies D+1 masks.
    ``out`` (int32 [rows, words], e.g. pinned memory) receives the result.
    """
    words = (vocab + 31) // 32
    bitmask = np.asarray(bitmask)
    if bitmask.ndim != 2 or bitmask.shape[1] != words or bitmask.dtype != np.int32:
        raise ValueError("vLLM grammar bitmask has unexpected shape/dtype")
    shape = (int(offsets[-1]), words)
    if out is None:
        result = np.full(shape, -1, dtype=np.int32)
    elif out.shape != shape or out.dtype != np.int32:
        raise ValueError("grammar mask output has unexpected shape/dtype")
    else:
        result = out
        result.fill(-1)
    index_by_id = {request: index for index, request in enumerate(req_ids)}
    if len(index_by_id) != len(req_ids) or len(set(grammar_req_ids)) != len(grammar_req_ids):
        raise ValueError("duplicate request identity in grammar mapping")
    cursor = 0
    for request in grammar_req_ids:
        if request not in index_by_id:
            raise ValueError("grammar request missing from sampled batch")
        index = index_by_id[request]
        start, end = map(int, offsets[index:index + 2])
        count = end - start
        if cursor + count > bitmask.shape[0]:
            raise ValueError("grammar bitmask has fewer rows than scheduled logits")
        result[start:end] = bitmask[cursor:cursor + count]
        cursor += count
    if cursor != bitmask.shape[0]:
        raise ValueError("grammar bitmask has unused rows; unsupported layout")
    return result


def _runner_reason(runner, batch):
    parallel = runner.parallel_config
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            return name
    if runner.batch_sharder is not None or runner.adaptive_verification is not None:
        return "sharding_or_adaptive_verification"
    if runner.model_state.num_new_sampled_tokens_per_step != 1:
        return "multi_bonus_model"
    sampler = runner.sampler
    if type(sampler).__name__ != "Sampler" or len(sampler.logits_processors) != 3:
        return "custom_sampler_or_processors"
    if sampler.return_sampling_mask or sampler.trace_replay_state is not None:
        return "sampling_mask_or_trace_replay"
    if sampler.compute_nans:
        return "requested_nan_metrics"
    if runner.vllm_config.watermark_config is not None:
        return "watermark"
    spec = runner.speculative_config
    if spec is not None:
        if spec.method != "draft_model" or spec.draft_sample_method != "greedy":
            return "requires_greedy_draft_model"
        if spec.rejection_sample_method != "standard" or spec.enable_adaptive_verification:
            return "requires_standard_verification"
        if getattr(spec, "use_heterogeneous_vocab", False):
            return "heterogeneous_vocab"
    for slot, request_id in zip(batch.idx_mapping_np, batch.req_ids):
        request = runner._exactsample_bridge["requests"].get(int(slot))
        if request is None or request["id"] != str(request_id):
            return "unregistered_or_reused_request_slot"
        if request["reason"] is not None:
            return request["reason"]
    return None


def _diagnostics(state, logits, batch, positions, params_np, mask_np, record, out):
    """Snapshot optional evidence on-device; AsyncOutput performs the D2H copies."""
    trace_path = os.environ.get("EXACTSAMPLE_TRACE_PATH")
    capture_dir = os.environ.get("EXACTSAMPLE_CAPTURE_DIR")
    capture = bool(capture_dir and state["step"] < int(os.environ.get("EXACTSAMPLE_CAPTURE_LIMIT", "8")))
    if not trace_path and not capture:
        return None
    import torch

    count = batch.num_reqs
    diagnostics = {
        "record": record, "trace_path": trace_path, "capture_dir": capture_dir,
        "provenance": dict(state.get("provenance", {})),
        "gpu_arrays": {"accepted_len": out.accepted_len[:count].clone(),
                       "path_used": out.path_used[:count].clone(),
                       "positions": positions.clone()},
    }
    if capture:
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        diagnostics["gpu_arrays"].update(target_logits=logits.to(dtype).clone(),
                                          draft_ids=state["drafts"][:count].clone())
        diagnostics["cpu_arrays"] = {
            "cu_rows": np.asarray(batch.cu_num_logits_np, dtype=np.int64).copy(),
            "seeds": params_np["seeds"].view(np.uint64).copy(),
            "stream_ids": np.zeros(count, dtype=np.int64),
            "draft_kind": np.zeros(count, dtype=np.int64),
            **{key: params_np[key].copy() for key in ("temperature", "top_k", "top_p", "min_p")},
        }
        if mask_np is not None:
            diagnostics["cpu_arrays"]["mask"] = mask_np.copy()
    return diagnostics


def try_sample(runner, logits, batch, grammar_output):
    """Return a SamplerOutput, or None for an untouched whole-batch fallback."""
    import torch
    from .cuda import CudaRequestParams, CudaRngKey, verify
    from .types import Status

    if batch.num_reqs == 0 or logits.shape[0] == 0:
        return None
    state = runner._exactsample_bridge
    reason = _runner_reason(runner, batch)
    if torch.cuda.is_current_stream_capturing():
        reason = "cuda_graph_capture"
    if reason is not None:
        _trace({"event": "fallback", "reason": reason, "requests": list(batch.req_ids)})
        return None
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled

    requests, rows = batch.num_reqs, logits.shape[0]
    offsets_np = np.asarray(batch.cu_num_logits_np, dtype=np.int64)
    if offsets_np.shape != (requests + 1,) or offsets_np[0] != 0 or offsets_np[-1] != rows:
        raise ExactSampleDecodeError("vLLM ragged metadata does not describe its raw logits")
    counts = np.diff(offsets_np) - 1
    max_draft = state["plan"].max_draft
    if np.any(counts < 0) or np.any(counts > max_draft):
        raise ExactSampleDecodeError("vLLM supplied invalid draft capacities")
    # vLLM writes -1 placeholders into the draft rows of a request whose step
    # starts inside its prefill (gather_draft_sampled: position of the first
    # logits row < prefill_len), which is exactly the host's is_prefilling.
    # Those batches keep vLLM's native handling. Deciding from host metadata
    # avoids a device->host sync; a negative draft that reached the kernel
    # anyway would be reported as INVALID_INPUT, never sampled as a token.
    if np.any((counts > 0) & np.asarray(batch.is_prefilling_np[:requests], dtype=bool)):
        _trace({"event": "fallback", "reason": "prefill_draft_placeholders", "requests": list(batch.req_ids)})
        return None
    gathered, positions = gather_draft_sampled(
        batch.input_ids, batch.positions, batch.logits_indices, batch.expanded_idx_mapping,
        batch.expanded_local_pos, runner.sampler.req_states.prefill_len.gpu)

    # Stage this step's parameters and draft-row indices in one pinned slot.
    slot = state["staging_next"]
    state["staging_next"] = (slot + 1) % len(state["staging"])
    state["staging_events"][slot].synchronize()  # its previous copy has executed
    fields = len(_STAGED_FIELDS)
    words = requests * (fields + max_draft)
    host = state["staging_np"][slot][:words]
    params_np = {}
    for index, name in enumerate(_STAGED_FIELDS):
        view = host[index * requests:(index + 1) * requests]
        params_np[name] = view.view(np.float64) if name in _STAGED_FLOATS else view
    slots = batch.idx_mapping_np
    sampling = runner.sampler.sampling_states
    for name in ("temperature", "top_k", "top_p", "min_p"):
        params_np[name][:] = getattr(sampling, name).np[slots]
    params_np["seeds"][:] = [state["request_seeds"][str(request)] for request in batch.req_ids]
    params_np["top_k"][params_np["top_k"] == logits.shape[1]] = 0
    draft_index = host[fields * requests:].reshape(requests, max_draft)
    has_drafts = rows > requests
    if has_drafts:
        # Request r's proposals are the gathered rows offsets[r]+1 .. +counts[r].
        np.add(offsets_np[:-1, None] + 1, np.arange(max_draft), out=draft_index)
        draft_index[np.arange(max_draft) >= counts[:, None]] = -1
    staged = state["staged"][:words]
    staged.copy_(state["staging"][slot][:words], non_blocking=True)
    state["staging_events"][slot].record()
    device = {name: staged[index * requests:(index + 1) * requests]
              for index, name in enumerate(_STAGED_FIELDS)}
    for name in _STAGED_FLOATS:
        device[name] = device[name].view(torch.float64)

    drafts = state["drafts"][:requests]
    if has_drafts:
        index = staged[fields * requests:].view(requests, max_draft)
        drafts.copy_(gathered[index.clamp(min=0)]).masked_fill_(index < 0, -1)
    elif max_draft:
        drafts.fill_(-1)
    offsets = state["offsets"][:requests + 1]
    offsets.copy_(batch.cu_num_logits)
    mask_np = None
    mask = None
    if grammar_output is not None and grammar_output.structured_output_request_ids:
        # The caching host allocator keeps this block until its copy completes.
        host_mask = torch.empty((rows, (logits.shape[1] + 31) // 32), dtype=torch.int32,
                                pin_memory=logits.is_cuda)
        mask_np = grammar_mask_rows(batch.req_ids, offsets_np,
                                    grammar_output.structured_output_request_ids,
                                    grammar_output.grammar_bitmask, logits.shape[1], out=host_mask.numpy())
        mask = state["mask"][:rows]
        mask.copy_(host_mask, non_blocking=True)
    params = CudaRequestParams(device["temperature"], device["top_k"], device["top_p"], device["min_p"],
                               state["kind"][:requests])
    rng = CudaRngKey(device["seeds"], state["streams"][:requests], positions.to(torch.int64))
    mode = os.environ.get("EXACTSAMPLE_MODE", "coupled")
    out = verify(logits, offsets, drafts, params=params, rng=rng, mask=mask,
                 mode=mode, plan=state["plan"], out=state["out"],
                 dispatch=os.environ.get("EXACTSAMPLE_DISPATCH", "auto"),
                 top_k_top_p=bool((params_np["top_k"] > 0).any() or (params_np["top_p"] < 1).any()))
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        out.accepted_len[:requests] + 1, batch.seq_lens, batch.cu_num_logits,
        batch.idx_mapping, runner.sampler.req_states.prefill_len.gpu)
    # Keep rejection bounded to D: the draft model needs at least its already
    # committed context row. Setting rejected=D+1 would cross request bounds in
    # its last-token gather. A failed request emits/commits zero new tokens;
    # any further draft work for that request is discarded by the scheduler.
    num_sampled.masked_fill_(out.status[:requests] != int(Status.OK), 0)
    result = SamplerOutput(sampled_token_ids=out.tokens[:requests].clone(),
                           logprobs_tensors=None, num_nans=None,
                           num_sampled=num_sampled, num_rejected=num_rejected)
    # All cross-stream copies use snapshots, never the reusable plan output.
    result.exactsample_status = out.status[:requests].clone()
    result.exactsample_diagnostics = None
    if os.environ.get("EXACTSAMPLE_TRACE_PATH") or os.environ.get("EXACTSAMPLE_CAPTURE_DIR"):
        record = {"event": "verify", "step": state["step"], "requests": list(batch.req_ids),
                  "mode": mode, "draft_counts": counts.tolist(),
                  "seeds_uint64": params_np["seeds"].view(np.uint64).tolist(),
                  "stream_ids": [0] * requests,
                  "grammar_rows": 0 if mask_np is None else int(grammar_output.grammar_bitmask.shape[0]),
                  "raw_logits_dtype": str(logits.dtype)}
        result.exactsample_diagnostics = _diagnostics(state, logits, batch, positions, params_np, mask_np,
                                                      record, out)
    state["step"] += 1
    return result
