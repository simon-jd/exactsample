"""Exercise vLLM 0.31.0's actual 1 GiB RejectionSampler chunk boundary.

No model, scheduler, exactsample op, copied rejection implementation or changed native
chunk limit is used. The real Sampler constructor and sampling processors run.
`--describe` is CPU-only; the experiment is an explicit, separate GPU command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


VOCAB = 151936
DRAFT = 4
REQUESTS = 354
CHUNK_BYTES = 1 << 30
SOURCE_HASHES = {
    "spec_decode/rejection_sampler.py": "b682782f05f43b21f4ad1f995e37b6288d782b0dd4e3ba66db0919f5c5a9ae29",
    "spec_decode/rejection_sampler_utils.py": "d9f29ad1ce4fe40a078a78b21d73fab97bbbe018775acb480c0ad14dea0c5c35",
    "sample/sampler.py": "ddbcdab46010bf38a5fab9b9b6a2deed2ec5e883ebfe81e83f6d22f41e546771",
    "sample/states.py": "7dbccea6fb6e24830dafb447a43f93fd0539f33ef02f1b5bf7c2547234cbffa3",
    "input_batch.py": "96163c486368f2cf0e539fe68eaa06f4df3ac814da8f7a92af2b04242e01bc6d",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fixture():
    """Only small CPU arrays; the large logits are allocated by run_gpu()."""
    rows_per_request = DRAFT + 1
    request = np.repeat(np.arange(REQUESTS), rows_per_request)
    local = np.tile(np.arange(rows_per_request), REQUESTS).astype(np.int32)
    mapping = ((np.arange(REQUESTS) * 5) % REQUESTS + 17).astype(np.int32)
    base_positions = (64 + (np.arange(REQUESTS) % 13) * 7).astype(np.int64)
    positions = np.repeat(base_positions, rows_per_request) + local
    support = ((request[:, None] * 313 + local[:, None] * 103
                + np.arange(8)[None, :]) % VOCAB).astype(np.int64)
    weights = np.tile(np.array([2., 1., .5, 0., -.5, -1., -2., -3.], np.float32),
                      (len(request), 1))
    # Group 0 accepts all drafts. Group 1 rejects first. Group 2 accepts one
    # then rejects. Group 3 has probabilistic acceptance and residual sampling.
    deterministic = (request % 4 == 0) | ((request % 4 == 2) & (local == 0))
    weights[deterministic, 1:] = -np.inf
    native_drafts = np.zeros(len(request), dtype=np.int32)
    for req in range(REQUESTS):
        start = req * rows_per_request
        for step in range(DRAFT):
            token = int(support[start + step, 0])
            if req % 4 == 1 and step == 0 or req % 4 == 2 and step == 1:
                token = (token + 1000) % VOCAB  # outside this row's support
            native_drafts[start + step + 1] = token
    # Signed representations preserve all 64 seed bits and differ by identity.
    seeds = np.asarray([((i + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                        for i in range(REQUESTS)], dtype=np.uint64).view(np.int64)
    temperatures = np.asarray([.6, .8, 1.2], dtype=np.float32)[np.arange(REQUESTS) % 3]
    return {"mapping": mapping, "positions": positions, "local": local,
            "base_positions": base_positions, "support": support, "weights": weights,
            "native_drafts": native_drafts, "seeds": seeds, "temperatures": temperatures}


def description():
    rows = REQUESTS * (DRAFT + 1)
    return {
        "scope": "direct native RejectionSampler class; no model/engine/scheduler execution",
        "requests": REQUESTS, "rows": rows, "vocab": VOCAB, "draft_length": DRAFT,
        "raw_dtype": "bfloat16", "raw_bytes": rows * VOCAB * 2,
        "fp32_equivalent_bytes": rows * VOCAB * 4,
        "native_cap_bytes": CHUNK_BYTES, "native_max_rows": CHUNK_BYTES // (VOCAB * 4),
        "expected_whole_chunk_rows": [1765, 5],
        "separate_request_split": 173,
        "minimum_free_bytes": 4 << 30,
        "memory_note": "about .501 GiB raw + 1.002 GiB FP32 + .210 GiB native penalty state; 4 GiB free required with allocator/JIT margin",
    }


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def nvml_snapshot():
    try:
        import pynvml as nv
        nv.nvmlInit()
        handle = nv.nvmlDeviceGetHandleByIndex(0)
        value = {"memory_used_bytes": nv.nvmlDeviceGetMemoryInfo(handle).used,
                 "compute_processes": [{"pid": p.pid, "used_gpu_memory_bytes": p.usedGpuMemory}
                                       for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]}
        nv.nvmlShutdown()
        return value
    except Exception as exc:
        return {"unavailable": str(exc)}


def run_gpu(report):
    import torch
    require(importlib.metadata.version("vllm") == "0.31.0", "requires vLLM 0.31.0")
    if not torch.cuda.is_available():
        report.update(state="unavailable", reason="CUDA runtime unavailable")
        return 77
    torch.cuda.set_device(0)
    import vllm
    source_root = Path(vllm.__file__).parent / "v1/worker/gpu"
    actual = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
              for name in SOURCE_HASHES}
    require(actual == SOURCE_HASHES, "native source differs from locked 0.31.0 files")
    report["native_source_sha256"] = actual
    report["gpu_before"] = nvml_snapshot()
    free, total = torch.cuda.mem_get_info()
    report["memory_before"] = {"free_bytes": free, "total_bytes": total}
    if free < report["minimum_free_bytes"]:
        report.update(state="unavailable", reason="less than 4 GiB free; no large allocation attempted")
        return 77

    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as native

    require(native.MAX_CHUNK_BYTES == CHUNK_BYTES, "native cap must remain 1 GiB")
    require(native.get_max_chunk_logits(VOCAB) == 1766, "unexpected native row cap")
    data = fixture()
    report["fixture_hashes"] = {k: array_hash(v) for k, v in data.items()}
    report["rng"] = {"algorithm": "native vLLM, use_fp64_gumbel=False", "seeds_signed": data["seeds"].tolist(),
                     "persistent_slots": data["mapping"].tolist(),
                     "base_positions": data["base_positions"].tolist()}
    device = torch.device("cuda:0")
    rows = REQUESTS * (DRAFT + 1)
    slots = int(data["mapping"].max()) + 1
    req_states = RequestState(slots, 256, rows, DRAFT, VOCAB, device)
    req_states.prompt_len.np.fill(1)
    req_states.prefill_len.np.fill(1)
    req_states.prompt_len.copy_to_uva()
    req_states.prefill_len.copy_to_uva()
    # Sampler's actual built-in constructors do not consult model config here.
    # No models are created; reasoning is disabled and default processors remain.
    sampler = Sampler(SimpleNamespace(reasoning_config=None), slots, VOCAB, device,
                      req_states, num_speculative_tokens=DRAFT, use_fp64_gumbel=False)
    for req, slot in enumerate(data["mapping"]):
        sampler.add_request(int(slot), SamplingParams(temperature=float(data["temperatures"][req]),
                            top_k=-1, top_p=1., seed=int(data["seeds"][req]), max_tokens=32))
    sampler.apply_staged_writes()
    require(np.all(sampler.needs_logits_processing[data["mapping"]]), "FP32 processing must be active")

    calls = []
    phase = [""]

    class ObservedRejectionSampler(native.RejectionSampler):
        def _verify(self, *args, **kwargs):
            result = super()._verify(*args, **kwargs)
            processed, sampled, num_sampled = result
            raw, mapping_np = args[0], args[6]
            require(processed.dtype == torch.float32, "native FP32 materialization missing")
            require(processed.data_ptr() != raw.data_ptr(), "processed buffer aliases bf16 raw")
            calls.append({"phase": phase[0], "rows": raw.shape[0], "requests": len(mapping_np),
                          "persistent_slots": mapping_np.tolist(), "raw_dtype": str(raw.dtype),
                          "processed_dtype": str(processed.dtype),
                          "processed_bytes": processed.numel() * processed.element_size()})
            return result

    verifier = ObservedRejectionSampler(sampler, SimpleNamespace(num_speculative_tokens=DRAFT,
                enable_adaptive_verification=False, rejection_sample_method="standard"), device)
    report["native_verify_calls"] = calls
    logits = torch.full((rows, VOCAB), -torch.inf, dtype=torch.bfloat16, device=device)
    logits.scatter_(1, torch.as_tensor(data["support"], device=device),
                    torch.as_tensor(data["weights"], device=device, dtype=torch.bfloat16))

    def raw_hash():
        digest = hashlib.sha256()
        for lo in range(0, rows, 32):
            digest.update(logits[lo:lo + 32].view(torch.uint16).cpu().numpy().tobytes())
        return digest.hexdigest()

    report["logits_sha256"] = raw_hash()

    def batch(begin, end):
        lo, hi = begin * (DRAFT + 1), end * (DRAFT + 1)
        count, num_rows = end - begin, hi - lo
        mapping_np = data["mapping"][begin:end].copy()
        cu_np = np.arange(count + 1, dtype=np.int32) * (DRAFT + 1)
        scheduled = np.full(count, DRAFT + 1, dtype=np.int32)
        seq_np = (data["base_positions"][begin:end] + DRAFT + 1).astype(np.int32)
        mapping = torch.as_tensor(mapping_np, device=device)
        cu = torch.as_tensor(cu_np, device=device)
        seq = torch.as_tensor(seq_np, device=device)
        return InputBatch(
            req_ids=[f"native-chunk-{i}" for i in range(begin, end)], num_reqs=count,
            num_reqs_after_padding=count, idx_mapping=mapping, idx_mapping_np=mapping_np,
            expanded_idx_mapping=torch.repeat_interleave(mapping, DRAFT + 1),
            expanded_local_pos=torch.as_tensor(data["local"][lo:hi], device=device),
            num_scheduled_tokens=scheduled, num_tokens=num_rows, num_tokens_after_padding=num_rows,
            num_draft_tokens=count * DRAFT, num_draft_tokens_per_req=scheduled - 1,
            query_start_loc=cu, query_start_loc_np=cu_np, seq_lens=seq,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_np), dcp_local_seq_lens=None,
            num_computed_tokens_np=data["base_positions"][begin:end].astype(np.int32),
            prefill_len_np=np.ones(count, np.int32), num_computed_prefill_tokens_np=np.ones(count, np.int32),
            is_prefilling_np=np.zeros(count, bool), has_prefill=False, decode_graph_eligible=True,
            input_ids=torch.as_tensor(data["native_drafts"][lo:hi], device=device),
            positions=torch.as_tensor(data["positions"][lo:hi], device=device),
            is_padding=torch.zeros(num_rows, dtype=torch.bool, device=device),
            logits_indices=torch.arange(num_rows, dtype=torch.int64, device=device),
            cu_num_logits=cu, cu_num_logits_np=cu_np, has_structured_output_reqs=False,
            prompt_lens=None)

    def canonical(tokens, counts):
        # Native output slots beyond num_sampled may be uninitialized. Compare
        # exactly all committable tokens/counts, never those unspecified slots.
        counts = counts.cpu().numpy()
        require(np.all((counts >= 1) & (counts <= DRAFT + 1)), "invalid emitted counts")
        tokens = tokens.cpu().numpy()
        tokens[np.arange(DRAFT + 1)[None, :] >= counts[:, None]] = -1
        return tokens, counts

    def run_batch(name, begin, end):
        phase[0] = name
        result = verifier(logits[begin * (DRAFT + 1):end * (DRAFT + 1)], batch(begin, end))
        torch.cuda.synchronize()
        tokens, counts = canonical(result.sampled_token_ids, result.num_sampled)
        return tokens, counts, result.num_rejected.cpu().numpy()

    torch.cuda.reset_peak_memory_stats()
    whole = run_batch("whole_native_chunked", 0, REQUESTS)
    require([c["rows"] for c in calls if c["phase"] == phase[0]] == [1765, 5],
            "native whole call did not execute expected two request-aligned chunks")
    left = run_batch("separate_left", 0, 173)
    right = run_batch("separate_right", 173, REQUESTS)
    separate = tuple(np.concatenate([a, b]) for a, b in zip(left, right))
    for name, a, b in zip(["tokens", "num_sampled", "num_rejected"], whole, separate):
        np.testing.assert_array_equal(a, b, err_msg=f"native chunk vs independent batches: {name}")
    require([c["rows"] for c in calls if c["phase"] == "separate_left"] == [865], "left unexpectedly chunked")
    require([c["rows"] for c in calls if c["phase"] == "separate_right"] == [905], "right unexpectedly chunked")

    phase[0] = "direct_verify_all_rows"
    full_batch = batch(0, REQUESTS)
    draft_sampled, pos = native.gather_draft_sampled(full_batch.input_ids, full_batch.positions,
        full_batch.logits_indices, full_batch.expanded_idx_mapping,
        full_batch.expanded_local_pos, sampler.req_states.prefill_len.gpu)
    processed, sampled, num_sampled = verifier._verify(logits, None, draft_sampled, pos,
        full_batch.cu_num_logits, full_batch.idx_mapping, full_batch.idx_mapping_np,
        full_batch.expanded_idx_mapping, full_batch.expanded_local_pos,
        full_batch.seq_lens_cpu_upper_bound.numpy())
    torch.cuda.synchronize()
    direct = canonical(sampled, num_sampled)
    del processed
    for name, a, b in zip(["tokens", "num_sampled"], whole[:2], direct):
        np.testing.assert_array_equal(a, b, err_msg=f"native chunk vs unchunked _verify: {name}")
    np.testing.assert_array_equal(whole[1][np.arange(REQUESTS) % 4 == 0], DRAFT + 1)
    np.testing.assert_array_equal(whole[1][np.arange(REQUESTS) % 4 == 1], 1)
    np.testing.assert_array_equal(whole[1][np.arange(REQUESTS) % 4 == 2], 2)
    require(raw_hash() == report["logits_sha256"], "native calls modified raw logits")
    report.update(state="passed", native_verify_calls=calls, comparisons={
        "chunked_vs_independent_batches": "all committable tokens and sampled/rejected counts equal",
        "chunked_vs_direct_verify": "all committable tokens and sampled counts equal"},
        output_tokens=whole[0].tolist(), num_sampled=whole[1].tolist(), num_rejected=whole[2].tolist(),
        output_sha256=array_hash(whole[0]), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu_after=nvml_snapshot(),
        environment={"vllm": importlib.metadata.version("vllm"), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
        limitations="Synthetic sparse finite-support bf16 rows, one-hot drafts, temperature-only native processors, no model/engine/preemption, no cross-exactsample RNG equality claim")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="CPU-only layout/memory description")
    parser.add_argument("--output", type=Path, help="new JSON path (never overwritten)")
    args = parser.parse_args()
    report = description()
    if args.describe:
        print(json.dumps(report, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless --describe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Claim the path before CUDA imports/work; failure evidence is retained.
    with args.output.open("x") as stream:
        report.update(state="running", started_at=datetime.now(timezone.utc).isoformat(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        stream.write(json.dumps(report, indent=2) + "\n")
    try:
        code = run_gpu(report)
    except Exception as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        code = 1
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"state": report["state"], "output": str(args.output), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
