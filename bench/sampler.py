"""Sampling and speculative-verification benchmark on captured logits.

Workloads are built from bench/capture_logits.py output: ordinary sampling
(one row per request) and 4-draft verification (five consecutive decode
steps of one prompt, the sampled tokens as drafts). Every implementation
starts from the same bf16 logits and request parameters.

Each implementation runs in its own process, because two exactsample builds register
the same torch op. Reported per call:
  gpu_us   summed kernel time (CUPTI, via torch.profiler);
  call_us  CUDA-event time over 100 back-to-back calls, logits in L2 as they
           are right after the LM head; includes host launch overhead;
  cold_us  median event time of one call after the L2 has been flushed.

    python bench/sampler.py --impl exactsample --run 1 --output OUT/exactsample-1.jsonl
    python bench/sampler.py --impl flashinfer:probs --run 1 --output OUT/flashinfer-probs-1.jsonl

The logits come from --capture (bench/capture_logits.py's output).
bench/run_sampler.sh repeats every implementation, interleaved, which is how
results/sampler/runs was made, and bench/summarize.py reports the median and
interquartile range over runs.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {
    "qwen": dict(temperature=0.6, top_k=20, top_p=0.95),  # Qwen3's recommended sampling
    "temperature": dict(temperature=1.0, top_k=0, top_p=1.0),
    "top_p": dict(temperature=0.7, top_k=0, top_p=0.9),
}
SHAPES = [(1, 0), (8, 0), (32, 0), (128, 0), (1, 4), (8, 4), (32, 4), (64, 4)]  # (requests, drafts)


def workload(capture, requests, drafts, params, device):
    """Rows, offsets, drafts and per-request parameters for one cell."""
    import torch
    from exactsample.cuda import CudaRequestParams, CudaRngKey

    steps = int(capture["step"].max()) + 1
    sequences = int(capture["sequence"].max()) + 1
    rows = []
    ids = np.full((requests, max(drafts, 1)), -1, dtype=np.int64)
    for i in range(requests):
        sequence = i % sequences
        start = (i // sequences) * (drafts + 1) % (steps - drafts)
        first = sequence * steps + start
        rows += list(range(first, first + drafts + 1))
        ids[i, :drafts] = capture["sampled"][first:first + drafts]
    rows = np.array(rows)
    logits = torch.as_tensor(capture["logits_bf16"][rows]).view(torch.bfloat16).to(device)
    offsets = torch.arange(requests + 1, device=device, dtype=torch.int64) * (drafts + 1)

    def full(value, dtype):
        return torch.full((requests,), value, device=device, dtype=dtype)

    gpu_params = CudaRequestParams(full(params["temperature"], torch.float64), full(params["top_k"], torch.int64),
                                   full(params["top_p"], torch.float64), full(0.0, torch.float64),
                                   full(0, torch.int64))
    rng = CudaRngKey(torch.arange(requests, device=device, dtype=torch.int64) + 1000,
                     torch.arange(requests, device=device, dtype=torch.int64),
                     torch.as_tensor(capture["position"][rows], device=device, dtype=torch.int64))
    return logits, offsets, torch.as_tensor(ids, device=device), gpu_params, rng


def kernel_name(name):
    """Kernel name without return type, namespaces' template arguments or parameters."""
    name = re.sub(r"^void ", "", name)
    depth, out = 0, []
    for ch in name:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).split(" ")[-1]


def measure(call, iterations=100, cold_iterations=30):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(10):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            call()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    gpu_us = sum(e.device_time_total for e in kernels) / 20
    by_name = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        entry = by_name[kernel_name(e.name)]
        entry[0] += 1
        entry[1] += e.device_time_total
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    stop.record()
    stop.synchronize()
    call_us = start.elapsed_time(stop) * 1000 / iterations
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    cold = []
    for _ in range(cold_iterations):
        flush.zero_()
        start.record()
        call()
        stop.record()
        stop.synchronize()
        cold.append(start.elapsed_time(stop) * 1000)
    return {"gpu_us": gpu_us, "kernels_per_call": len(kernels) / 20, "call_us": call_us,
            "cold_us": float(np.median(cold)),
            "kernels": {n: {"launches": c / 20, "us": t / 20} for n, (c, t) in sorted(by_name.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["exactsample", "vllm_mrv2", "flashinfer:logits", "flashinfer:probs"],
                        required=True)
    parser.add_argument("--label")
    parser.add_argument("--run", type=int, default=0, help="repetition index, recorded with each cell")
    parser.add_argument("--exactsample-root", type=Path, default=ROOT, help="exactsample checkout whose python/ package is used")
    parser.add_argument("--capture", type=Path, default=ROOT / "results/raw/logits/qwen3-8b.npz")
    parser.add_argument("--params", nargs="*", default=list(PARAMS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.exactsample_root / "python"), str(ROOT / "bench")]
    import torch

    capture = dict(np.load(args.capture))
    label = args.label or args.impl
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for name in args.params:
            for requests, drafts in SHAPES:
                logits, offsets, ids, params, rng = workload(capture, requests, drafts, PARAMS[name], "cuda")
                record = {"impl": label, "run": args.run, "params": name, "requests": requests,
                          "drafts": drafts, "rows": int(logits.shape[0])}
                if args.impl == "exactsample":
                    from exactsample.cuda import plan, verify
                    workspace = plan(logits.shape[0], requests, ids.shape[1], logits.shape[1])
                    output = workspace.allocate_output()
                    # What a caller with host-side parameters knows, as vLLM's sampler does.
                    hints = {"top_k_top_p": PARAMS[name]["top_k"] > 0 or PARAMS[name]["top_p"] < 1}
                    record["top_k_top_p"] = hints["top_k_top_p"]

                    def call():
                        return verify(logits, offsets, ids, params=params, rng=rng, plan=workspace, out=output,
                                      **hints)

                    record.update(measure(call))
                    record["accepted_mean"] = output.accepted_len[:requests].float().mean().item()
                    record["paths"] = sorted(set(output.path_used[:requests].tolist()))
                    if hasattr(workspace, "fallbacks"):
                        call()
                        record["fp64_fallbacks"] = workspace.fallbacks()
                else:
                    from native import prepare
                    prepared = prepare(args.impl, logits, offsets, ids[:, :drafts], params, rng)
                    if prepared is None:
                        continue
                    call, metadata = prepared
                    record.update(measure(call))
                    result = call()
                    record["accepted_mean"] = result.accepted_len.float().mean().item()
                    record["version"] = metadata["version"]
                    record["scope"] = metadata["scope"]
                print(json.dumps(record), flush=True)
                out.write(json.dumps(record) + "\n")
                del logits
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
