"""Token-for-token agreement of the CUDA fused path with the FP64 oracle on
captured logits, and the number of FP64 fallbacks the fused path needed.

    python bench/agreement.py --output results/agreement.json
    python bench/agreement.py --capture results/logits/qwen3-8b-8x8.npz --output results/agreement-8x8.json

The first uses the full 512-row capture (results/raw/logits, not in git: 155
MB); the second the 64-row subset in the repository (bench/subset_logits.py),
which runs the same six workloads from fewer distinct rows.

--outputs OUT.npz also saves the fused path's raw outputs (tokens, accepted
lengths, statuses and the path each request took) per workload, so that two
GPUs can be compared bit for bit with bench/compare_outputs.py.
"""
import argparse
import re
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "bench"), str(ROOT)]
from sampler import PARAMS, workload  # noqa: E402

CELLS = [("qwen", 64, 0), ("qwen", 16, 4), ("top_p", 64, 0), ("top_p", 16, 4), ("temperature", 24, 0),
         ("temperature", 8, 4)]


def model_id(model):
    """'Org/Name' for a Hugging Face cache path, so results carry no local paths."""
    match = re.search(r"models--([^/]+)--([^/]+)", model)
    return f"{match[1]}/{match[2]}" if match else model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=ROOT / "results/raw/logits/qwen3-8b.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, help="also save the raw outputs of every workload to this .npz")
    args = parser.parse_args()
    import torch
    from exactsample import RequestParams, RngKey, verify_reference
    from exactsample.cuda import plan, verify

    capture = dict(np.load(args.capture))
    report, raw = [], {}
    for name, requests, drafts in CELLS:
        logits, offsets, ids, params, rng = workload(capture, requests, drafts, PARAMS[name], "cuda")
        workspace = plan(logits.shape[0], requests, ids.shape[1], logits.shape[1])
        out = verify(logits, offsets, ids, params=params, rng=rng, plan=workspace, out=workspace.allocate_output())
        fallbacks = workspace.fallbacks()
        for field in ("tokens", "accepted_len", "status", "path_used"):
            raw[f"{name}-{requests}x{drafts}-{field}"] = getattr(out, field)[:requests].cpu().numpy()
        expected = verify_reference(
            logits.double().cpu().numpy(), offsets.cpu().numpy(), ids[:, :max(drafts, 1)].cpu().numpy(),
            params=RequestParams(**PARAMS[name]),
            rng=RngKey(rng.seed.cpu().numpy().astype(np.uint64), rng.stream_id.cpu().numpy(), rng.positions.cpu().numpy()))
        same = (out.tokens[:requests].cpu().numpy() == expected.tokens).all(axis=1) & \
               (out.accepted_len[:requests].cpu().numpy() == expected.accepted_len) & \
               (out.status[:requests].cpu().numpy() == expected.status)
        cell = {"params": name, "requests": requests, "drafts": drafts, "rows": int(logits.shape[0]),
                "identical_requests": int(same.sum()), "fp64_fallbacks": fallbacks,
                "accepted_mean": float(expected.accepted_len.mean())}
        print(json.dumps(cell), flush=True)
        report.append(cell)
    meta = json.loads(args.capture.with_suffix(".json").read_text())
    meta["model"] = model_id(meta["model"])
    args.output.write_text(json.dumps({"capture": {k: meta[k] for k in ("model", "revision", "rows", "sha256")},
                                       "cells": report}, indent=2) + "\n")
    if args.outputs:
        np.savez(args.outputs, **raw)


if __name__ == "__main__":
    main()
