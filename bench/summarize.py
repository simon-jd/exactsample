"""Tables from bench/run_sampler.sh output: median GPU time per call over runs.

    python bench/summarize.py results/raw/sampler/<name> [--json results/sampler/summary.json]

For each cell and implementation, reports the median and interquartile range
over runs of gpu_us (summed kernel time per call), call_us (event time per
call over back-to-back calls, with launch overhead) and cold_us (one call
after flushing L2). FlashInfer's column is its faster public entry point for
the cell (by median gpu_us), named in the JSON.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["exactsample", "vllm_mrv2", "flashinfer"]
NAMES = {"exactsample": "exactsample", "vllm_mrv2": "vLLM 0.31", "flashinfer": "FlashInfer 0.7"}
TITLES = {"qwen": "T=0.6, top-k 20, top-p 0.95", "temperature": "T=1.0", "top_p": "T=0.7, top-p 0.9"}
METRICS = ("gpu_us", "call_us", "cold_us")


def stats(values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3), "min": float(min(values)),
            "max": float(max(values)), "runs": len(values)}


def summarize(directory):
    runs = defaultdict(list)
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            runs[(r["params"], r["drafts"], r["requests"], r["impl"])].append(r)
    cells = defaultdict(dict)
    for (params, drafts, requests, impl), records in runs.items():
        entry = {m: stats([r[m] for r in records]) for m in METRICS}
        entry["kernels_per_call"] = records[0]["kernels_per_call"]
        entry["api"] = impl
        family = impl.partition(":")[0]
        best = cells[(params, drafts, requests)].get(family)
        if best is None or entry["gpu_us"]["median"] < best["gpu_us"]["median"]:
            cells[(params, drafts, requests)][family] = entry
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    cells = summarize(args.directory)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"params": k[0], "drafts": k[1], "requests": k[2], "impl": impl, **entry}
                for k, by_impl in sorted(cells.items()) for impl, entry in by_impl.items()]
        env = args.directory / "env.json"
        json.dump({"env": json.loads(env.read_text()) if env.exists() else None, "cells": rows},
                  args.json.open("w"), indent=1)
    impls = [i for i in ORDER if any(i in c for c in cells.values())]
    for params in TITLES:
        keys = sorted(k for k in cells if k[0] == params)
        if not keys:
            continue
        print(f"\n### {TITLES[params]}: median GPU time per call, us (IQR); cold L2 in brackets\n")
        print("| requests x rows each | " + " | ".join(NAMES[i] for i in impls) + " |")
        print("|---|" + "---:|" * len(impls))
        for key in keys:
            values = []
            for impl in impls:
                e = cells[key].get(impl)
                if e is None:
                    values.append("–")
                    continue
                g, c = e["gpu_us"], e["cold_us"]
                values.append(f"{g['median']:.0f} ({g['q1']:.0f}–{g['q3']:.0f}) [{c['median']:.0f}]")
            print(f"| {key[2]} x {key[1] + 1} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
