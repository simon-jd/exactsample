"""Tables from bench/sampler.py output: GPU time per call, per implementation.

    python bench/summarize.py results/sampler/*.jsonl
"""
import json
import sys
from collections import defaultdict

ORDER = ["exactsample", "vllm_mrv2", "flashinfer"]
NAMES = {"exactsample": "exactsample", "vllm_mrv2": "vLLM 0.31",
         "flashinfer": "FlashInfer 0.7"}
TITLES = {"qwen": "T=0.6, top-k 20, top-p 0.95", "temperature": "T=1.0", "top_p": "T=0.7, top-p 0.9"}


def main():
    cells = defaultdict(dict)
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            cells[(r["params"], r["drafts"], r["requests"])][r["impl"]] = r
    impls = [i for i in ORDER if any(i in c for c in cells.values())]
    for params in TITLES:
        keys = sorted(k for k in cells if k[0] == params)
        if not keys:
            continue
        print(f"\n### {TITLES[params]}: GPU time per call, us (call time with launch overhead)\n")
        print("| requests x rows each | " + " | ".join(NAMES[i] for i in impls) + " |")
        print("|---|" + "---:|" * len(impls))
        for key in keys:
            row = cells[key]
            label = f"{key[2]} x {key[1] + 1}"
            values = []
            for impl in impls:
                r = row.get(impl)
                values.append("–" if r is None else f"{r['gpu_us']:.0f} ({r['call_us']:.0f})")
            print(f"| {label} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
