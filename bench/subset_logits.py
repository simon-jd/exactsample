"""A subset of a bench/capture_logits.py capture small enough for git.

Keeps the first --sequences prompts and their first --steps decode steps, the
row grid bench/sampler.py's workloads expect, and writes them with metadata
that names the source capture. The output is byte-for-byte reproducible.

    python bench/subset_logits.py results/raw/logits/qwen3-8b.npz results/logits/qwen3-8b-8x8.npz
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    source = dict(np.load(args.capture))
    keep = (source["sequence"] < args.sequences) & (source["step"] < args.steps)
    subset = {name: value[keep] for name, value in source.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **subset)
    meta = json.loads(args.capture.with_suffix(".json").read_text())
    match = re.search(r"models--([^/]+)--([^/]+)", meta["model"])
    meta.update(model=f"{match[1]}/{match[2]}" if match else meta["model"], prompts=meta["prompts"][:args.sequences],
                steps=args.steps, rows=int(keep.sum()), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                subset_of={"rows": int(keep.size), "sha256": meta["sha256"], "sequences": args.sequences,
                           "steps": args.steps})
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "prompts"}))


if __name__ == "__main__":
    main()
