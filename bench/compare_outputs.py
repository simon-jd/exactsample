"""Bit-for-bit comparison of two bench/agreement.py --outputs files, for
example the same workloads run on two GPUs.

    python bench/compare_outputs.py A.npz B.npz [--json OUT.json]

For each workload, counts the requests whose tokens, accepted length, status
and path are all identical, and says whether every array is byte-identical.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ("tokens", "accepted_len", "status", "path_used")


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    if sorted(a.files) != sorted(b.files):
        raise SystemExit(f"different workloads: {sorted(set(a.files) ^ set(b.files))}")
    workloads = list(dict.fromkeys(key.rsplit("-", 1)[0] for key in a.files))
    rows = []
    for workload in workloads:
        same, byte_identical = None, {}
        for field in FIELDS:
            x, y = a[f"{workload}-{field}"], b[f"{workload}-{field}"]
            byte_identical[field] = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            equal = (x == y).reshape(len(x), -1).all(axis=1) if x.shape == y.shape else np.zeros(len(x), bool)
            same = equal if same is None else same & equal
        rows.append({"workload": workload, "requests": len(same), "identical_requests": int(same.sum()),
                     "byte_identical": byte_identical})
    return {"a": Path(a_path).name, "b": Path(b_path).name, "workloads": rows,
            "requests": sum(r["requests"] for r in rows),
            "identical_requests": sum(r["identical_requests"] for r in rows),
            "all_byte_identical": all(all(r["byte_identical"].values()) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = compare(args.a, args.b)
    text = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
