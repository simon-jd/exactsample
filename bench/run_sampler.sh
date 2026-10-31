#!/usr/bin/env bash
# Runs bench/sampler.py for every implementation, one process each, holding
# the machine-wide GPU lock. Assumes the logits capture exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${EXACTSAMPLE_PYTHON:-python3}"
OUT="${EXACTSAMPLE_BENCH_OUT:-results/sampler}"
LOCK="${EXACTSAMPLE_GPU_LOCK:-/var/tmp/rtx4090.lock}"
mkdir -p "$OUT"
# Triton's bundled ptxas is not executable in some installs; use the toolkit's.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$(command -v ptxas)}"
run() { flock "$LOCK" "$PYTHON" bench/sampler.py "$@"; }
run --impl exactsample --output "$OUT/exactsample.jsonl"
run --impl vllm_mrv2 --output "$OUT/vllm_mrv2.jsonl"
run --impl flashinfer --output "$OUT/flashinfer.jsonl"
"$PYTHON" bench/summarize.py "$OUT"/*.jsonl
