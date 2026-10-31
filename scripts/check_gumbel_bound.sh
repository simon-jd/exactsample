#!/usr/bin/env bash
# Checks the FP32 Gumbel noise against its documented error bound for all
# 2^32 Philox words (tests/exhaustive/gumbel_bound.cu), compiled with the
# library's flags, and writes the report. About three minutes on an RTX 4090.
#
#   bash scripts/check_gumbel_bound.sh [results/gumbel-bound.json]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/gumbel-bound.json}"
BUILD="${EXACTSAMPLE_BUILD:-build/cuda}"
mkdir -p "$BUILD"
nvcc -O3 -std=c++17 --fmad=false -arch="sm_${EXACTSAMPLE_CUDA_ARCHITECTURES:-89}" \
  -Iinclude -Icsrc/verify tests/exhaustive/gumbel_bound.cu -o "$BUILD/gumbel_bound"
"$BUILD/gumbel_bound" | tee "$OUT"
