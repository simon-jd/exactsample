#!/usr/bin/env bash
# Checks the FP32 Gumbel noise against its documented error bound for all
# 2^32 Philox words (tests/exhaustive/gumbel_bound.cu), compiled with the
# library's flags, and writes the report. About four seconds on an RTX 4090, compiling included.
#
#   bash scripts/check_gumbel_bound.sh [results/gumbel-bound.json]
#   bash scripts/check_gumbel_bound.sh --compile-only    (no GPU needed)
set -euo pipefail
cd "$(dirname "$0")/.."
COMPILE_ONLY=0
if [ "${1:-}" = --compile-only ]; then COMPILE_ONLY=1; shift; fi
OUT="${1:-results/gumbel-bound.json}"
BUILD="${EXACTSAMPLE_BUILD:-build/cuda}"
mkdir -p "$BUILD"
ARCHS="${EXACTSAMPLE_CUDA_ARCHITECTURES:-86;89}"  # CMake's list form, as in build_cuda.sh
GENCODE=()
for arch in ${ARCHS//;/ }; do
  GENCODE+=(-gencode "arch=compute_${arch},code=[sm_${arch},compute_${arch}]")
done
nvcc -O3 -std=c++17 --fmad=false "${GENCODE[@]}" \
  -Iinclude -Icsrc/verify tests/exhaustive/gumbel_bound.cu -o "$BUILD/gumbel_bound"
[ "$COMPILE_ONLY" = 1 ] && exit 0
"$BUILD/gumbel_bound" | tee "$OUT"
