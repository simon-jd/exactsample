#!/usr/bin/env bash
# Compiles, without a GPU or PyTorch, everything CUDA in the repository except
# the PyTorch operator (csrc/bindings.cpp): the kernels, through
# CMakeLists.txt with scripts/build_cuda.sh's settings (Release, sm_86 and sm_89, and
# the library's --fmad=false -lineinfo) plus the CUDA flags PyTorch 2.13's
# CMake adds to that build (TORCH_CUDA_FLAGS), so every nvcc flag is the
# local build's; and the exhaustive Gumbel check with
# scripts/check_gumbel_bound.sh's flags. CI runs it.
#
#   bash scripts/compile_cuda.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD="${EXACTSAMPLE_BUILD:-build/cuda-compile}"
TORCH_CUDA_FLAGS="-DONNX_NAMESPACE=onnx_c2 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DEXACTSAMPLE_STANDALONE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${EXACTSAMPLE_CUDA_ARCHITECTURES:-86;89}" -DCMAKE_CUDA_FLAGS="$TORCH_CUDA_FLAGS"
cmake --build "$BUILD" --parallel "${EXACTSAMPLE_BUILD_JOBS:-2}"
EXACTSAMPLE_BUILD="$BUILD" bash scripts/check_gumbel_bound.sh --compile-only
