#pragma once
// Device helpers shared by the reference and fused paths.
#include "exactsample/api.h"
#include "exactsample/philox.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace exactsample {
template <class T> __device__ inline T at(Vector<T> v, int64_t i) { return v.data[i * v.stride]; }
__device__ inline double get(Matrix m, int64_t r, int64_t c) {
  int64_t i = r * m.s0 + c * m.s1;
  switch (m.dtype) {
    case F64: return static_cast<const double*>(m.data)[i];
    case F16: return __half2float(static_cast<const __half*>(m.data)[i]);
    case BF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(m.data)[i]);
    default: return static_cast<const float*>(m.data)[i];
  }
}
__device__ inline bool allowed(const Arguments& a, int64_t row, int64_t token) {
  return !a.mask || ((uint32_t(a.mask[row * a.mask_s0 + (token / 32) * a.mask_s1]) >> (token % 32)) & 1u);
}
__device__ inline bool better(double av, int ai, double bv, int bi) {
  return av > bv || (av == bv && ai < bi);
}
__device__ inline double uniform_at(const Arguments& a, int req, int64_t row, int sub, int token) {
  return uniform(uint64_t(at(a.seed, req)), uint32_t(at(a.stream, req)),
                 uint32_t(at(a.position, row)), sub, token);
}
__device__ inline double gumbel(const Arguments& a, int req, int64_t row, int sub, int token) {
  return -log(-log(uniform_at(a, req, row, sub, token)));
}

// Fused-path workspace, after the RowResult array: RowState[rows], per-slice
// parts, top-k candidates, then two counters per row (zero between calls).
constexpr int MAX_SLICES = 32;
constexpr int CANDIDATES = 2048;  // top-k candidates gathered per row by row_gather
// Distance-below-maximum histogram per row: 0.5-logit bins over 64 logits, then one for the rest.
constexpr int HIST_BINS = 128;
struct RowState {
  double maximum, qmax;
  int greedy, valid, status, qstatus;
  int part;    // which kernel finishes the row (fused.cuh, Part)
  int gather;  // candidates row_hist and row_gather prepare (fused.cuh, Gather)
  double runner_up;  // a valid logit other than the greedy token's, near the top, and its ID (INT_MAX if none)
  int runner_up_id, pad;  // pad: the alignment padding, named so that row_stats sets every byte
};
static_assert(offsetof(RowState, runner_up) == 40 && offsetof(RowState, pad) == 52 && sizeof(RowState) == 56,
              "RowState layout: pad is padding");
__host__ __device__ inline RowState* row_states(const Arguments& a) {
  return reinterpret_cast<RowState*>(a.workspace + 8 + a.rows * sizeof(RowResult));
}
__host__ __device__ inline unsigned* arrival_counters(const Arguments& a) {
  return reinterpret_cast<unsigned*>(a.workspace + 8 + a.rows * (sizeof(RowResult) + FUSED_ROW_BYTES)) - 2 * a.rows;
}

// Enqueues the fused row kernels for every request whose path is FUSED.
cudaError_t launch_fused(const Arguments& a, cudaStream_t stream);
}  // namespace exactsample
