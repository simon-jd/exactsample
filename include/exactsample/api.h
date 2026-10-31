#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace exactsample {
enum Status : int32_t { OK = 0, NO_VALID_TOKEN = 1, INVALID_INPUT = 2, DRAFT_ZERO_PROB = 3 };
enum Path : int32_t { REFERENCE = 0, FUSED = 1 };
enum DType : int32_t { F32 = 0, F64 = 1, F16 = 2, BF16 = 3 };
struct Matrix {
  const void* data;
  int64_t s0, s1;
  DType dtype;
};
template <class T> struct Vector {
  const T* data;
  int64_t stride;
};
struct RowResult { double p, q; int32_t sample, residual, status, target_status; };
struct Arguments {
  Matrix target, draft;
  Vector<int64_t> cu_rows, top_k, draft_kind, seed, stream, position;
  Vector<double> temperature, top_p, min_p;
  const int64_t* draft_ids;
  int64_t draft_s0, draft_s1;
  const int32_t* mask;
  int64_t mask_s0, mask_s1;
  int64_t* tokens;
  int32_t *accepted, *status, *path;
  int64_t token_s0, token_s1, accepted_s, status_s, path_s;
  int64_t rows, requests, vocab, max_draft;
  int mode, dispatch;  // mode: 0 ratio, 1 coupled; dispatch: 0 auto, 1 reference, 2 fused.
  int list_max;        // fused path: largest top-k processed as an FP64 list (at most 256)
  uint8_t* workspace;
};
// Layout has no hidden allocator: an 8-byte header and one RowResult per row,
// then per-row scratch: FUSED_ROW_BYTES for the fused path, or (reference
// path) target probabilities, q and sort indices. A call uses one of the two.
constexpr size_t FUSED_ROW_BYTES = 12288;
inline size_t workspace_size(int64_t rows, int64_t vocab, bool reference) {
  size_t scratch = reference ? size_t(vocab) * (2 * sizeof(double) + sizeof(int32_t)) : 0;
  return 8 + size_t(rows) * (sizeof(RowResult) + (scratch > FUSED_ROW_BYTES ? scratch : FUSED_ROW_BYTES));
}
cudaError_t launch(const Arguments&, cudaStream_t);
}  // namespace exactsample
