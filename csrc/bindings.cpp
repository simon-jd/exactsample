#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include "exactsample/api.h"
#include <limits>
#include <numeric>
#include <vector>

namespace {
using at::Tensor;
using OptionalTensor = c10::optional<Tensor>;
exactsample::DType dtype(const Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return exactsample::F32;
    case at::kDouble: return exactsample::F64;
    case at::kHalf: return exactsample::F16;
    case at::kBFloat16: return exactsample::BF16;
    default: TORCH_CHECK(false, "exactsample logits require float16/bfloat16/float32/float64");
  }
}
exactsample::Matrix matrix(const Tensor& t) { return {t.const_data_ptr(), t.stride(0), t.stride(1), dtype(t)}; }
template <class T> exactsample::Vector<T> vector(const Tensor& t) { return {t.const_data_ptr<T>(), t.stride(0)}; }
bool disjoint(const Tensor& a, const Tensor& b) {
  if (!a.numel() || !b.numel()) return true;
  auto end = [](const Tensor& t) {
    uintptr_t span = t.element_size();
    for (int i = 0; i < t.dim(); ++i) span += (t.size(i) - 1) * t.stride(i) * t.element_size();
    return reinterpret_cast<uintptr_t>(t.const_data_ptr()) + span;
  };
  // PyTorch returns TooHard for ordinary non-dense inputs, even when their
  // allocations are distinct. Bounding address intervals prove disjointness
  // without treating an unknown overlap result as safe.
  return end(a) <= reinterpret_cast<uintptr_t>(b.const_data_ptr()) ||
         end(b) <= reinterpret_cast<uintptr_t>(a.const_data_ptr());
}
bool no_internal_overlap(const Tensor& t) {
  if (t.numel() <= 1) return true;
  for (int d = 0; d < t.dim(); ++d) if (t.size(d) > 1 && t.stride(d) == 0) return false;
  if (t.dim() == 1 || t.size(0) <= 1 || t.size(1) <= 1) return true;
  int64_t divisor = std::gcd(t.stride(0), t.stride(1));
  return t.size(0) <= t.stride(1) / divisor || t.size(1) <= t.stride(0) / divisor;
}
void verify_out(const Tensor& logits, const Tensor& cu_rows, const Tensor& draft_ids,
                const OptionalTensor& draft_logits, const Tensor& temperature, const Tensor& top_k,
                const Tensor& top_p, const Tensor& min_p, const Tensor& draft_kind,
                const Tensor& seeds, const Tensor& stream_ids, const Tensor& positions,
                const OptionalTensor& mask, Tensor workspace, Tensor tokens, Tensor accepted_len,
                Tensor status, Tensor path_used, int64_t mode, int64_t dispatch, int64_t max_draft,
                int64_t list_max) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2, "target_logits must be a 2D CUDA tensor");
  const auto device = logits.device();
  auto check = [&](const Tensor& t, at::ScalarType kind, int dim, const char* name) {
    TORCH_CHECK(t.device() == device && t.scalar_type() == kind && t.dim() == dim,
                name, " has wrong device, dtype or rank");
    for (int d = 0; d < dim; ++d) TORCH_CHECK(t.stride(d) >= 0, name, " has a negative stride");
  };
  check(cu_rows, at::kLong, 1, "cu_rows");
  TORCH_CHECK(cu_rows.numel() >= 1, "cu_rows must contain an endpoint");
  int64_t requests = cu_rows.numel() - 1, rows = logits.size(0), vocab = logits.size(1);
  TORCH_CHECK(requests != 0 || rows == 0, "an empty request batch cannot contain target rows");
  TORCH_CHECK(vocab > 0 && vocab <= std::numeric_limits<int32_t>::max() &&
              rows <= std::numeric_limits<int32_t>::max() && requests <= std::numeric_limits<int32_t>::max(),
              "exactsample dimensions exceed int32 kernel indexing");
  TORCH_CHECK(rows <= 65535, "exactsample supports at most 65535 target rows per call");
  TORCH_CHECK(max_draft >= 0 && max_draft < std::numeric_limits<int32_t>::max(), "invalid max_draft");
  TORCH_CHECK(mode >= 0 && mode <= 1 && dispatch >= 0 && dispatch <= 2, "invalid mode/dispatch");
  TORCH_CHECK(list_max >= 0 && list_max <= 256, "list_max must be in [0, 256]");
  check(draft_ids, at::kLong, 2, "draft_ids");
  TORCH_CHECK(draft_ids.size(0) == requests && draft_ids.size(1) >= max_draft, "draft_ids capacity mismatch");
  std::vector<Tensor> inputs{logits, cu_rows, draft_ids, temperature, top_k, top_p, min_p, draft_kind, seeds, stream_ids, positions};
  for (const auto& t : {temperature, top_p, min_p}) {
    check(t, at::kDouble, 1, "floating request parameter");
    TORCH_CHECK(t.numel() == requests, "request parameter length mismatch");
  }
  for (const auto& t : {top_k, draft_kind, seeds, stream_ids}) {
    check(t, at::kLong, 1, "integer request parameter");
    TORCH_CHECK(t.numel() == requests, "request parameter length mismatch");
  }
  check(positions, at::kLong, 1, "positions");
  TORCH_CHECK(positions.numel() == rows, "positions must be per target row");
  exactsample::Matrix draft{nullptr, 0, 0, exactsample::F32};
  if (draft_logits.has_value()) {
    const auto& t = *draft_logits;
    TORCH_CHECK(t.device() == device && t.dim() == 2 && t.sizes() == logits.sizes(), "draft_logits shape/device mismatch");
    TORCH_CHECK(t.scalar_type() == logits.scalar_type(), "draft_logits must have the target_logits dtype");
    draft = matrix(t); inputs.push_back(t);
  }
  const int32_t* mask_ptr = nullptr;
  int64_t mask_s0 = 0, mask_s1 = 0;
  if (mask.has_value()) {
    const auto& t = *mask;
    check(t, at::kInt, 2, "mask");
    TORCH_CHECK(t.size(0) == rows && t.size(1) == (vocab + 31) / 32, "mask shape mismatch");
    mask_ptr = t.const_data_ptr<int32_t>(); mask_s0 = t.stride(0); mask_s1 = t.stride(1);
    inputs.push_back(t);
  }
  check(workspace, at::kByte, 1, "workspace");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(uint64_t(rows) <= uint64_t(std::numeric_limits<int64_t>::max() - 8) /
              (sizeof(exactsample::RowResult) + uint64_t(vocab) * 20), "workspace size overflow");
  TORCH_CHECK(uint64_t(workspace.numel()) >= exactsample::workspace_size(rows, vocab, dispatch == 1),
              dispatch == 1 ? "workspace capacity exceeded (the reference path needs a plan made with reference=True)"
                            : "workspace capacity exceeded");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % alignof(double) == 0, "workspace alignment must be 8 bytes");
  check(tokens, at::kLong, 2, "tokens");
  TORCH_CHECK(tokens.size(0) >= requests && tokens.size(1) >= max_draft + 1, "tokens capacity exceeded");
  for (const auto& t : {accepted_len, status, path_used}) {
    check(t, at::kInt, 1, "output"); TORCH_CHECK(t.numel() >= requests, "output capacity exceeded");
  }
  std::vector<Tensor> outputs{workspace, tokens, accepted_len, status, path_used};
  for (size_t i = 0; i < outputs.size(); ++i) {
    TORCH_CHECK(no_internal_overlap(outputs[i]), "output has overlapping strides");
    for (const auto& input : inputs)
      TORCH_CHECK(disjoint(outputs[i], input), "exactsample input/output aliasing is forbidden");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(disjoint(outputs[i], outputs[j]), "exactsample outputs alias");
  }
  c10::cuda::CUDAGuard guard(device);
  exactsample::Arguments a{};
  a.target = matrix(logits); a.draft = draft;
  a.cu_rows = vector<int64_t>(cu_rows); a.top_k = vector<int64_t>(top_k); a.draft_kind = vector<int64_t>(draft_kind);
  a.seed = vector<int64_t>(seeds); a.stream = vector<int64_t>(stream_ids); a.position = vector<int64_t>(positions);
  a.temperature = vector<double>(temperature); a.top_p = vector<double>(top_p); a.min_p = vector<double>(min_p);
  a.draft_ids = draft_ids.const_data_ptr<int64_t>(); a.draft_s0 = draft_ids.stride(0); a.draft_s1 = draft_ids.stride(1);
  a.mask = mask_ptr; a.mask_s0 = mask_s0; a.mask_s1 = mask_s1;
  a.tokens = tokens.data_ptr<int64_t>(); a.token_s0 = tokens.stride(0); a.token_s1 = tokens.stride(1);
  a.accepted = accepted_len.data_ptr<int32_t>(); a.accepted_s = accepted_len.stride(0);
  a.status = status.data_ptr<int32_t>(); a.status_s = status.stride(0);
  a.path = path_used.data_ptr<int32_t>(); a.path_s = path_used.stride(0);
  a.rows = rows; a.requests = requests; a.vocab = vocab; a.max_draft = max_draft;
  a.mode = int(mode); a.dispatch = int(dispatch); a.list_max = int(list_max); a.workspace = workspace.data_ptr<uint8_t>();
  auto result = exactsample::launch(a, c10::cuda::getCurrentCUDAStream(device.index()).stream());
  TORCH_CHECK(result == cudaSuccess, "exactsample kernel launch failed: ", cudaGetErrorString(result));
}
}  // namespace

TORCH_LIBRARY(exactsample, m) {
  m.def("verify_out(Tensor target_logits, Tensor cu_rows, Tensor draft_ids, Tensor? draft_logits, "
        "Tensor temperature, Tensor top_k, Tensor top_p, Tensor min_p, Tensor draft_kind, "
        "Tensor seeds, Tensor stream_ids, Tensor positions, Tensor? mask, "
        "Tensor(a!) workspace, Tensor(b!) tokens, Tensor(c!) accepted_len, Tensor(d!) status, "
        "Tensor(e!) path_used, int mode, int dispatch, int max_draft, int list_max) -> ()");
}
TORCH_LIBRARY_IMPL(exactsample, CUDA, m) { m.impl("verify_out", &verify_out); }
