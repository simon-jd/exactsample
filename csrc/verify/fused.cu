// Dispatches the fused path to its per-dtype translation unit.
#include "device.cuh"

namespace exactsample {
namespace fused {
template <DType D> void launch_rows(const Arguments& a, cudaStream_t stream);
extern template void launch_rows<F32>(const Arguments&, cudaStream_t);
extern template void launch_rows<F64>(const Arguments&, cudaStream_t);
extern template void launch_rows<F16>(const Arguments&, cudaStream_t);
extern template void launch_rows<BF16>(const Arguments&, cudaStream_t);
}  // namespace fused

cudaError_t launch_fused(const Arguments& a, cudaStream_t stream) {
  if (a.rows == 0) return cudaSuccess;
  switch (a.target.dtype) {
    case F32: fused::launch_rows<F32>(a, stream); break;
    case F64: fused::launch_rows<F64>(a, stream); break;
    case F16: fused::launch_rows<F16>(a, stream); break;
    case BF16: fused::launch_rows<BF16>(a, stream); break;
  }
  return cudaGetLastError();
}
}  // namespace exactsample
