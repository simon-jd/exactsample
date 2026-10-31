#include "fused.cuh"

namespace exactsample::fused {
template void launch_rows<F64>(const Arguments&, cudaStream_t);
}
