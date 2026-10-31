#include "fused.cuh"

namespace exactsample::fused {
template void launch_rows<F32>(const Arguments&, cudaStream_t);
}
