#include "fused.cuh"

namespace exactsample::fused {
template void launch_rows<F16>(const Arguments&, cudaStream_t);
}
