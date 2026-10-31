#include "fused.cuh"

namespace exactsample::fused {
template void launch_rows<BF16>(const Arguments&, cudaStream_t);
}
