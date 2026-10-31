// Exhaustive check of the FP32 Gumbel noise against its documented bounds,
// over all 2^32 Philox words (docs/design.md, Certified FP32):
//
//   |gumbel32(w) - G(w)| <= 2^-19 * (1 + |G(w)|)   the error the draws assume
//   gumbel_bound(w)      >= G(w)                   the pruning screen is an upper bound
//   G(w) <= GUMBEL_MAX                             the pruning bar's noise ceiling
//
// where G(w) = -log(-log u), u = (w + 0.5) / 2^32, is computed in FP64 (with
// log1p for u >= 1/2, so the tail keeps its precision). Compile with the
// library's flags (--fmad=false); scripts/check_gumbel_bound.sh does that and
// writes the JSON report.
#include "fused.cuh"
#include <cstdio>
#include <vector>

namespace {
constexpr int BLOCKS = 4096, THREADS = 256;

struct Stats {
  double worst_ratio;   // max |g32 - G| / (2^-19 (1 + |G|))
  double min_margin;    // min gumbel_bound - G
  double max_g, min_g;
  unsigned long long violations, bound_violations;
  unsigned worst_word;
};

__device__ double exact(uint32_t w) {
  double e = w < 0x80000000u ? -log((double(w) + 0.5) * (1.0 / 4294967296.0))
                             : -log1p(-(double(~w) + 0.5) * (1.0 / 4294967296.0));
  return -log(e);
}

__global__ void check(Stats* out) {
  using namespace exactsample::fused;
  Stats s{0., 1e300, -1e300, 1e300, 0, 0, 0};
  const uint64_t stride = uint64_t(BLOCKS) * THREADS;
  for (uint64_t i = uint64_t(blockIdx.x) * THREADS + threadIdx.x; i < (1ull << 32); i += stride) {
    const uint32_t w = uint32_t(i);
    const double g = exact(w);
    const double ratio = fabs(double(gumbel32(w)) - g) / (1.9073486328125e-06 * (1. + fabs(g)));
    if (ratio > s.worst_ratio) { s.worst_ratio = ratio; s.worst_word = w; }
    s.violations += ratio > 1.;
    const double margin = double(gumbel_bound(w)) - g;
    s.min_margin = fmin(s.min_margin, margin);
    s.bound_violations += margin < 0.;
    s.max_g = fmax(s.max_g, g);
    s.min_g = fmin(s.min_g, g);
  }
  out[blockIdx.x * THREADS + threadIdx.x] = s;
}
}  // namespace

int main() {
  const int n = BLOCKS * THREADS;
  Stats* device = nullptr;
  if (cudaMalloc(&device, n * sizeof(Stats)) != cudaSuccess) return 2;
  check<<<BLOCKS, THREADS>>>(device);
  if (cudaDeviceSynchronize() != cudaSuccess) return 2;
  std::vector<Stats> host(n);
  cudaMemcpy(host.data(), device, n * sizeof(Stats), cudaMemcpyDeviceToHost);
  Stats t = host[0];
  for (const Stats& s : host) {
    if (s.worst_ratio > t.worst_ratio) { t.worst_ratio = s.worst_ratio; t.worst_word = s.worst_word; }
    t.min_margin = s.min_margin < t.min_margin ? s.min_margin : t.min_margin;
    t.max_g = s.max_g > t.max_g ? s.max_g : t.max_g;
    t.min_g = s.min_g < t.min_g ? s.min_g : t.min_g;
  }
  t.violations = t.bound_violations = 0;
  for (const Stats& s : host) { t.violations += s.violations; t.bound_violations += s.bound_violations; }
  const bool pass = t.violations == 0 && t.bound_violations == 0 && t.max_g <= exactsample::fused::GUMBEL_MAX;
  std::printf("{\"words\": 4294967296, \"error_bound\": \"2^-19 * (1 + |G|)\", "
              "\"worst_error_over_bound\": %.6f, \"worst_word\": %u, \"error_violations\": %llu, "
              "\"screen_min_margin\": %.9g, \"screen_violations\": %llu, \"max_noise\": %.9f, "
              "\"min_noise\": %.9f, \"gumbel_max\": %.6f, \"passed\": %s}\n",
              t.worst_ratio, t.worst_word, t.violations, t.min_margin, t.bound_violations, t.max_g, t.min_g,
              double(exactsample::fused::GUMBEL_MAX), pass ? "true" : "false");
  return pass ? 0 : 1;
}
