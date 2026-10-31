#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define EXACTSAMPLE_HD __host__ __device__
#else
#define EXACTSAMPLE_HD
#endif

namespace exactsample {
struct Uint4 { uint32_t x, y, z, w; };

// Random123 Philox4x32-10. Address words are part of the v1 contract.
EXACTSAMPLE_HD inline Uint4 philox(uint64_t seed, uint32_t stream, uint32_t position,
                         uint32_t substream, uint32_t token_block) {
  Uint4 c{stream, position, substream, token_block};
  uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
  for (int round = 0; round < 10; ++round) {
    uint64_t a = uint64_t(0xD2511F53u) * c.x;
    uint64_t b = uint64_t(0xCD9E8D57u) * c.z;
    c = {uint32_t(b >> 32) ^ c.y ^ k0, uint32_t(b),
         uint32_t(a >> 32) ^ c.w ^ k1, uint32_t(a)};
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c;
}

EXACTSAMPLE_HD inline double uniform(uint64_t seed, uint32_t stream, uint32_t position,
                             uint32_t substream, uint32_t token) {
  Uint4 c = philox(seed, stream, position, substream, token / 4);
  uint32_t u = token % 4 == 0 ? c.x : token % 4 == 1 ? c.y : token % 4 == 2 ? c.z : c.w;
  return (double(u) + 0.5) * (1.0 / 4294967296.0);
}
}  // namespace exactsample
