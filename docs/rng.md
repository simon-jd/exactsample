# Philox address contract v1.0

exactsample implements Philox4x32-10, with multipliers 0xD2511F53 and 0xCD9E8D57,
and Weyl constants 0x9E3779B9 and 0xBB67AE85. Arithmetic wraps modulo 2^32.

The key is `(seed & 0xffffffff, seed >> 32)`. The four counter words are
`(stream_id, position, substream, token_id // 4)`, and the selected output lane
is `token_id % 4`. Seed is an unsigned 64-bit bit pattern. Stream and absolute
position are unsigned 32-bit integers; values outside the domain are rejected
rather than truncated. Batch slots, row numbers, CUDA stream numbers and launch
configuration are never included. A scheduler must retain stream IDs across
preemption and allocate distinct ones for n>1 request continuations.

| Substream | Purpose |
|---:|---|
| 0 | Ordinary target sampling, coupled verification, bonus token |
| 1 | Ratio acceptance (token address 0) |
| 2 | Residual categorical sampling |
| 3 | Reserved; must not be used in v1 |

Convert a word w to double using `(w + 0.5) / 4294967296`. This produces an
open-interval uniform and avoids logarithms of zero. Reusing a complete key
intentionally replays a sample. Different algorithms upstream have different
RNG contracts; distribution comparisons do not require identical token streams.

Known-answer primitive check: zero counter and zero key produce
`6627e8d5 e169c58d bc57ac4c 9b00dbd8`. This checks the primitive separately from
counter packing. The implementation uses no cuRAND state or skip-ahead calls.
