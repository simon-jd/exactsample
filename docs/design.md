# Design

The contract (what is computed) is in [contract.md](contract.md) and the RNG
address in [rng.md](rng.md). This document covers how the CUDA fused path
computes it, and why its results are exact and batch invariant.

## Pipeline

A call enqueues these kernels on the caller's stream:

| Kernel | Grid | Regs | Work |
|---|---|---|---|
| `initialize` | per request | | global `cu_rows` check (every block makes it), parameter checks, output reset, counter reset |
| `row_stats` | rows x slices | 40 | validates the row (NaN, +inf, masked), finds its maximum and the lowest-ID argmax, and for a row that makes a ratio test a runner-up (below); the same for q on full-q draft rows. The last slice to finish merges the slices and decides how the row is finished |
| `row_scan` | rows x slices | 64 | sliced rows (temperature only, no draft or a one-hot draft): the draw, and the weight sum for a ratio test. The last slice merges and commits the result when it is certified |
| `row_hist` | rows x slices | 52 | list rows with a top-k or top-p: a histogram of distances below the maximum (128 bins of 0.5 logits); for top-p rows also the FP32 sum of kept weights with its bound |
| `row_gather` | rows x slices | 40 | the token IDs in the bins the row needs (below) |
| `fused_rows<LIST>` | per row | 64 | greedy rows, failed rows, top-k lists, top-p lists |
| `fused_rows<FULL>` | per row | 128 | rows with a large top-k, a top-p nucleus too large for a list, full-q rows, sliced rows `row_scan` could not certify, and greedy and failed rows when there are no list kernels |
| `finalize` | warp per request | | commits the accepted prefix and the final token; one lane per row, ballots for the first failed row and the first rejection |

`row_hist`, `row_gather` and `fused_rows<LIST>` are launched only when
`list_max` is positive. A caller sets it to zero with `top_k_top_p=False`
when no request has a top-k or top-p; `row_stats` then sends greedy and failed
rows to `fused_rows<FULL>` and top-k or top-p rows, should there be any, to the
full-row path, so the result does not change. A temperature-only call is five
launches, and any other eight.

`slices` is 32 below 16 rows, `512 / rows` up to 512 rows and 1 above, so a
single row is spread over 32 SMs while a large batch runs one block per row.
The streaming kernels are kept small (256 threads and at most 64 registers, so
four to six blocks fit an SM and a 512-block grid runs in one wave); the
selection logic and every FP64 fallback live in `fused_rows`, so they never add
register pressure to a streaming loop. A `fused_rows` block whose row belongs
to the other variant returns before it loads anything; that early exit is most
of the cost of the variant a batch does not use. Its arguments are
`__grid_constant__`: the kernel passes them by reference to code that is not
inlined, and the compiler otherwise copies all 736 bytes to local memory in
every thread before the exit; with them in constant space, a launch with
nothing to do costs 1.4 to 1.6 us.

Logits are read in groups of four tokens, which is also the Philox block: one
Philox call yields the noise for all four. Contiguous rows (checked on the
host) are read with one vector load per group, two groups in flight per thread.

Each sliced pass writes one partial result per slice to the workspace; the
last block to arrive (an atomic counter per row) merges the slices in slice
order in one warp. The merge never depends on which block arrived last.

## Two representations of the kept set

After mask, temperature, min-p, top-k and top-p, the kept set is always a
prefix of the order (logit descending, token ID ascending): min-p, top-k and
top-p each remove a suffix of that order. The kernel represents it in one of
two ways.

**List.** When top-k leaves at most 256 tokens (`list_max`), they are gathered,
sorted and processed in FP64 with the oracle's own formulas: weights
`exp((l - max) / T)`, min-p and positivity on those weights, the shortest top-p
prefix by double-double prefix sums, normalization, and `log(p) + gumbel(u)`
for each draw. Double-double sums agree with the oracle's `math.fsum`.

**Full row.** Otherwise the kept set is a predicate evaluated per token: an
exact key cut (from top-k or top-p) plus min-p and positivity. Sums and Gumbel
scores are computed in FP32 together with an error bound, as described next.

## Certified FP32

Every FP32 quantity that feeds a decision carries a bound on its distance from
the exact value.

| Quantity | FP32 computation | Bound used |
|---|---|---|
| `x = (l - max) / T` | `(l - max) * (1/T)` | `4.8e-7 * |x|` (three roundings) |
| `w = exp(x)` | `expf(x)` | `4.8e-7 * (1 + |x|) * w` |
| Gumbel `G = -log(-log u)` | `-__logf(e)`, with `e = -__logf(u)` for `u < 1/2` and `e = -log1pf(-(1-u))` above | `2^-19 * (1 + |G|)`, inside the score bound below |
| score `s = x + G` | one addition | `2^-19 * (1 + |x| + |G|)` |
| a sum of weights | Kahan sum, then FP64 across threads | the sum of the per-token bounds, plus `4.8e-7` of the total, plus `4.25e-18` per weight below `e^-40` that is not computed |

`-(1-u)` is formed from the complement of the 32-bit word, so the noise of
the winning tail (u close to 1) keeps its full 32-bit resolution.

**Draws.** Each token contributes an interval `[lo, hi]` around its FP32 score.
The reduction keeps the token with the largest FP32 score (lowest ID on ties)
and the largest `hi` of every other token. That merge is associative and
commutative, so its result does not depend on the order or grouping of the
reduction. If the winner's `lo` exceeds every other `hi`, the winner is the
exact FP64 winner. Otherwise the row is rescored in FP64: every token whose
`hi` reaches the FP32 winner's `lo` is recomputed with `x64 + gumbel64`, and
the exact winner is among them.

**Ratio test.** `u < min(1, p/q)` is decided from the interval of `p/q`
implied by the sum's bound; if `u` falls inside it, the sum is recomputed in
FP64 (double-double) and the test redone exactly.

**Thresholds.** Min-p and positivity (`exp(x) > 0`) are decided in FP32 when
`x` is further than its bound from the threshold, and in FP64 for the token
otherwise. Top-p is described below.

The workspace counts FP64 recomputations (`CudaPlan.fallbacks()`). On the
benchmark's real logits the count is zero; the tests construct inputs that
force each fallback.

Because every decision is either certified or recomputed in FP64, the result
does not depend on reduction order, on how a row is split across blocks, or
on what else is in the batch. Different slice counts produce bitwise the same
tokens.

## Pruning the draw

The greedy token is always kept, and its score is computed first; a lower
bound on it bounds the winner's score from below. A token can only win if
`x + G_max` reaches that bound, where `G_max = 22.875` bounds the noise any
32-bit uniform produces. Tokens below are skipped without Philox; on real
logits that is most of the vocabulary.

A ratio test's residual draw excludes the draft token, and the draft is
usually the greedy token, which would leave no bound at all: every token would
get its noise computed. For those rows `row_stats` also keeps a runner-up, the
best of the other threads' maxima (a valid token near the top, found in the
block reduction at no cost per token). When it is certainly kept (min-p and a
weight above FP64 underflow, by the FP32 tests with their margins) its score
bounds the draw instead. Without it those draws had no lower bound and every
token's noise was computed; with it, `row_scan` takes 32 us for eight
temperature-only requests of five rows.

For tokens that remain, a cheap upper bound on the noise comes first: two
`__logf` and the inequality `-log(u) >= 1 - u`. Only tokens whose bound
reaches the thread's best lower bound so far get the accurate noise. A
skipped token's upper bound still enters the "every other `hi`" of the
certificate, so skipping never weakens it.

## Candidates: the distance histogram

On real logits the mass sits in a few tokens near the maximum (on the Qwen3-8B
capture the 20th-largest logit is typically more than 20 below the first), so
the tokens a list row needs are found by distance from the maximum rather than
by sorting the row. `row_hist` counts valid tokens in 128 bins of 0.5 logits
(distances 0 to 64, plus one bin for everything further). From the histogram:

- **top-k**: the smallest bin prefix holding at least k tokens, if it holds at
  most 2,048 (`window_of`);
- **top-p without top-k**: the largest bin prefix holding at most 128 tokens
  (`window_prefix`).

`row_gather` then writes the IDs of every valid token in those bins to the
row's candidate buffer (one atomic per warp). `fused_rows` checks that the
gathered count matches the histogram before it trusts the buffer.

## Top-k selection

With a gathered buffer, `fused_rows` radix-selects the k-th token in shared
memory: 8-bit digits of an order-preserving key, then of the token ID among
equal keys. It exits early once a digit's bucket is entirely kept, and then
computes the exact cut (the smallest kept key and the largest ID at that key),
since tokens outside the buffer may share the prefix. Without one (k is
larger than 2,048 tokens of the window, or ties fill the histogram), the
block radix-selects the exact key over the whole row and resolves ties by a
scan in ID order. Either way the cut is exact: ties at the boundary are split
by token ID, as the contract requires.

## Top-p

**As a list.** A top-p row without top-k first takes the histogram prefix of
at most 128 tokens. `fused_rows<LIST>` sorts it, computes FP64 weights and
double-double prefix sums, and finds the shortest prefix reaching
`top_p * total`. If the list is the whole kept set, the total is its own sum.
Otherwise the total is `row_hist`'s FP32 sum, an interval, and the boundary is
accepted only if every total in the interval puts it at the same token; the
list must also reach it. If not, the row moves to `fused_rows<FULL>`.

**Over the full row.** The block scans the kept set in sorted chunks: each is
a bin-aligned prefix of what is left, as large as the 2,048-entry buffer allows
(the first at most 256, since the nucleus is usually small), or an exact cut
of 1,024 tokens when one bin alone overflows. Chunks are sorted with CUB's
block radix sort, and FP32 prefix sums with their bounds are accumulated until
the prefix certainly reaches `top_p * total`, where the total is the FP32 sum
with its own bound. If the bounds cannot tell whether the boundary has been
crossed, the block computes the total and the mass before the chunk in FP64
and redoes the chunk with double-double prefix sums. The final prefix sum is
also the normalizer for the ratio test.

## Costs

- Every row is read by `row_stats`; sliced rows once more by `row_scan`; list
  rows with a top-k or top-p twice more (`row_hist`, `row_gather`). At 320 rows
  (97 MB of bf16 logits) `row_stats`, `row_hist` and `row_gather` take 100 to
  120 us each, close to the memory bandwidth; `row_scan`, which also computes
  noise for the tokens it cannot prune, takes 215 us on temperature-only rows.
  At one row fixed costs dominate: for a temperature-only call, `row_stats`
  takes about 4 us, `row_scan` 8, and each of the other three launches 1.4 to
  1.9 (per-kernel times from CUPTI, `results/sampler/runs/`).
- List work is FP64 on at most 256 tokens in one warp (about 17 us for one
  row on one SM, overlapped across rows).
- A full-row top-p touches the row once per chunk of the nucleus; a flat
  distribution with a large nucleus is the slowest case.
- An FP64 fallback costs one FP64 pass over the row by one block.
- A large top-k that the window cannot isolate costs up to eight radix passes.

## Determinism and batch invariance

The RNG address is `(seed, stream, position, substream, token)`; no batch slot
or launch parameter enters it. Every row is processed by the same code
whatever else is in the batch, every reduction has a fixed order, candidate
buffers filled by atomics are sorted or radix-selected before use (both depend
only on the set), and certified decisions do not depend on rounding. A request
therefore produces the same tokens alone, in any slot, next to any other
requests, and under any slice count.
