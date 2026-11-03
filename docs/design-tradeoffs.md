# Design trade-offs

The decisions below shaped the kernels and the API. For each one: what was
chosen, what else was on the table, the reason, and the price paid. Timings
are GPU time per call on an RTX 4090 with the captured Qwen3-8B logits
(151,936-token vocabulary), from the README's benchmark.

## Exact tokens: certified FP32 with an FP64 fallback

**Decision.** Every FP32 quantity that feeds a decision (a score, a weight, a
sum) carries a bound on its error. A decision the bound settles is final; any
other is recomputed in FP64 ([design.md](design.md)).

**Alternatives.** Plain FP32, as vLLM and FlashInfer compute; FP64
throughout.

**Why.** In plain FP32 the token depends on rounding and on reduction order,
so it changes with the batch around a request and with how a row is split
across blocks, and it cannot match an FP64 definition. FP64 throughout is
exact, but the 4090 runs FP64 at 1/64 of the FP32 rate, so a streaming pass
over a 151,936-token row in FP64 is compute-bound rather than memory-bound.
With bounds, the 192 captured requests of the
agreement check, and every benchmark cell, needed no FP64 fallback.

**Cost.** Each token carries its bound through the streaming loops, which is
part of why a small temperature-only batch takes longer than vLLM's kernel
(below). Every decision has two implementations, and the tests build inputs
that defeat FP32 to force each fallback. A fallback costs one FP64 pass over
the row by one block.

## A counter-based RNG addressed by request, position and token

**Decision.** Every random value is Philox4x32-10 at `(seed, stream,
position, substream, token / 4)` ([rng.md](rng.md)).

**Alternatives.** A generator state per batch (cuRAND, `torch.Generator`),
where a request's draws depend on its slot and on the draws before it.
Keying by request and position is not new: vLLM's model-runner-v2 sampler
derives its Gumbel noise from a per-request seed and the position, and
FlashInfer's samplers take a seed and an offset per request.

**Why.** A token's noise depends on nothing about the batch: not the slot,
the row number, the launch configuration or how many draws came before. What
this address adds to the per-request keying above is a substream per use
(target draw, ratio acceptance, residual draw), so coupled verification and
ordinary sampling at the same position use the same target draw, and a stream ID
the engine keeps across preemption. Together with exact decisions, the same
key gives the same token on any batch layout, alone or batched, and after
preemption and recomputation.

**Cost.** One Philox call for each group of four tokens that survives
pruning. The engine must keep a request's stream ID across preemption.

## Two representations of the kept set

**Decision.** When at most 256 tokens survive top-k, they are gathered,
sorted and processed as an FP64 list; larger kept sets are a predicate over
the full row in certified FP32.

**Alternatives.** One full-row FP32 path for everything; sorting every row.

**Why.** With real settings (top-k 20, top-p 0.95) the kept set is a handful
of tokens, where FP64 is cheap and uses the oracle's own formulas. A
full-row FP64 path would cost the old kernel's milliseconds.

**Cost.** Two paths that must agree. The differential tests and the
statistical plan run both (`list_max` 256 and 0) and require identical
results. A list row is read twice more than a temperature-only row (histogram
and gather).

## Candidates by distance from the maximum

**Decision.** A 128-bin histogram of each token's distance below the row
maximum tells which bins hold the top-k or top-p candidates; only those bins
are gathered.

**Alternatives.** A radix select or sort over the whole row, as FlashInfer
and vLLM do for top-k and top-p.

**Why.** On real logits the mass sits within a few logits of the maximum, so a
top-k 20 row's candidates fall in a couple of bins and selection runs on at
most 2,048 tokens in shared memory. The Qwen3-settings cells take 2.5 to 6.0
times less GPU time than vLLM's and 1.7 to 12 times less than FlashInfer's.

**Cost.** Two passes over each list row, and a flat distribution or a large k
falls back to a radix select over the whole row in one block.

## Rows split across blocks, merged by the last to arrive

**Decision.** Below 512 rows the streaming kernels split each row over several
blocks (32 below 16 rows). Each block writes a partial result, and the last
block to arrive, found with an atomic counter per row, merges them in slice
order.

**Alternatives.** One block per row; a cooperative launch with a grid
barrier; more blocks per row for tiny batches.

**Why.** One block per row leaves 127 of 128 SMs idle for a single request. The
last-arrival merge needs no grid barrier, and merging in slice order rather
than arrival order keeps the result deterministic.

**Cost.** Partial results in the workspace and one atomic per row and pass.
More slices per row (64 or 128) were tried and saved less on one row than
they cost on a few rows, from more blocks to schedule and merge. For the same
reason `row_stats` runs 256-thread blocks: 512 blocks of 512 threads do not
fit in one wave on 128 SMs.

## A fixed kernel sequence, specialized only from host knowledge

**Decision.** A call enqueues a fixed sequence of kernels, and a block whose
row belongs to another kernel returns at once. The one specialization is
`top_k_top_p=False`, which skips the three kernels that only serve top-k and
top-p lists; the vLLM bridge passes it from its host-side parameters.

**Alternatives.** Device-side dispatch (dynamic parallelism, CUDA graph
conditional nodes); a separate single-kernel path for small temperature-only
batches; a host synchronization to look at the parameters.

**Why.** Which kernel finishes a row depends on the data (how many tokens are
valid, whether a certificate holds), which only the device knows, and a host
synchronization would stall the decode loop. vLLM specializes the same way
(it skips its top-k and top-p kernels from host flags). The hint never
changes a result: without the list kernels, greedy and failed rows are
finished by `fused_rows<FULL>`, and a row that does filter takes the
full-row path.

**Cost.** A launch with nothing to do still costs 1.4 to 1.6 us, and a
temperature-only call is five launches with two passes over each row. Up to
eight such requests, vLLM's single Gumbel kernel takes 1.25 to 1.85 times
less GPU time (10 vs 17 us for one request, 32 vs 49 us for eight 5-row
verifications). From 32 requests exactsample is level or ahead (84 vs 179 us
at 128), and its call time, which adds host overhead, is lower in every cell:
29 vs 50 us for one request. A single-kernel path would need the maximum before the
draw (or the error analysis redone around per-block maxima) and a second
implementation of every fallback.

**One function body.** `fused_rows` is one long kernel body. Splitting it
into `__forceinline__` device helpers (9a369f2) did not give nvcc the same
code: over 10 interleaved runs per kernel (`results/sampler/split-check/`),
the split was 2.5% faster on the largest verification batches with a warm L2
(373 vs 383 us at 64 x 5 with the Qwen3 settings, 405 vs 415 us top-p only)
and level with a cold one (437 vs 436 us), but 0.5 to 1.4% slower in the
other Qwen3-setting cells from 8 requests up (96.1 vs 95.6 us at 128 x 1,
outside that cell's 0.16 us IQR); temperature-only and top-p-only cells
moved by at most 1.3% either way.
The single body was kept (1aa7ebf, whose machine code matches 88df42c),
so the benchmark tables stay those of the measured kernel.

## Every draft position drawn in the same pass

**Decision.** All rows of a request (each draft position and the bonus row)
are drawn in one pass, including each position's residual draw, and
`finalize` keeps the prefix up to the first rejection.

**Alternatives.** Walk the positions in order and stop at the first
rejection.

**Why.** A walk would serialize up to D + 1 passes over the vocabulary per
request. In one pass, verifying four drafts costs about as much as sampling
five independent rows.

**Cost.** Draws for positions after the first rejection, and residual draws
for accepted positions, are computed and thrown away. Such a residual draw
excludes the draft token, usually the argmax, which left the pruning without
its lower bound until a runner-up took its place ([design.md](design.md),
Pruning the draw); with it, `row_scan` takes 32 us at eight 5-row
temperature-only requests.

## A plan with a fixed workspace

**Decision.** `plan()` allocates the workspace (12 KB per row) and the
outputs once, and calls on one stream reuse them without allocating.

**Alternatives.** Allocate per call through the PyTorch caching allocator.

**Why.** No allocator calls on the decode path and nothing to release on an
error path. The layout is fixed by the row count, so no kernel needs a
pointer table.

**Cost.** The caller sizes the plan for its largest batch, and a plan serves
one call at a time (a host lock and a completion event enforce it). CUDA
graph capture is refused in v1.

## A status per request instead of a failed batch

**Decision.** Invalid parameters, NaN or +inf logits, a request with no valid
token and a draft with zero probability produce a status for that request,
and the others complete. Only malformed offsets fail the whole call.

**Alternatives.** Raise on any bad input, as most samplers do.

**Why.** In a serving engine one bad request (a grammar mask that leaves
nothing, say) must not take down the batch; the vLLM bridge finishes it with
an error and the rest continue.

**Cost.** Every kernel checks its request's status first, and malformed
offsets still fail everything, since rows can no longer be attributed to
requests.
