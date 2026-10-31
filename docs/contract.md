# exactsample decoding contract v1.0

This document defines exactsample, not a claim of compatibility with a third-party
sampler. Contract changes require a version change and corresponding oracle,
CUDA, and independent-test changes.

## Shapes and failure boundaries

Target logits have shape `[M,V]`. `cu_rows[R+1]` partitions them into requests;
a request with D proposals owns D+1 consecutive rows, including a bonus row.
`draft_ids[R,K]` uses only the first D entries. A zero-length segment is invalid;
D=0 is ordinary sampling. Empty batches are valid. Offsets, capacities, shapes
and dtypes are checked before unsafe access. Malformed global structures raise
an exception; numerical errors belonging to a request produce a status.

Outputs are `tokens[R,K+1]`, `accepted_len[R]`, `status[R]`, and `path_used[R]`.
Unused tokens are -1. `accepted_len` counts accepted proposals, not the final
replacement or bonus token. Every successful request emits accepted_len+1
tokens. A failed request emits no tokens: its entire output is -1 and its
accepted length is zero. An error in a later supplied row invalidates the whole
request even if an earlier rejection would make that row unused.

| Status | Value | Meaning |
|---|---:|---|
| OK | 0 | Tokens may be committed |
| NO_VALID_TOKEN | 1 | No finite allowed target logit remains |
| INVALID_INPUT | 2 | Invalid parameters, NaN/+inf, invalid IDs or offsets |
| DRAFT_ZERO_PROB | 3 | A proposed token has zero probability under actual q |

## Target distribution

1. Apply the packed int32 bitmask (bit 0 means excluded). Each row contains
   ceil(V/32) words in xgrammar layout; padding bits are ignored. Masked NaN or
   +inf values are still INVALID_INPUT: validation checks the raw row before
   masking, so malformed inputs cannot be hidden by a grammar mask. -inf is
   allowed and carries zero mass.
2. Temperature must be finite and nonnegative. At zero, choose the largest
   finite allowed logit, breaking ties by the lowest token ID; the target is a
   point mass and the following filters have no effect.
3. At positive temperature, divide by temperature and apply min-p: retain
   weights `exp((logit - maximum)/temperature) >= min_p`. min_p is in [0,1].
4. Apply top-k. Zero disables it; otherwise k is an integer in [1,V]. Retain
   exactly k entries (or all remaining entries if fewer), ordered by descending
   logit and then ascending token ID. Boundary ties do not expand the support.
5. Normalize the surviving weights for top-p. top_p is in (0,1]. Keep the
   shortest prefix in the same order whose cumulative probability is at least
   top_p; equality includes the current token and excludes the next one.
6. Normalize the final surviving weights. Filtering never reinstates a token.

FP64 is the independent oracle's arithmetic. The CUDA fused path computes
in FP32 with error bounds and recomputes in FP64 any decision the bounds do not
settle ([design.md](design.md)), so its tokens match the FP64 definition. The
remaining difference from the NumPy oracle is that of two FP64 libraries: a
decision that falls within a few ulps of a threshold can depend on which
`exp` or `log` rounds it.

## Sampling and verification

Categorical selection is `argmax(log(p_i) - log(-log(u_i)))` over positive
probabilities, with lowest-ID tie breaking. The RNG address is specified in
`rng.md`. A single ordinary draw uses substream 0.

For one-hot draft, q is a point mass at the proposed ID. For full draft,
`draft_logits[M,V]` are **log weights of the distribution actually used to draw
the proposals**, after all draft-side filtering. They are normalized directly;
target parameters and target masks are not applied to q. Bonus-row q is unused.
The proposal ID must have positive q. An all-minus-infinity proposal row has
NO_VALID_TOKEN; a valid proposal distribution assigning zero mass to its
proposed ID has DRAFT_ZERO_PROB.

Ratio mode accepts proposal x iff `u < min(1,p[x]/q[x])`, using substream 1.
On the first rejection, draw from `max(p-q,0)` normalized, using substream 2,
and stop. If all proposals pass, sample the bonus target row with substream 0.
An impossible zero-residual rejection is an error, not an EOS fallback.

Coupled mode supports only one-hot draft. Sample each target row with substream
0 and compare the sample to its proposal. Commit the matching prefix and the
first mismatch, or a bonus if all proposals match. With identical target logits
and absolute positions this is exactly ordinary sampling up to the first
mismatch. It does not promise equality when model forward logits differ.

## Ownership

A CUDA plan owns fixed-capacity workspaces and output storage. Inputs and the
plan must remain alive until the completion event. Overlapping use of one plan
is rejected; independent plans are required for simultaneous streams. Output
views are overwritten on the next completed use and must be copied if retained.
Structural errors raise before launch; asynchronous numerical statuses must be
checked before any output token is submitted to an inference engine.
