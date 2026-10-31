# Batch-invariant sampling and speculative verification in CUDA

A CUDA library (a PyTorch custom op, with a vLLM integration) for the last
step of LLM decoding: sampling the next token from logits, and verifying a
speculative-decoding draft against the target model's logits.

Every random draw is a pure function of `(seed, request stream, absolute
position)` and the request's own logits. A request therefore produces the same
tokens whatever else is in the batch, in whatever slot, however the kernel
splits the work, and after preemption and recomputation. The tokens also
match an FP64 reference implementation: the kernel computes in FP32 with an
error bound and recomputes in FP64 the rare decision the bound does not settle.

It supports temperature, top-k, top-p, min-p, grammar bitmasks and greedy
decoding; ratio (Leviathan/Chen) and coupled verification; one-hot drafts and
full proposal distributions; ragged batches, with a per-request error status
instead of a failed batch.

**Speed on real logits** (Qwen3-8B, 151,936-token vocabulary, RTX 4090; GPU
time per call in microseconds, kernels only, lower is better; full tables
below):

| Workload | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| T=0.6, top-k 20, top-p 0.95, 1 request | **36** | 129 | 96 |
| the same, 128 requests | **96** | 461 | 857 |
| the same, verifying 4 drafts for each of 64 requests (320 rows) | **382** | 1,532 | 4,583 |
| T=0.7, top-p 0.9, 128 requests | **110** | 499 | 794 |
| T=1.0 only, 1 request | 17 | **10** | 56 |
| T=1.0 only, 128 requests | **84** | 177 | 782 |

Temperature-only sampling of up to eight requests is the case vLLM wins on GPU
time, by 1.3 to 1.7 times: its Gumbel kernel is one launch with no error
bounds, while exactsample reads each row twice (statistics, then the certified draw)
in five launches. From 32 requests exactsample is level or ahead, and counting host
overhead (the call time below) it is ahead in every cell, 30 vs 50 us for one
request.

## How it works

[docs/design.md](docs/design.md) has the details, and
[docs/design-tradeoffs.md](docs/design-tradeoffs.md) the main decisions with
their alternatives and costs. In short:

- **Kernels.** A validation pass finds each row's maximum. Temperature-only
  rows are drawn in one streaming pass whose last block commits the result.
  For top-k and top-p rows, a histogram pass counts tokens by distance from the
  maximum and a gather pass collects exactly the bins that hold the top k (or,
  for top-p, the largest prefix of at most 128 tokens). Per-row kernels finish
  the rest. The streaming kernels use at most 64 registers and split a row
  across up to 32 blocks for small batches.
- **Two representations of the kept set.** Up to 256 tokens after top-k are
  sorted and processed in FP64 with the reference's formulas. Larger sets are a
  predicate (an exact key cut plus min-p), evaluated per token in FP32.
- **Certified FP32.** Every FP32 score, weight and sum carries an error bound.
  A draw is decided when the winner's lower bound clears every other token's
  upper bound, a ratio test when `u` falls outside the interval of `p/q`, a
  threshold when the token is outside its bound; anything else is recomputed
  in FP64. On the benchmark's real logits that never happened; the tests build
  inputs that force each fallback.
- **Pruning.** The greedy token's score bounds the winner's from below (the
  runner-up's, when the draft being verified is the greedy token and is
  excluded from the residual draw), so a token whose logit plus the largest
  possible Gumbel noise falls short is skipped without generating its noise.
  A second, cheap bound (two fast logs) screens the rest before the accurate
  noise is computed.
- **Exact selection.** Top-k and top-p cuts are exact, including ties at the
  boundary, which are split by token ID as the contract requires: radix select
  over order-preserving keys in shared memory, from candidates the histogram
  localizes.

## Correctness

| Check | Where |
|---|---|
| Token-for-token agreement with the FP64 NumPy reference: random batches over parameter combinations (temperature, top-k, top-p, min-p), dtype (fp16/bf16/fp32/fp64), stride, mask, draft kind and mode, in both kept-set representations | `tests/test_cuda.py`, `tests/test_fused.py` |
| Inputs that defeat FP32 (scores 1e-10 apart, a ratio test 1e-12 from the threshold, min-p and top-p boundaries on exact sums) are settled in FP64 | `tests/test_fused.py` |
| Batch invariance: alone, permuted, next to other requests, and at different slice counts | `tests/test_fused.py` |
| Exact probability laws (rational arithmetic, every branch enumerated) for the verification algorithms | `reference/exact.py`, `tests/exact/` |
| A preregistered statistical plan (G-tests against exact laws), 2M draws per cell | `tests/stat/`, `tests/stat_plan.yaml`, `results/stat-cuda*.json` |
| compute-sanitizer memcheck, racecheck, initcheck and synccheck | `tests/sanitizer/run.py` |
| Agreement on the captured Qwen3-8B logits | `bench/agreement.py`, `results/agreement.json` |

`bench/agreement.py` runs six workloads from the captured logits (192
requests, 352 rows, with and without 4-token drafts) through the fused path
and the FP64 oracle: every request's tokens, accepted length and status are
identical, and the fused path needed no FP64 fallback. The benchmark records
the fallback counter for every exactsample cell below; it is zero in all 24.

The statistical plan passes on the CUDA path in all ten cells (sampling,
temperature, top-k, a top-p boundary, min-p, one-hot and full-q verification,
accepted length; smallest p-value 0.44). Run with lists disabled
(`--list-max 0`, every row through the FP32 predicate path), it produces
identical counts, as exactness requires. The four compute-sanitizer tools
report no errors (`results/sanitizer/summary.json`).

## Benchmarks

`bench/capture_logits.py` records 512 decode-step logit rows from Qwen3-8B
(32 prompts x 16 steps, sampled at T=0.7). `bench/sampler.py` builds ordinary
requests (one row each) and 4-draft verification requests (five consecutive
steps of one prompt, the tokens actually sampled as drafts) from them, and
times each implementation in its own process:

- **exactsample**: this library.
- **vLLM 0.31 MRv2**: its Triton kernels in the order its sampler applies them
  (temperature, min-p, top-k/top-p, Gumbel sampling or rejection sampling).
- **FlashInfer 0.7.0.post1**: `top_k_top_p_sampling_from_logits` for ordinary
  sampling; softmax, the renormalization APIs and `chain_speculative_sampling`
  for verification.

Each starts from the same bf16 logits. The baselines include their dtype
conversion and probability materialization, because exactsample consumes raw logits.
vLLM skips its top-k/top-p kernels when no request uses them, and exactsample is told
the same (`top_k_top_p`, below). Their RNGs differ from exactsample's, so only
performance is compared here; contract differences are covered by
`conformance/`.

### T=0.6, top-k 20, top-p 0.95 (Qwen3's recommended settings): GPU time per call, us (call time with launch overhead)

| requests x rows each | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| 1 x 1 | 36 (42) | 129 (181) | 96 (103) |
| 8 x 1 | 42 (47) | 137 (182) | 114 (121) |
| 32 x 1 | 53 (58) | 163 (193) | 175 (161) |
| 128 x 1 | 96 (101) | 461 (466) | 857 (844) |
| 1 x 5 | 41 (46) | 134 (238) | 219 (272) |
| 8 x 5 | 59 (59) | 148 (244) | 456 (515) |
| 32 x 5 | 105 (110) | 704 (710) | 2235 (2336) |
| 64 x 5 | 382 (386) | 1532 (1538) | 4583 (4695) |

### T=1.0: GPU time per call, us (call time with launch overhead)

| requests x rows each | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| 1 x 1 | 17 (30) | 10 (50) | 56 (59) |
| 8 x 1 | 20 (29) | 16 (51) | 72 (75) |
| 32 x 1 | 33 (36) | 33 (51) | 94 (101) |
| 128 x 1 | 84 (87) | 177 (179) | 782 (780) |
| 1 x 5 | 24 (30) | 14 (102) | 97 (170) |
| 8 x 5 | 49 (52) | 32 (99) | 200 (269) |
| 32 x 5 | 131 (134) | 232 (237) | 1239 (1328) |
| 64 x 5 | 325 (327) | 524 (526) | 2548 (2680) |

### T=0.7, top-p 0.9: GPU time per call, us (call time with launch overhead)

| requests x rows each | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| 1 x 1 | 39 (43) | 57 (155) | 63 (67) |
| 8 x 1 | 43 (48) | 83 (154) | 93 (100) |
| 32 x 1 | 57 (61) | 224 (235) | 140 (146) |
| 128 x 1 | 110 (114) | 499 (494) | 794 (793) |
| 1 x 5 | 43 (47) | 79 (212) | 164 (214) |
| 8 x 5 | 64 (68) | 384 (400) | 419 (476) |
| 32 x 5 | 135 (140) | 778 (780) | 1964 (2081) |
| 64 x 5 | 415 (418) | 1677 (1681) | 4076 (4054) |

`gpu` is summed kernel time from CUPTI; the call time (CUDA events over 100
back-to-back calls) adds host launch overhead, and a cold call after an L2
flush is in `results/sampler/*.jsonl`. Reproduce with `bench/run_sampler.sh`.

## Use it

```bash
pip install -e '.[test]'                                     # NumPy reference and tests
EXACTSAMPLE_PYTHON=python3 bash scripts/build_cuda.sh                # builds exactsample_cuda.so (PyTorch 2.13, CUDA 13, sm_89)
python -m pytest -q                                          # CPU tests; CUDA tests run when a GPU is present
```

```python
import torch
from exactsample.cuda import CudaRequestParams, CudaRngKey, plan, verify

work = plan(max_rows=320, max_requests=64, max_draft=4, vocab=151936)
out = work.allocate_output()
verify(logits, cu_rows, draft_ids,           # bf16 [rows, V], int64 [R+1], int64 [R, max_draft]
       params=CudaRequestParams(temperature, top_k, top_p, min_p, draft_kind),
       rng=CudaRngKey(seed, stream_id, positions),
       plan=work, out=out, mode="ratio")
# out.tokens[r, :out.accepted_len[r] + 1] holds request r's new tokens when out.status[r] == 0.
```

A plan owns a fixed workspace (12 KB per row) and its outputs; calls on one
stream reuse them without allocation. `verify(..., top_k_top_p=False)` says
that no request has a top-k or top-p (vLLM's bridge knows this from its
host-side parameters): the three kernels that only serve those filters are not
launched, and the result is the same. `docs/contract.md` defines the semantics
and statuses, `docs/rng.md` the RNG address. The vLLM integration is described
in [integration/](integration/README.md).

## Layout

| Path | What it is |
|---|---|
| `csrc/verify/fused.cuh` | the fused kernels |
| `csrc/verify/verify.cu` | validation, the slow FP64 GPU reference path (`dispatch="reference"`), finalize |
| `python/exactsample/` | CUDA wrapper, NumPy FP64 reference, Philox, vLLM bridge |
| `reference/exact.py` | rational-arithmetic enumeration of the verification laws |
| `tests/` | CUDA differential, fused-path, exact, statistical and sanitizer tests |
| `bench/` | logit capture, benchmark, agreement check |
| `conformance/` | adapters for FlashInfer and vLLM |
| `vllm-patches/`, `integration/` | the vLLM 0.31 patch and its checks |

## Limitations

- Built and measured on one RTX 4090 (sm_89). FP64 throughput there is 1/64
  of FP32, which is why FP64 is confined to small lists and fallbacks.
- A request with a very large top-k (thousands) or a top-p nucleus of
  thousands of tokens is finished by one block per row and is several times
  slower than the common cases above.
- Up to eight temperature-only requests take 1.3 to 1.7 times vLLM's GPU
  time ([docs/design-tradeoffs.md](docs/design-tradeoffs.md)).
- CUDA graph capture of the call is not supported.
- The vLLM bridge covers greedy draft models and ordinary sampling; other
  speculative methods and sampling features fall back to vLLM's own path.
