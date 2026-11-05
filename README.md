# exactsample: batch-invariant sampling and speculative verification in CUDA

A CUDA library (a PyTorch custom op, with a vLLM integration) for the last
step of LLM decoding: sampling the next token, and verifying a speculative
draft against the target model's logits.

FP32 rounding, tie order and the kernel's split of the work can change which
tokens survive a top-k or top-p cut. exactsample defines every decision in
FP64, computes in FP32 with an error bound, and recomputes in FP64 only what
the bound does not settle; each random draw is keyed by `(seed, request
stream, absolute position)`. A request gets the FP64 reference's tokens
whatever else is in the batch (192 of 192 requests in `results/agreement.json`).

**Scope:** one RTX 4090 (sm_89), clocks not locked. Kernel timings use 512
logit rows captured from Qwen3-8B decoding (not synthetic); end-to-end runs
serve Qwen3-1.7B and Qwen3-8B in vLLM 0.31.

| Median GPU time per call, us, 10 runs | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| T=0.6, top-k 20, top-p 0.95 (Qwen3's settings), 128 requests | **96** | 455 | 626 |
| Qwen3's settings, 64 requests verifying 4 drafts each | **382** | 1,535 | 2,924 |
| T=1.0 only, 128 requests | **84** | 179 | 172 |
| T=1.0 only, 1 request | 17 | **10** | 26 |
| T=1.0 only, 8 requests verifying 4 drafts each | 49 | **32** | 91 |
| T=0.7, top-p 0.9, 1 request | 39 | 56 | **34** |

Source: `results/sampler/summary.json` (`gpu_us.median`); every shape is in
[Benchmarks](#benchmarks). vLLM's sampler is the one the integration replaces,
and FlashInfer is a widely used standalone GPU sampling library with the same
filters and draft verification; no hand-tuned radix top-k kernel was compared.

## Limitations

- Up to eight temperature-only requests take 1.3 to 1.8 times vLLM's GPU time,
  one top-p-only request 1.14 times FlashInfer's ([details](#where-exactsample-loses)).
- End to end the gain is small: TPOT drops 0.5 to 2.5% with top-k and top-p,
  and **rises 0.1 to 0.7% with temperature alone** ([details](#end-to-end-in-vllm)).
- One GPU (sm_89) only; a top-k or top-p set of thousands of tokens is several times slower.
- Batch invariance covers the sampler, not the model ([details](#batch-invariance)).
- No CUDA graph capture; the vLLM bridge covers greedy draft models and ordinary sampling.

## More

- How it works: [below](#how-it-works), [docs/design.md](docs/design.md), and the
  decisions with their costs in [docs/design-tradeoffs.md](docs/design-tradeoffs.md).
- Results: [Correctness](#correctness), [Benchmarks](#benchmarks), [integration/](integration/README.md)
  (vLLM), and [results/README.md](results/README.md) (each result file's commit and date).
- Reproducing: [Use it](#use-it) and the commands under [Benchmarks](#benchmarks).

Start here in the source:

- `csrc/verify/fused.cuh`: the fused kernels (section index at the top)
- `csrc/verify/verify.cu`: validation, the FP64 GPU reference path, finalize
- `python/exactsample/cuda.py`: the Python API (`plan`, `verify`)
- `python/exactsample/reference.py`: the FP64 NumPy reference
- `docs/contract.md`: the semantics and per-request statuses
- `tests/test_fused.py`: fused-path tests, including inputs that force each FP64 fallback

## What it supports

Temperature, top-k, top-p, min-p, grammar bitmasks and greedy decoding; ratio
(Leviathan/Chen) and coupled verification; one-hot drafts and full proposal
distributions; ragged batches, with a per-request error status instead of a
failed batch.

Every random draw is a pure function of `(seed, request stream, absolute
position)` and the request's own logits. A request therefore produces the same
tokens whatever else is in the batch, in whatever slot, however the kernel
splits the work, and after preemption and recomputation.

## What is new here

Keying the noise by request and position is not new. vLLM's model-runner-v2
sampler derives its Gumbel noise from a per-request seed and the position, and
FlashInfer's samplers take a seed and an offset per request. vLLM's
batch-invariant mode makes the model's logits independent of the batch.

What exactsample adds is that the decisions themselves are exact. Top-k and
top-p cuts split ties by token ID; min-p and verification thresholds are
settled against an FP64 definition. Equal logits and keys therefore give equal
tokens on any batch layout and any kernel split.

It also gives each use of randomness its own substream (target draw, ratio
acceptance, residual draw), and reports a status per request instead of
failing the batch.

## How it works

[docs/design.md](docs/design.md) has the details, and
[docs/design-tradeoffs.md](docs/design-tradeoffs.md) the main decisions with
their alternatives and costs. In short:

- **Kernels.** A validation pass finds each row's maximum. Temperature-only
  rows are drawn in one streaming pass whose last block commits the result.
  For top-k and top-p rows, a histogram pass counts tokens by distance from the
  maximum, and a gather pass collects exactly the bins that hold the top k (or,
  for top-p, the largest prefix of at most 128 tokens). Per-row kernels finish
  the rest. The streaming kernels use at most 64 registers and split a row
  across up to 32 blocks for small batches.
- **Two representations of the kept set.** Up to 256 tokens after top-k are
  sorted and processed in FP64 with the reference's formulas. Larger sets are a
  predicate (an exact key cut plus min-p), evaluated per token in FP32. FP64
  throughput on the RTX 4090 is 1/64 of FP32, which is why FP64 is confined to
  small lists and fallbacks.
- **Certified FP32.** Every FP32 score, weight and sum carries an error bound.
  A draw is decided when the winner's lower bound clears every other token's
  upper bound. A ratio test is decided when `u` falls outside the interval of
  `p/q`, a threshold when the token is outside its bound. Anything else is
  recomputed in FP64. On the benchmark's real logits that never happened; the
  tests build inputs that force each fallback.
- **Pruning.** The greedy token's score bounds the winner's from below (the
  runner-up's, when the draft being verified is the greedy token and is
  excluded from the residual draw). A token whose logit plus the largest
  possible Gumbel noise falls short is skipped without generating its noise. A
  second, cheap bound (two fast logs) screens the rest before the accurate
  noise is computed.
- **Exact selection.** Top-k and top-p cuts are exact, including ties at the
  boundary, which are split by token ID as the contract requires: radix select
  over order-preserving keys in shared memory, from candidates the histogram
  localizes.

## Correctness

| Check | Where |
|---|---|
| Token-for-token agreement with the FP64 NumPy reference: random batches over parameter combinations (temperature, top-k, top-p, min-p), dtype (fp16/bf16/fp32/fp64), stride, mask, draft kind and mode, in both kept-set representations | `tests/test_cuda.py`, `tests/test_fused.py` |
| The FP32 Gumbel noise is within its error bound, `2^-19 (1 + |G|)`, for all 2^32 Philox words (worst case 0.13 of the bound), and the pruning screen never undercuts it | `tests/exhaustive/gumbel_bound.cu`, `scripts/check_gumbel_bound.sh`, `results/gumbel-bound.json` |
| Inputs that defeat FP32 (scores 1e-10 apart, a ratio test 1e-12 from the threshold, min-p and top-p boundaries on exact sums) are settled in FP64 | `tests/test_fused.py` |
| Batch invariance: alone, permuted, next to other requests, and at different slice counts | `tests/test_fused.py` |
| Exact probability laws (rational arithmetic, every branch enumerated) for the verification algorithms | `reference/exact.py`, `tests/exact/` |
| A statistical plan fixed in advance, its sha256 recorded in each result (G-tests against exact laws), 2M draws per cell, on small vocabularies (2 to 4 tokens, FP64 inputs) where the exact law is enumerable | `tests/stat/`, `tests/stat_plan.json`, `results/stat-cuda*.json` |
| compute-sanitizer memcheck, racecheck, initcheck and synccheck over the CUDA tests (the full-vocabulary cases are skipped under the sanitizer, which would take hours) | `tests/sanitizer/run.py`, `results/sanitizer/summary.json` |
| Agreement on the captured Qwen3-8B logits | `bench/agreement.py`, `results/agreement.json` |

`bench/agreement.py` runs six workloads from the captured logits (192
requests, 352 rows, with and without 4-token drafts) through the fused path
and the FP64 oracle. Every request's tokens, accepted length and status are
identical, and the fused path needed no FP64 fallback. The benchmark records
the fallback counter for every exactsample cell below; it is zero in all 24.

The capture is not in the repository ([Benchmarks](#benchmarks)), but a 64-row
subset of it is. From a clone, `python bench/agreement.py --capture
results/logits/qwen3-8b-8x8.npz --output results/agreement-8x8.json` runs the
same six workloads, 192 requests and 352 rows, drawn from 64 distinct captured
rows instead of 112. They are all identical too, with no fallback
(`results/agreement-8x8.json`). That is the same check on less data; the
figures above are the full capture's.

The statistical plan passes on the CUDA path in all ten cells (sampling,
temperature, top-k, a top-p boundary, min-p, one-hot and full-q verification,
accepted length; smallest p-value 0.44). Run with lists disabled
(`--list-max 0`, every row through the FP32 predicate path), it produces
identical counts, as exactness requires.

The four compute-sanitizer tools report no errors. The record is
`results/sanitizer/summary.json`: each tool's command and exit status. The
per-tool logs it names are not committed. `python tests/sanitizer/run.py`,
run from the repository root on a GPU with the library built, reruns the four
tools and writes the logs and a new summary to `results/sanitizer/`.
[results/README.md](results/README.md) says which kernel each recorded run
checked.

## Benchmarks

`bench/capture_logits.py` records 512 decode-step logit rows from Qwen3-8B
(32 prompts x 16 steps, sampled at T=0.7). The file is 155 MB and not in git.
Its sha256 is `7ce5e6575abc529fbbd8bfd3636e6389f448ad3f24f84443a9bf8d51e2f07d2a`,
and in the environment of `bench/requirements.txt` this command regenerates it
byte for byte on the measuring machine:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_PTXAS_PATH=$(command -v ptxas) python bench/capture_logits.py \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --output results/raw/logits/qwen3-8b.npz
```

`results/logits/qwen3-8b-8x8.npz` (14 MB, `bench/subset_logits.py`) holds its
first 8 prompts x 8 steps. `bench/sampler.py` builds ordinary requests (one
row each) and 4-draft verification requests (five consecutive steps of one
prompt, the tokens actually sampled as drafts) from them, and times each
implementation in its own process:

- **exactsample**: this library, with its plan (a fixed workspace) made once.
- **vLLM 0.31, model-runner-v2**: the sampler's own sequence for seeded
  requests: an FP32 copy of the logits, temperature and min-p in place, the
  per-step gather of each row's top-k and top-p, vLLM's Triton top-k/top-p
  kernel, then its Triton Gumbel sampler or rejection sampler.
- **FlashInfer 0.7.0.post1**: both public entry points, scaled logits into
  `sampling_from_logits` or `top_k_top_p_sampling_from_logits`, and
  `sampling.softmax` with per-row temperature into the probability samplers.
  Each cell shows the faster of the two (named in
  `results/sampler/summary.json`). Verification is softmax, the
  renormalization APIs and `chain_speculative_sampling` on a view of the
  rows, with a persistent one-hot draft buffer.

Everything a caller can keep across decode steps (index and parameter
tensors, buffers) is prepared outside the timed call for every
implementation, and all start from the same bf16 logits. Their RNGs differ,
so only speed is compared here; contract differences are covered by
`conformance/`. vLLM skips its top-k/top-p kernels when no request uses
them, and exactsample is told the same (`top_k_top_p`, below).

`bench/run_sampler.sh` runs every implementation 10 times, interleaved, each
run a fresh process holding the GPU lock. A run reports three times per call:

- the GPU time: kernel time from CUPTI, averaged over 20 profiled calls;
- the call time: CUDA events over 100 back-to-back calls, so launch overhead
  included;
- a cold call: median of 30 calls, each after writing 256 MB to evict the
  72 MB L2.

The tables give the median GPU time over runs, its interquartile range, and
the cold-call median in brackets.

Run-to-run spread is small. The interquartile range is within 1% of the
median in 69 of 72 cells, and 1.55% in one more. In the other two it is
10.03% (exactsample, 32 x 5, and FlashInfer, 1 x 5, both with Qwen3's
settings).

Clocks were not locked: there are no administrator rights on the machine,
whose GPU also drives a desktop. The SM clock at the start of each run,
recorded in `results/sampler/runs/clocks.csv`, was 2,805 or 2,820 MHz in 39
of the 40 runs; one sample caught the GPU still at its 210 MHz idle clock.

### T=0.6, top-k 20, top-p 0.95 (Qwen3's recommended settings): median GPU time per call, us (IQR); cold L2 in brackets

| requests x rows each | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| 1 x 1 | 36 (36–36) [48] | 128 (128–128) [147] | 60 (60–60) [66] |
| 8 x 1 | 42 (42–42) [54] | 137 (137–137) [158] | 78 (78–78) [90] |
| 32 x 1 | 53 (53–53) [74] | 163 (163–163) [202] | 122 (122–122) [153] |
| 128 x 1 | 96 (96–96) [153] | 455 (455–456) [486] | 626 (624–626) [658] |
| 1 x 5 | 41 (41–41) [53] | 120 (120–120) [139] | 146 (133–148) [150] |
| 8 x 5 | 59 (59–59) [84] | 147 (147–147) [194] | 292 (292–292) [338] |
| 32 x 5 | 117 (106–117) [187] | 704 (703–704) [751] | 1395 (1394–1395) [1431] |
| 64 x 5 | 382 (382–387) [436] | 1535 (1534–1535) [1595] | 2924 (2921–2924) [2968] |

### T=1.0: median GPU time per call, us (IQR); cold L2 in brackets

| requests x rows each | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| 1 x 1 | 17 (17–17) [22] | 10 (10–10) [17] | 26 (26–26) [31] |
| 8 x 1 | 20 (20–20) [30] | 16 (16–16) [28] | 41 (41–41) [51] |
| 32 x 1 | 33 (33–33) [51] | 33 (33–33) [64] | 55 (55–55) [87] |
| 128 x 1 | 84 (83–84) [134] | 179 (179–179) [211] | 172 (172–172) [199] |
| 1 x 5 | 24 (24–24) [32] | 13 (13–13) [23] | 67 (67–67) [79] |
| 8 x 5 | 49 (49–49) [69] | 32 (32–32) [70] | 91 (91–91) [131] |
| 32 x 5 | 131 (130–131) [192] | 232 (232–233) [267] | 475 (475–475) [509] |
| 64 x 5 | 324 (323–324) [381] | 523 (523–524) [581] | 1008 (1008–1008) [1048] |

### T=0.7, top-p 0.9: median GPU time per call, us (IQR); cold L2 in brackets

| requests x rows each | exactsample | vLLM 0.31 | FlashInfer 0.7 |
|---|---:|---:|---:|
| 1 x 1 | 39 (39–39) [49] | 56 (56–57) [68] | 34 (34–34) [39] |
| 8 x 1 | 43 (43–43) [55] | 83 (83–83) [102] | 49 (49–49) [59] |
| 32 x 1 | 57 (57–57) [78] | 223 (223–224) [261] | 63 (63–63) [96] |
| 128 x 1 | 109 (109–109) [165] | 474 (474–474) [522] | 394 (394–394) [429] |
| 1 x 5 | 43 (43–43) [55] | 79 (79–79) [96] | 112 (111–112) [127] |
| 8 x 5 | 64 (64–64) [88] | 383 (383–383) [429] | 177 (177–177) [220] |
| 32 x 5 | 135 (135–135) [204] | 756 (756–756) [800] | 976 (976–977) [1017] |
| 64 x 5 | 413 (413–413) [462] | 1634 (1633–1634) [1693] | 2130 (2130–2131) [2175] |

Per-kernel times for every cell and run are in `results/sampler/runs/`, and
`python bench/summarize.py results/sampler/runs` regenerates the tables.

### Where exactsample loses

Temperature-only sampling of up to eight requests (the 1 x 1, 8 x 1, 1 x 5
and 8 x 5 cells at T=1.0) takes 1.3 to 1.8 times vLLM's GPU time: 17 against
10 us for one request. vLLM draws with a single Gumbel kernel, while
exactsample reads each row twice (statistics, then the certified draw) in five
launches. With top-p alone and one request, FlashInfer is slightly faster: 34
against 39 us, 1.14 times less. [docs/design-tradeoffs.md](docs/design-tradeoffs.md)
explains the choice.

A request with a very large top-k (thousands) or a top-p nucleus of thousands
of tokens is finished by one block per row, and is several times slower than
the common cases above.

### End to end in vLLM

End to end, the kernel speedups shrink to a few percent at most. Sampling is
0.2 to 1.7% of a decode step with exactsample, and 0.1 to 5.8% with vLLM's
sampler. On Qwen3-1.7B and Qwen3-8B with 1 to 64 requests and no draft model,
exactsample lowers TPOT by 0.5 to 2.5% with top-k and top-p. **With
temperature alone it raises TPOT by 0.1 to 0.7%.** The runs, five per arm,
are in [integration/](integration/README.md#end-to-end-decoding).

### Batch invariance

Batch invariance covers the sampler, not the model. Without
`VLLM_BATCH_INVARIANT=1`, a request's logits, and so its tokens, depend on its
batch with either sampler. With batch-invariant kernels, vLLM's own sampler is
also invariant for seeded requests. exactsample adds unseeded requests keeping
their tokens across preemption
([integration/](integration/README.md#does-a-request-depend-on-its-batch)).

## Use it

```bash
pip install -e '.[test]'                                     # NumPy reference and tests
EXACTSAMPLE_PYTHON=python3 bash scripts/build_cuda.sh                # builds exactsample_cuda.so (PyTorch 2.13, CUDA 13, sm_86 and sm_89)
python -m pytest -q                                          # CPU tests; CUDA tests run when a GPU is present
# bench/requirements.txt pins the GPU environment the results were measured in
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
host-side parameters). The three kernels that only serve those filters are
then not launched, and the result is the same.

`docs/contract.md` defines the semantics and statuses, `docs/rng.md` the RNG
address. The vLLM integration is described in [integration/](integration/README.md).
CUDA graph capture of the call is not supported. The vLLM bridge covers greedy
draft models and ordinary sampling; other speculative methods and sampling
features fall back to vLLM's own path.

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
| `results/` | the committed results ([results/README.md](results/README.md)) |
