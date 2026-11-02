# Engine integration

## vLLM 0.31 (Model Runner V2)

[`vllm-patches/`](../vllm-patches/PATCHES.md) holds a 3-file, 53-line patch to
vLLM 0.31.0. It hands the raw logits and grammar masks to exactsample right after
`compute_logits`, before vLLM mutates them, and routes exactsample's per-request status
through the existing asynchronous output path, so a request whose row fails is
finished with an error without affecting the rest of the batch. The bridge
logic lives in `python/exactsample/vllm_bridge.py`, `vllm_async.py` and
`vllm_protocol.py`; `EXACTSAMPLE_DECODE=0` (the default) leaves vLLM untouched.

Request keys are the admission-time seed and stream, kept across preemption
and recomputation, so a request preempted and re-admitted draws the same tokens
it would have drawn without preemption.

The bridge does no device-to-host read per step. It decides vLLM's
prefill-placeholder fallback from the host's `is_prefilling` flags, gathers
all drafts with one indexed copy, and sends the step's parameters, seeds and
draft indices in one non-blocking copy from a pinned three-slot ring. A slot
is reused only after the event recorded behind its last copy.

| Script | Checks |
|---|---|
| `vllm_smoke.py` | ordinary, seeded and JSON-constrained requests, with and without a greedy draft model; records which path ran each step |
| `vllm_error_smoke.py` | one all-masked request beside a healthy one: the failed request emits no tokens, the healthy one completes |
| `vllm_lifecycle.py` | cancellation, ID reuse, and genuine scheduler preemption under a small KV budget: keys stay constant and outputs match an isolated run |
| `vllm_determinism.py` | the same seeded requests alone, in mixed batches and under preemption, for both samplers; attributes each token difference to the logits or the sampler |
| `../bench/vllm_e2e.py` | throughput, TPOT and per-step timing with exactsample on and off |

```sh
EXACTSAMPLE_TARGET=<Qwen3-1.7B dir> EXACTSAMPLE_DRAFT=<Qwen3-0.6B dir> EXACTSAMPLE_PYTHON=<env>/bin/python \
  bash integration/run_checks.sh results/integration
```

All runs below used Qwen3-1.7B, Qwen3-0.6B and Qwen3-8B (bf16) on one RTX 4090.
The software was vLLM 0.31.0 with the patch applied by `vllm-patches/apply.py`
to a fresh environment (`bench/requirements.txt`), torch 2.13 and CUDA 13.
Each run records the GPU state before and after it, and its commit.

### Integration checks

[`results/integration/`](../results/integration/) (Qwen3-1.7B with a Qwen3-0.6B
draft) holds all six checks, and all pass:

- **Smoke, exactsample on, with the draft.** 21 steps went through exactsample,
  20 of them verifying drafts and 4 with grammar masks, and none fell back.
  The JSON-constrained request returned `{"answer": 7}`.
- **Smoke, exactsample on, without the draft.** 33 steps, 7 of them with
  grammar masks.
- **Smoke, exactsample off.** vLLM's own path passes with the patch applied.
- **Error isolation.** In both injections the all-masked request finished
  with `error` and `stop_reason` `exactsample_status:1`, and the healthy
  request completed its 12 tokens. The failed request committed no token in
  its failing step: 0 tokens in total in the ordinary case, and in the draft
  case only the 1 it had committed in an earlier step.
- **Lifecycle.** The scheduler preempted and re-admitted two requests. vLLM's
  own per-slot seed changed at re-admission while the bridge's key did not.
  Their 96 tokens equal an isolated run with the admission seed. Cancellation
  and external-ID reuse also behaved as expected.

`results/integration/bridge-equivalence/` repeats the smoke with
`VLLM_BATCH_INVARIANT=1`, once with the current bridge and once with the
bridge from before the host-sync change. Every request's tokens are identical
between the two. With the draft, the per-step trace (tokens, accepted lengths,
statuses, seeds, positions) is identical in all 21 steps.

### End-to-end decoding

`bench/run_vllm_e2e.sh results/e2e/timed 5 bench/vllm_e2e_plan.sh` serves the
32 chat prompts the kernel benchmark's logits came from, with one seed per
request. Two settings are measured: T=0.6, top-k 20, top-p 0.95, and T=1.0
alone. Each cell has 1, 16 or 64 requests generating 256 tokens each, all
admitted to the same step. Each arm runs in its own engine:

- **off**: vLLM's sampler (`EXACTSAMPLE_DECODE=0`).
- **on**: exactsample.
- **before**: exactsample with the bridge from before the host-sync change.

The engines use CUDA graphs, torch.compile and async scheduling. Each arm has
five interleaved repetitions. The tables give the median TPOT in ms, with the
range in brackets.

Qwen3-1.7B:

| setting | requests | off | on | before |
|---|---:|---:|---:|---:|
| T=0.6, top-k, top-p | 1 | 4.44 [4.44, 4.45] | **4.36** [4.36, 4.36] | 5.03 [4.98, 5.10] |
| | 16 | 5.30 [5.28, 5.48] | **5.19** [5.18, 5.26] | 5.99 [5.97, 6.11] |
| | 64 | 6.39 [6.38, 6.56] | **6.23** [6.23, 6.38] | 7.42 [7.35, 7.60] |
| T=1.0 | 1 | **4.32** [4.32, 4.32] | 4.34 [4.33, 4.44] | 5.11 [4.99, 5.57] |
| | 16 | **5.15** [5.14, 5.20] | 5.18 [5.17, 5.20] | 6.04 [6.00, 6.43] |
| | 64 | **6.19** [6.19, 6.25] | 6.24 [6.24, 6.66] | 7.50 [7.44, 7.62] |

Qwen3-8B:

| setting | requests | off | on |
|---|---:|---:|---:|
| T=0.6, top-k, top-p | 1 | 16.94 [16.93, 16.95] | **16.86** [16.85, 16.87] |
| | 16 | 18.18 [18.18, 18.19] | **18.08** [18.08, 18.08] |
| | 64 | 20.95 [20.95, 20.96] | **20.80** [20.79, 20.81] |
| T=1.0 | 1 | **16.82** [16.82, 16.83] | 16.84 [16.84, 16.84] |
| | 16 | **18.05** [18.04, 18.06] | 18.07 [18.07, 18.08] |
| | 64 | **20.76** [20.76, 20.78] | 20.81 [20.81, 20.82] |

With top-k and top-p, exactsample lowers TPOT by 1.8 to 2.5% on Qwen3-1.7B. At
64 requests output throughput rises from 9,852 to 10,093 tokens/s. On Qwen3-8B
the reduction is 0.5 to 0.7%. **With temperature alone, decoding is slower
with exactsample**: by 0.5 to 0.8% on Qwen3-1.7B (10,167 against 10,091
tokens/s at 64 requests) and by 0.1 to 0.2% on Qwen3-8B. These are the cases
where vLLM's one-launch Gumbel kernel is faster on GPU time. The bridge from
before the host-sync change made every Qwen3-1.7B cell 13 to 21% slower
than vLLM's sampler.

**With the draft model** (Qwen3-0.6B, 4 greedy drafts), each engine step takes
26.3 against 26.5 ms at 64 requests with top-k/top-p, and 26.5 against 26.2 ms
at T=1.0 (exactsample against vLLM). In the other cells the two
ranges overlap. The ranges are wide: one repetition's c=1 cell averaged 29.0 ms
per step. The earlier bridge took 25.6 to 33.6 ms per step, 21 to 29% more. TPOT also depends on how many drafts
are accepted, and that differs between the two samplers because they sample
different sequences: 2.45 against 2.71 tokens per step at c=1 with top-k/top-p,
for example. Both accept a one-hot draft with probability p(draft), so those
differences come from the sampled text, not the method. The time per step is
the comparable measure (`results/e2e/timed/summary.json`).

**Bridge host time.** `results/e2e/profile/` is one pass with CUDA events and
host timers around the runner's sampling step. Medians per decode step:

| | exactsample call on the host, before | after | of which `verify()` | GPU time of sampling, off | on | share of step, off | on |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B, top-k/top-p, 1 to 64 requests | 4.4 to 6.2 ms | 256 to 298 µs | 56 to 62 µs | 0.14 to 0.23 ms | 0.06 to 0.10 ms | 3.0 to 3.5% | 1.3 to 1.7% |
| Qwen3-1.7B, T=1.0 | 4.4 to 6.3 ms | 284 to 311 µs | 62 to 68 µs | 0.02 to 0.05 ms | 0.04 to 0.10 ms | 0.5 to 0.8% | 0.8 to 1.5% |
| Qwen3-1.7B with the draft | 4.9 to 11.3 ms | 385 to 451 µs | 72 to 82 µs | 0.03 to 1.56 ms | 0.05 to 0.41 ms | 0.1 to 5.8% | 0.3 to 1.6% |
| Qwen3-8B | 16.9 to 20.9 ms | 370 to 532 µs | 84 to 109 µs | 0.02 to 0.23 ms | 0.04 to 0.10 ms | 0.1 to 1.1% | 0.2 to 0.5% |

Before the change, the host time per step was about the GPU's step time. Five
parameter copies from pageable memory each waited for the stream, which meant
waiting for the forward pass, and with drafts each request added an
`.item()`. Now the host waits 2.5 to 4.3 µs per step for the pinned slot's
event. The sampling stage is measured from the end of the logits computation
to the end of sampling, as a share of the GPU period between steps.

### Does a request depend on its batch?

`integration/run_determinism.sh results/determinism` serves six seeded probes,
two per sampling setting, on chat prompts of Qwen3-1.7B, 64 tokens each. They
run alone, among 15 or 31 other requests (first or last, so in other slots),
and joining others mid-generation. Two unseeded probes join the preemption
case, which uses a small KV cache. There vLLM's scheduler preempted and
recomputed four of the eight probes: two seeded, and both unseeded. A hook hashes every probe's logits row before any sampling
code runs. A token difference is therefore attributed to the model when that
row already differed from the alone run, and to the sampler when it was
identical. With `VLLM_BATCH_INVARIANT=1`, the count of probes with the same
tokens as alone:

| case | exactsample | vLLM |
|---|---:|---:|
| 15 others, probes first | 6/6 | 6/6 |
| 31 others, probes last | 6/6 | 6/6 |
| joining mid-generation | 6/6 | 6/6 |
| small KV cache, 4 of 8 preempted (6 seeded, 2 unseeded) | 8/8 | 6/8 |

- **Without batch-invariant kernels**, no probe's logits match its alone run
  in any mixed case, for either sampler, so those runs measure the model's
  batch dependence (noise in the logits), not the samplers; their token
  counts, from one run of 6 to 8 probes per case, are in
  `results/determinism/summary.json`. Every token difference there is
  attributed to the logits. exactsample's sampling cannot remove differences
  that the model's batch-dependent logits create.
- **With `VLLM_BATCH_INVARIANT=1`**, every probe's logits match bitwise in
  every case, and both samplers reproduce all seeded probes exactly, including
  after preemption. vLLM's sampler addresses seeded Gumbel noise by seed and
  position too, so for seeded requests the two are equally batch-invariant.
- **The difference is in unseeded requests.** vLLM draws a new random seed
  each time it re-admits a preempted request (900450186894289455, then
  1895649597198586619, for one probe). Its two unseeded probes left the
  alone-with-admission-seed run at tokens 59 and 42, with identical logits.
  The bridge keeps the admission seed, and exactsample matched for both.
  vLLM also samples unseeded top-k/top-p batches that contain no seeded or
  greedy request with FlashInfer's kernel and the process-wide CUDA
  generator (`vllm/v1/worker/gpu/sample/sampler.py`, `flashinfer_sample`), so
  those draws depend on the batch and on call history. Those batches are not
  measured here.

The end-to-end runs show the same thing across processes. With admission
controlled, exactsample's tokens were identical in all five repetitions for
every request in every cell. vLLM's sampler gave different tokens in some
repetition in 11 of its 18 cells, for up to 49 of 64 requests (Qwen3-8B,
T=1.0). Those runs record token hashes, not logits, so they do not establish
the cause. In the determinism runs, every probe served alone had identical
logits and tokens in both engines of each sampler and setting.

## Without an engine

`generate_transformers.py` runs a plain Hugging Face model loop in which exactsample
verifies a greedy draft model's proposals and its output tokens are the ones
appended to the context. It reruns full contexts and is for checking behavior,
not for timing.
