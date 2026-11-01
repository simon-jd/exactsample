# vLLM v0.31.0 local fork patch

This is a local integration patch, not an upstream contribution. Its three base
files are the exact installed 0.31.0 wheel sources; `model_runner.py` additionally
matches the downloaded tag `v0.31.0` source byte for byte. `manifest.json` records
their before/after SHA-256 hashes and the
patch hash. The patch is real `git format-patch` output from a temporary source
repository. That repository's import commit is local, not the upstream tag's
commit ID; the source-file hashes establish the base used here.

| File | Functions | Changed lines | Purpose |
|---|---|---:|---|
| `vllm/v1/worker/gpu/model_runner.py` | `load_model`, `add_requests`, `finish_requests`, `sample`, `sample_tokens` | 29 additions, 3 deletions | Reserve exactsample workspace; track admission keys across preemption; hand raw logits and grammar masks to exactsample before native processing |
| `vllm/v1/engine/parallel_sampling.py` | `ParentRequest.get_child_info` | 6 additions, 1 deletion | Mark expanded n>1 children before n becomes 1, so the bridge correctly retains the native path |
| `vllm/v1/core/sched/scheduler.py` | `update_from_output` | 14 additions | Complete only failed requests as `FINISHED_ERROR`, after stale-output checks and before token commit |

The patch changes **3 files / 53 lines**.
All substantial integration logic is owned by this repository in
`python/exactsample/vllm_bridge.py`, `vllm_async.py`, and `vllm_protocol.py`.

## Why the hook moved

The natural hooks would be `Sampler._sample_random` and
`RejectionSampler._verify`. Inspection of the pinned source showed that
`GPUModelRunner.sample` applies grammar masks before either sampler, while
`Sampler.sample` also applies parameters before `_sample_random`. Those hooks
cannot recover original raw logits. The implemented shared hook sits immediately
after `compute_logits` and before grammar mutation. On success its
`SamplerOutput` follows the existing engine postprocessing and token-submission
path. On fallback the original grammar and sampling branches execute unchanged.

## Apply and disable

With a clean checkout of the tag:

```sh
python vllm-patches/apply.py --root /path/to/vllm-checkout --check
python vllm-patches/apply.py --root /path/to/vllm-checkout
```

For an isolated test wheel, `--root` can be its `site-packages` directory.
The script verifies exact source hashes before mutation, runs `git apply
--check`, applies the patch, and verifies exact resulting hashes. It rejects
other versions or unrelated edits. `--reverse` restores the verified original.
An upstream Git checkout can alternatively use `git am` with the patch file.

`EXACTSAMPLE_DECODE=0` (or unset) disables every bridge call and allocation. exactsample is not
imported, and the original grammar and sampling code runs with the same inputs.

With `EXACTSAMPLE_DECODE=1`, also expose the installed exactsample package and built library:

```sh
export EXACTSAMPLE_LIBRARY=/path/to/exactsample_cuda.so
export VLLM_USE_V2_MODEL_RUNNER=1
export EXACTSAMPLE_MODE=coupled
```

`EXACTSAMPLE_MODE` accepts `coupled` or `ratio`; `EXACTSAMPLE_DISPATCH` accepts `auto`, `reference`,
or `fused`. Optional `EXACTSAMPLE_TRACE_PATH` writes a per-step JSONL trace (seeds,
positions, grammar rows, statuses, tokens, kernel path) and `EXACTSAMPLE_CAPTURE_DIR`
saves replayable raw inputs (`EXACTSAMPLE_CAPTURE_LIMIT` steps, default 8). Both add
copies and I/O; leave them off when timing.

## Supported requests and fallback

The bridge supports single-device MRv2, ordinary sampling and standard
rejection with a greedy `draft_model`, one target bonus row, n=1,
temperature/min-p/top-k/top-p, and the per-position grammar masks vLLM already
produces. Every request uses its admission-time vLLM seed and stream 0;
scheduling slots never enter RNG addresses. The bridge caches the seed by
request ID across preemption and re-addition (vLLM otherwise reassigns
seed=None) and clears it only on completion or cancellation. Positions come
from the actual source logit rows; proposals are gathered from local rows 1..D.

Any unsupported request sends the **entire batch** through the original path:
penalties, bias, bad words, allowed-token processors, min_tokens, logprobs,
thinking budgets, custom processors or samplers, n>1, watermarking,
adaptive/block/synthetic verification, probabilistic (full-q) drafts,
heterogeneous vocabularies, CUDA graph capture of the sampler, or several
devices. Full-q support in the library does not imply it in the engine: the
bridge does not guess the distribution an unfamiliar draft path used.

## Request errors

Numerical status is copied with the token outputs on the existing copy stream
and completion event, so there is no extra host wait. A typed
`ExactSampleModelRunnerOutput` carries the status per request ID to the scheduler,
which (after its stale and aborted checks) finishes each failed request through
its existing `FINISHED_ERROR` path with `stop_reason="exactsample_status:<code>"`. A
failed request emits no tokens in that step; the others keep theirs. Malformed
bridge metadata raises before the step is submitted.

The bridge still gathers metadata, copies parameters and masks, and clones
outputs for their asynchronous lifetime. It does so without a host wait, in
about 0.3 to 0.5 ms of host time per step. The end-to-end measurements in
[integration/](../integration/README.md#end-to-end-decoding) include these
costs.
