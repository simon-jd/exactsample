# Cross-implementation conformance

Adapters that call the real exactsample, FlashInfer and vLLM APIs on the same inputs.
They never substitute NumPy or another library for a missing implementation;
an unavailable library is reported as skipped (`--require-all` makes it fail).
Each library keeps its own RNG, so these compare distributions and contract
behavior, not token identity.

| Command | What it reports |
|---|---|
| `python -m conformance.run --output OUT.json` | ordinary sampling on small reproducers: token and status counts, input hashes, library versions |
| `python -m conformance.speculative --backend {exactsample_cuda,flashinfer,vllm_mrv2} --output OUT.json` | padded full-q verification with hand-computed p and q: first-token and accepted-prefix counts |
| `python -m conformance.invariance --backend BACKEND --output OUT.json` | per-library differences under reversed order, single-request runs and replay with each request's native seed |
| `python -m tests.stat.run --backend {reference,cuda,flashinfer,vllm_mrv2} --output OUT.json` | the preregistered statistical plan (`tests/stat_plan.yaml`) |

`--flashinfer-version` pins a release; `--flashinfer-commit FULL_SHA` requires
an imported clean checkout at that commit (set a separate
`FLASHINFER_WORKSPACE_BASE`, since FlashInfer's JIT cache key does not include
the commit).

Every external library differs from exactsample in which tied tokens it keeps when a
top-p boundary falls inside a run of equal probabilities (the `topp_boundary`
cell of the statistical plan): exactsample keeps exactly the shortest prefix, splitting
the ties by token ID.

Source signatures were checked against
[FlashInfer sampling.py v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/flashinfer/sampling.py),
[vLLM gumbel.py v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu/sample/gumbel.py) and
[vLLM rejection_sampler_utils.py v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py).
