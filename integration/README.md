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

| Script | Checks |
|---|---|
| `vllm_smoke.py` | ordinary, seeded and JSON-constrained requests, with and without a greedy draft model; records which path ran each step |
| `vllm_error_smoke.py` | one all-masked request beside a healthy one: the failed request emits no tokens, the healthy one completes |
| `vllm_lifecycle.py` | cancellation, ID reuse, and genuine scheduler preemption under a small KV budget: keys stay constant and outputs match an isolated run |

```sh
python integration/vllm_smoke.py --target /models/Qwen3-1.7B --draft /models/Qwen3-0.6B \
  --exactsample on --output results/vllm-on.json
```

## Without an engine

`generate_transformers.py` runs a plain Hugging Face model loop in which exactsample
verifies a greedy draft model's proposals and its output tokens are the ones
appended to the context. It reruns full contexts and is for checking behavior,
not for timing.
