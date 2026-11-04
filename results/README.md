# Results

The committed result files, in the order the top-level README cites them.
"Commit" is the commit that recorded the file; where a run records the code it
measured, that commit is given as "code". Dates are the recording commit's.
Everything else under `results/` (engine logs, raw captures, scratch runs) is
ignored by `.gitignore` and stays on the measuring machine.

| File | Commit | Date | What it measured | Runs |
|---|---|---|---|---|
| `sampler/summary.json`, `sampler/runs/` | 557b69b (code edfbbab) | 2026-10-30 | GPU, call and cold-L2 time per call of exactsample, vLLM 0.31's model-runner-v2 sampler and FlashInfer 0.7.0.post1 (two entry points), 3 settings x 8 shapes, from the captured Qwen3-8B logits | 10 per implementation, interleaved (40 runs) |
| `sampler/split-check/before/`, `sampler/split-check/split/` | 1aa7ebf | 2026-11-02 | exactsample alone: the single `fused_rows` body (code 88df42c) against the split into device helpers (code 9a369f2); the single body was kept, and 1aa7ebf restores 88df42c's machine code | 10 per kernel, alternating |
| `agreement.json` | 6f77b66 | 2026-10-30 | tokens, accepted lengths and statuses against the FP64 oracle: 6 workloads, 192 requests, 352 rows from the full capture | 1 (the check is deterministic) |
| `logits/qwen3-8b-8x8.npz`, `logits/qwen3-8b-8x8.json` | 7a9716e | 2026-11-02 | the first 8 prompts x 8 steps of the capture, and the full capture's sha256; that day `bench/capture_logits.py` regenerated the 155 MB capture byte for byte | 1 |
| `agreement-8x8.json` | 7a9716e | 2026-11-02 | the agreement check on the committed 64-row subset: the same 6 workloads, 192 requests and 352 rows | 1 |
| `gumbel-bound.json` | fa1f0fd | 2026-10-30 | the FP32 Gumbel noise against its error bound, and the pruning screen, over all 2^32 Philox words | 1 (exhaustive) |
| `stat-cuda.json`, `stat-cuda-predicate.json` | 6f77b66 | 2026-10-30 | the statistical plan on the CUDA path, 10 cells, with lists on (`list_max=256`) and off (`list_max=0`) | 2M draws per cell |
| `sanitizer/summary.json` | 7d32c36 | 2026-11-04 | memcheck, racecheck, initcheck and synccheck over `tests.test_cuda` and `tests.test_fused` (23 tests each, 3 skipped), on the single `fused_rows` body restored in 1aa7ebf, with the extension rebuilt from the sources of 7d32c36; all four passed. The summary is byte-identical to the one a688602 recorded, so the file did not change; this row is the record of the rerun | 1 per tool |
| `sanitizer/summary.json`, a688602 version | a688602 | 2026-11-02 | the same four tools on the split kernel (code 9a369f2), which 1aa7ebf later reverted; all four passed | 1 per tool |
| `sanitizer/summary.json`, first version | 6f77b66 | 2026-10-30 | the same four tools on the single `fused_rows` body; all four passed (`git show 6f77b66:results/sanitizer/summary.json`) | 1 per tool |
| `integration/` | ad72fad (code d4e6295) | 2026-11-01 | the six vLLM integration checks (Qwen3-1.7B with a Qwen3-0.6B draft) and the smoke repeated with the bridge before and after the host-sync change (`integration/bridge-equivalence/`) | 1 per check |
| `e2e/timed/` | 35049b4 (code d4e6295) | 2026-11-01 | TPOT and throughput in vLLM 0.31 with vLLM's sampler (off), exactsample (on) and, on Qwen3-1.7B, the bridge from before the host-sync change (before: the `python/` tree of fae7b39); Qwen3-1.7B with and without a draft, Qwen3-8B | 5 per arm, interleaved |
| `e2e/profile/` | 35049b4 (code d4e6295) | 2026-11-01 | sampling's share of a decode step in the same cells, profiled | 1 per arm |
| `determinism/` | 17cb97f (code d4e6295) | 2026-11-01 | whether a request's tokens depend on its batch, for both samplers, with and without vLLM's batch-invariant kernels: three batch mixes with a large KV cache, preemption with a small one | 1 per sampler, mode and KV size (8 files) |

The sanitizer's per-tool logs (`<tool>.log`, `<tool>-test.log`) are not
committed; `summary.json` records each tool's command and exit status. Run
from the repository root on a GPU with the library built,
`python tests/sanitizer/run.py` reruns the four tools and writes the logs and
a new summary here.

`python bench/summarize.py results/sampler/runs` regenerates the sampler
tables in the top-level README from `sampler/runs/`.
