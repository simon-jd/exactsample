# Configurations of the end-to-end runs (sourced by bench/run_vllm_e2e.sh):
# "name|target|draft|arms|run arguments". Model directories are local Hugging
# Face snapshots given in the environment. E2E_ARMS_8B overrides the 8B arms.
configs=(
  "qwen3-1.7b|${QWEN3_1_7B:?}||off on old|--gpu-memory-utilization 0.6"
  "qwen3-1.7b-spec|${QWEN3_1_7B:?}|${QWEN3_0_6B:?}|off on old|--gpu-memory-utilization 0.6"
  "qwen3-8b|${QWEN3_8B:?}||${E2E_ARMS_8B:-off on}|--gpu-memory-utilization 0.83 --max-num-batched-tokens 4096"
)
