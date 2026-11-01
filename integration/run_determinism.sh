#!/usr/bin/env bash
# integration/vllm_determinism.py for both samplers, with and without
# VLLM_BATCH_INVARIANT, with a large and a small KV cache: eight engines. The
# four engines of each VLLM_BATCH_INVARIANT value run under one hold of the
# GPU lock. Then the summary.
#
#   EXACTSAMPLE_TARGET=<Qwen3-1.7B dir> EXACTSAMPLE_PYTHON=<env>/bin/python \
#     bash integration/run_determinism.sh results/determinism
#
# Relative paths are taken from the repository root.
set -euo pipefail
script=$(realpath "$0")
cd "$(dirname "$script")/.."
out=${1:?output directory}
target=${EXACTSAMPLE_TARGET:?local target model directory}
python=${EXACTSAMPLE_PYTHON:-python}
lock=${EXACTSAMPLE_GPU_LOCK:-/var/tmp/rtx4090.lock}
mkdir -p "$out/logs"

if [[ -n ${DETERMINISM_INVARIANT:-} ]]; then  # inside the lock
  invariant=$DETERMINISM_INVARIANT failures=0
  # The worker extension is imported as integration.vllm_determinism.
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  flags=()
  [[ $invariant == 1 ]] && flags+=(--batch-invariant)
  for kv in large small; do
    for exactsample in on off; do
      name=$([[ $exactsample == on ]] && echo exactsample || echo vllm)-bi$invariant-$kv
      [[ -e $out/$name.json ]] && { echo "skip $name (exists)"; continue; }
      echo "== $name $(date -u +%FT%TZ)"
      csv=$out/$name.gpu.csv
      echo "phase,timestamp,pstate,sm_clock,mem_clock,temperature,power,memory_used,utilization" > "$csv"
      echo "before,$(nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader)" >> "$csv"
      "$python" integration/vllm_determinism.py run --target "$target" --exactsample "$exactsample" \
        --kv "$kv" "${flags[@]}" --output "$out/$name.json" > "$out/logs/$name.log" 2>&1 \
        || { echo "FAILED: $name (log: $out/logs/$name.log)"; failures=$((failures + 1)); }
      echo "after,$(nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader)" >> "$csv"
    done
  done
  exit $((failures > 0))
fi

failures=0
for invariant in 0 1; do
  echo "== VLLM_BATCH_INVARIANT=$invariant: waiting for the GPU lock $(date -u +%FT%TZ)"
  DETERMINISM_INVARIANT=$invariant flock "$lock" bash "$script" "$out" || failures=$((failures + 1))
done
"$python" integration/vllm_determinism.py summarize "$out" --output "$out/summary.json"
exit $((failures > 0))
