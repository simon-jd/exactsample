#!/usr/bin/env bash
# Interleaved repetitions of bench/vllm_e2e.py over the configurations of a
# plan (bench/vllm_e2e_plan.sh): one engine per configuration, arm and
# repetition. A repetition runs every configuration and arm back to back, in
# an order rotated each repetition, under one hold of the GPU lock.
#
#   EXACTSAMPLE_PYTHON=<env>/bin/python QWEN3_1_7B=<dir> QWEN3_0_6B=<dir> QWEN3_8B=<dir> \
#     bash bench/run_vllm_e2e.sh <out dir> <reps> bench/vllm_e2e_plan.sh [<extra run arguments>]
#
# Arms: off = EXACTSAMPLE_DECODE=0, on = 1, old = on with the package in
# E2E_OLD_PYTHONPATH first on PYTHONPATH. Plans with an old arm need it: the
# python/ directory of a checkout of fae7b39, the bridge before the host-sync
# change (3dec1b3 changed only python/exactsample/vllm_bridge.py; run files
# record the bridge's sha256, 732c7686... for fae7b39), with
# EXACTSAMPLE_LIBRARY set to the current build's exactsample_cuda.so:
#   git worktree add .runtime/e2e-old fae7b39
#   E2E_OLD_PYTHONPATH=.runtime/e2e-old/python EXACTSAMPLE_LIBRARY=$PWD/python/exactsample/exactsample_cuda.so
# Relative paths are taken from the repository root; existing run files are
# kept, so an interrupted series resumes.
set -euo pipefail
script=$(realpath "$0")
cd "$(dirname "$script")/.."
out=$1 reps=$2 plan=$3
shift 3
python=${EXACTSAMPLE_PYTHON:-python}
lock=${EXACTSAMPLE_GPU_LOCK:-/var/tmp/rtx4090.lock}
# shellcheck source=vllm_e2e_plan.sh
source "$plan"  # defines configs: "name|target|draft|arms|arguments"
mkdir -p "$out/logs"

state() {
  echo "$1,$(nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader),$(cut -d' ' -f1 /proc/loadavg)"
}

if [[ -n ${E2E_INNER_REP:-} ]]; then  # inside the lock: one repetition of everything
  rep=$E2E_INNER_REP failures=0
  # The worker extension is imported as bench.vllm_e2e.E2EProbe.
  base="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  runs=()
  for config in "${configs[@]}"; do
    IFS='|' read -r name target draft arms arguments <<< "$config"
    read -r -a arm_list <<< "$arms"
    for ((k = 0; k < ${#arm_list[@]}; k++)); do
      runs+=("$name|$target|$draft|${arm_list[$(( (k + rep - 1) % ${#arm_list[@]} ))]}|$arguments")
    done
  done
  if (( rep % 2 == 0 )); then  # alternate the configuration order too
    for ((i = 0, j = ${#runs[@]} - 1; i < j; i++, j--)); do tmp=${runs[i]}; runs[i]=${runs[j]}; runs[j]=$tmp; done
  fi
  for item in "${runs[@]}"; do
    IFS='|' read -r name target draft arm arguments <<< "$item"
    file=$name/rep$rep-$arm
    [[ -e $out/$file.json ]] && { echo "skip $file (exists)"; continue; }
    mkdir -p "$out/$name" "$out/logs/$name"
    exactsample=on path=$base
    [[ $arm == off ]] && exactsample=off
    [[ $arm == old ]] && path=${E2E_OLD_PYTHONPATH:?package directory of the earlier bridge}:$base
    echo "== $file $(date -u +%FT%TZ)"
    csv=$out/$file.gpu.csv
    echo "phase,timestamp,pstate,sm_clock,mem_clock,temperature,power,memory_used,utilization,loadavg" > "$csv"
    state before >> "$csv"
    read -r -a extra <<< "$arguments"
    PYTHONPATH=$path "$python" bench/vllm_e2e.py run --target "$target" ${draft:+--draft "$draft"} \
      --exactsample "$exactsample" --arm "$arm" --rep "$rep" --label "$name" --output "$out/$file.json" \
      "${extra[@]}" "$@" > "$out/logs/$file.log" 2>&1 \
      || { echo "FAILED: $file (log: $out/logs/$file.log)"; failures=$((failures + 1)); }
    state after >> "$csv"
  done
  exit $((failures > 0))
fi

failures=0
for ((rep = 1; rep <= reps; rep++)); do
  echo "== repetition $rep: waiting for the GPU lock $(date -u +%FT%TZ), load $(cut -d' ' -f1 /proc/loadavg)"
  E2E_INNER_REP=$rep flock "$lock" bash "$script" "$out" "$reps" "$plan" "$@" || failures=$((failures + 1))
done
exit $((failures > 0))
