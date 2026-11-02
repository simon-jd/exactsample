#!/usr/bin/env bash
# Runs bench/sampler.py for every implementation RUNS times (default 10),
# interleaved so that drift in clocks or background load spreads over all of
# them. Each run is a fresh process holding the machine-wide GPU lock.
# Writes OUT/<impl>-<run>.jsonl, the GPU state before each run (clocks.csv)
# and the environment (env.json); bench/summarize.py OUT reports the median
# and interquartile range over runs.
#
#   bash bench/run_sampler.sh results/sampler/<name>    (results/sampler/runs is the committed series)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: run_sampler.sh OUTPUT_DIR}
PYTHON="${EXACTSAMPLE_PYTHON:-python3}"
RUNS="${RUNS:-10}"
IMPLS="${IMPLS:-exactsample vllm_mrv2 flashinfer:logits flashinfer:probs}"
LOCK="${EXACTSAMPLE_GPU_LOCK:-/var/tmp/rtx4090.lock}"
# Triton's bundled ptxas is not executable in some installs; use the toolkit's.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$(command -v ptxas)}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "refusing to write into non-empty $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
"$PYTHON" - "$OUT/env.json" <<'PY'
import importlib.metadata as m, json, subprocess, sys, torch
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,"
                      "power.limit,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True).stdout.strip())
versions = {}
for name in ("torch", "vllm", "flashinfer-python", "triton"):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        pass
json.dump({"gpu": smi, "cuda": torch.version.cuda, "versions": versions, "git_commit": commit,
           "git_dirty": dirty, "clocks_locked": False}, open(sys.argv[1], "w"), indent=1)
PY
echo "run,impl,sm_mhz,mem_mhz,temp_c,power_w,util_pct,mem_used_mib" > "$OUT/clocks.csv"
for run in $(seq 1 "$RUNS"); do
  for impl in $IMPLS; do
    flock "$LOCK" bash -c '
      echo "$1,$2,$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits)" >> "$3/clocks.csv"
      "$4" bench/sampler.py --impl "$2" --run "$1" --output "$3/${2/:/-}-$1.jsonl" > /dev/null' _ "$run" "$impl" "$OUT" "$PYTHON"
  done
done
