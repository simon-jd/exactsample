#!/usr/bin/env bash
# Run the vLLM integration checks against the patched engine, all under one
# hold of the GPU lock (a few minutes), recording the GPU state before and
# after each check.
#
#   EXACTSAMPLE_TARGET=<Qwen3-1.7B dir> EXACTSAMPLE_DRAFT=<Qwen3-0.6B dir> \
#   EXACTSAMPLE_PYTHON=<env>/bin/python bash integration/run_checks.sh results/integration
#
# EXACTSAMPLE_CHECKS selects a subset (names below), EXACTSAMPLE_SMOKE_ARGS adds
# arguments to the smoke runs (e.g. --batch-invariant); logs go to
# EXACTSAMPLE_LOG_DIR (default <output>/logs). Relative paths are taken from
# the repository root.
set -euo pipefail
script=$(realpath "$0")
cd "$(dirname "$script")/.."
out=${1:?output directory}
target=${EXACTSAMPLE_TARGET:?local target model directory}
draft=${EXACTSAMPLE_DRAFT:?local draft model directory}
python=${EXACTSAMPLE_PYTHON:-python}
lock=${EXACTSAMPLE_GPU_LOCK:-/var/tmp/rtx4090.lock}
if [[ -z ${CHECKS_LOCKED:-} ]]; then
  echo "waiting for the GPU lock $(date -u +%FT%TZ)"
  CHECKS_LOCKED=1 exec flock "$lock" bash "$script" "$@"
fi
logs=${EXACTSAMPLE_LOG_DIR:-$out/logs}
checks=${EXACTSAMPLE_CHECKS:-smoke-on-spec smoke-on-nospec smoke-off-spec error-prefill error-spec lifecycle}
mkdir -p "$out" "$logs"
# The worker extension is imported as integration.vllm_worker_probe.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$python" - "$out/provenance.json" "$target" "$draft" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
import torch, vllm
import exactsample.vllm_bridge as bridge
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
runner = Path(vllm.__file__).parent / "v1/worker/gpu/model_runner.py"
record = {"commit": git("rev-parse", "HEAD"), "worktree_clean": git("status", "--porcelain") == "",
          "bridge_file": bridge.__file__, "bridge_sha256": sha(bridge.__file__),
          "vllm": vllm.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
          "model_runner_sha256": sha(runner), "gpu": torch.cuda.get_device_name(0),
          "target": sys.argv[2], "draft": sys.argv[3]}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
PY

state() {
  echo "$1,$(nvidia-smi --query-gpu=timestamp,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader)"
}
run() {
  local name=$1; shift
  case " $checks " in *" $name "*) ;; *) return 0 ;; esac
  echo "== $name $(date -u +%FT%TZ)"
  local csv=$out/$name.gpu.csv
  echo "phase,timestamp,name,driver,pstate,sm_clock,mem_clock,temperature,power,memory_used,utilization" > "$csv"
  state before >> "$csv"
  "$@" > "$logs/$name.log" 2>&1 || { echo "FAILED: $name (log: $logs/$name.log)"; failures=$((failures + 1)); }
  state after >> "$csv"
}
failures=0

common=(--target "$target" --draft "$draft")
small=(--gpu-memory-utilization .28 --kv-cache-memory-bytes 1073741824)
read -r -a smoke_extra <<< "${EXACTSAMPLE_SMOKE_ARGS:-}"
run smoke-on-spec "$python" integration/vllm_smoke.py "${common[@]}" "${small[@]}" "${smoke_extra[@]}" --exactsample on \
  --max-tokens 32 --output "$out/smoke-on-spec.json"
run smoke-on-nospec "$python" integration/vllm_smoke.py "${common[@]}" "${small[@]}" "${smoke_extra[@]}" --exactsample on \
  --no-spec --max-tokens 32 --output "$out/smoke-on-nospec.json"
run smoke-off-spec "$python" integration/vllm_smoke.py "${common[@]}" "${small[@]}" "${smoke_extra[@]}" --exactsample off \
  --max-tokens 32 --output "$out/smoke-off-spec.json"
run error-prefill "$python" integration/vllm_error_smoke.py "${common[@]}" "${small[@]}" --min-draft 0 \
  --output "$out/error-prefill.json"
run error-spec "$python" integration/vllm_error_smoke.py "${common[@]}" "${small[@]}" --min-draft 1 \
  --output "$out/error-spec.json"
run lifecycle "$python" integration/vllm_lifecycle.py "${common[@]}" --output "$out/lifecycle.json"

# The scripts record absolute paths; keep model IDs and relative paths instead.
"$python" - "$out" "$target" "$draft" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from bench.vllm_e2e import model_label
out, target, draft = sys.argv[1:]
pairs = [(str(Path(target).resolve()), model_label(target)), (str(Path(draft).resolve()), model_label(draft)),
         (str(Path.cwd()), "."), (str(Path.home()), "~")]
for path in [*Path(out).glob("*.json"), *Path(out).glob("*.jsonl")]:
    text = path.read_text()
    for before, after in pairs:
        text = text.replace(before, after)
    path.write_text(text)
PY
exit $((failures > 0))
