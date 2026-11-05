"""exactsample's sampler benchmark on one GPU of a Snowflake compute pool, against vLLM 0.31's model-runner-v2
sampler and FlashInfer 0.7.0.post1 (two entry points), on the 512-row Qwen3-8B capture.

The procedure of bench/run_sampler.sh: each run is one fresh bench/sampler.py process per implementation,
interleaved, with the GPU state recorded before each. Differences: --no-cupti (CUPTI records no kernels on the
A10 vGPU, so only the event times call_us and cold_us exist), one unrecorded warm-up round first (it fills
the FlashInfer and Triton JIT caches, which on the RTX 4090 host already existed), and no new round starts
after PF_BENCH_MINUTES (default 17) so that the job ends inside its time limit; PF_MAX_RUNS (default 10)
caps the rounds. FlashInfer's JIT and Triton use nvcc and ptxas from cuda-13.2-nvcc.tar, a copy of the
compiler, headers and runtime libraries of the CUDA 13.2 toolkit the RTX 4090 host used (the venv's nvcc 13.4
wheel does not match its 13.0 runtime headers). Inputs as in exactsample_a10_check.py, plus the ninja wheel. Prints sampler-runs.tar.gz base64-encoded between FILE/END markers, then one RESULT line.
PF_LOCAL_ASSETS=<dir> PF_WORK=<dir on disk>: local dry run.
"""
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

WORK = os.environ.get("PF_WORK", "/tmp/pf-exactsample-sampler")
LOCAL = os.environ.get("PF_LOCAL_ASSETS")
BENCH_MINUTES = float(os.environ.get("PF_BENCH_MINUTES", "17"))
MAX_RUNS = int(os.environ.get("PF_MAX_RUNS", "10"))
NINJA = "ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
NAMES = ["wheelhouse-vllm-0.31.0.tar", "cuda-compat-13-2.tar", "exactsample-d5322a9.tar",
         "exactsample-qwen3-8b-logits.tar", NINJA, "cuda-13.2-nvcc.tar"]
IMPLS = ["exactsample", "vllm_mrv2", "flashinfer:logits", "flashinfer:probs"]
VENV = os.path.join(WORK, "venv")
T0 = time.time()

TORCH_CHECK = r"""
import json, torch
r = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
r["libcuda"] = sorted({x.split()[-1] for x in open("/proc/self/maps") if "libcuda.so" in x})
if r["available"]:
    r["device"] = torch.cuda.get_device_name(0); r["cc"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(r))
"""

VERSIONS = r"""
import importlib.metadata as m, json
v = {}
for name in ("torch", "vllm", "flashinfer-python", "triton", "nvidia-cuda-nvcc", "ninja"):
    try:
        v[name] = m.version(name)
    except m.PackageNotFoundError:
        pass
print(json.dumps(v))
"""


def run(cmd, env=None, timeout=900, cwd=None):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr, "seconds": round(time.time() - t, 1)}
    except Exception as e:
        return {"rc": None, "out": "", "err": repr(e)[:500], "seconds": round(time.time() - t, 1)}


def last_json(res):
    for line in reversed((res.get("out") or "").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return {"rc": res.get("rc"), "err_tail": (res.get("err") or "").strip().splitlines()[-8:]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def untar(path, dest):
    with tarfile.open(path) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:
            t.extractall(dest)


def emit(name, data):
    text = base64.b64encode(data).decode()
    print(f"=== FILE {name} {len(data)} {hashlib.sha256(data).hexdigest()} ===", flush=True)
    for i in range(0, len(text), 76):
        print(text[i:i + 76])
    print(f"=== END {name} ===", flush=True)


def smi(query):
    r = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], timeout=60)
    return (r["out"] or r["err"]).strip()


out = {"python": sys.version.split()[0]}
try:
    os.makedirs(WORK, exist_ok=True)
    t = time.time()
    if LOCAL:
        for n in NAMES + ["SHA256SUMS"]:
            shutil.copy(os.path.join(LOCAL, n), WORK)
    else:
        from snowflake.snowpark import Session
        s = Session.builder.getOrCreate()
        for n in NAMES + ["SHA256SUMS"]:
            s.file.get(f"@PORTFOLIO_LAB.GPU.ASSETS/{n}", WORK)
    out["fetch_s"] = round(time.time() - t, 1)
    want = dict(reversed(line.split()) for line in open(os.path.join(WORK, "SHA256SUMS")) if line.strip())
    out["sha256_mismatch"] = [n for n in NAMES if sha256(os.path.join(WORK, n)) != want.get(n)]
    wheels, compat = os.path.join(WORK, "wheels"), os.path.join(WORK, "compat")
    for d, n in ((wheels, NAMES[0]), (compat, NAMES[1]), (WORK, NAMES[2]), (WORK, NAMES[3]), (WORK, NAMES[5])):
        os.makedirs(d, exist_ok=True)
        untar(os.path.join(WORK, n), d)
        os.remove(os.path.join(WORK, n))
    shutil.move(os.path.join(WORK, NINJA), os.path.join(wheels, NINJA))
    t = time.time()
    run([sys.executable, "-m", "venv", VENV], timeout=300)
    whls = sorted(os.path.join(wheels, f) for f in os.listdir(wheels) if f.endswith(".whl"))
    r = run([os.path.join(VENV, "bin", "pip"), "install", "-q", "--no-index", "--no-deps", *whls], timeout=1800)
    shutil.rmtree(wheels)
    out["install"] = {"rc": r["rc"], "wheels": len(whls), "seconds": round(time.time() - t, 1),
                      "err_tail": (r.get("err") or "").strip().splitlines()[-4:]}
    py = os.path.join(VENV, "bin", "python")
    cuda_home = os.path.join(WORK, "cuda-13.2")
    repo = os.path.join(WORK, "exactsample")
    commit = open(os.path.join(repo, "COMMIT")).read().strip()
    base = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                XDG_CACHE_HOME=os.path.join(WORK, "cache"), FLASHINFER_WORKSPACE_BASE=os.path.join(WORK, "fi"),
                TRITON_CACHE_DIR=os.path.join(WORK, "triton"), CUDA_HOME=cuda_home,
                TRITON_PTXAS_PATH=os.path.join(cuda_home, "bin", "ptxas"),
                PATH=os.path.join(VENV, "bin") + ":" + os.path.join(cuda_home, "bin") + ":" + os.environ["PATH"])
    base.pop("PYTHONPATH", None)
    node = last_json(run([py, "-c", TORCH_CHECK], env=base, timeout=300))
    env = base
    if not node.get("available"):
        env = dict(base, LD_LIBRARY_PATH=compat + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        out["torch_compat"] = last_json(run([py, "-c", TORCH_CHECK], env=env, timeout=300))
    out["torch_node_driver"] = node
    out["compat_driver_used"] = env is not base
    nvcc = run([os.path.join(cuda_home, "bin", "nvcc"), "--version"], env=env, timeout=60)
    ptxas = run([env["TRITON_PTXAS_PATH"], "--version"], env=env, timeout=60)

    runs_dir = os.path.join(WORK, "runs")
    os.makedirs(os.path.join(runs_dir, "warmup"), exist_ok=True)
    gpu = smi("name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.max,pcie.link.width.max")
    json.dump({"gpu": gpu, "cuda": node.get("cuda"), "versions": last_json(run([py, "-c", VERSIONS], env=env)),
               "nvcc": nvcc["out"].strip().splitlines()[-1:], "ptxas": ptxas["out"].strip().splitlines()[-1:],
               "git_commit": commit, "git_dirty": False, "clocks_locked": False, "cupti": False,
               "compat_driver": out["compat_driver_used"], "warmup_rounds": 1},
              open(os.path.join(runs_dir, "env.json"), "w"), indent=1)
    clocks = open(os.path.join(runs_dir, "clocks.csv"), "w")
    clocks.write("run,impl,sm_mhz,mem_mhz,temp_c,power_w,util_pct,mem_used_mib\n")
    capture = os.path.join(WORK, "logits", "qwen3-8b.npz")
    failures, rounds, round_s = [], 0, []
    for number in range(0, MAX_RUNS + 1):
        if number > 0:
            elapsed = (time.time() - T0) / 60
            needed = (max(round_s[1:]) if len(round_s) > 1 else round_s[0]) / 60
            if elapsed + needed > BENCH_MINUTES:
                break
        t = time.time()
        for impl in IMPLS:
            target = os.path.join(runs_dir, "warmup") if number == 0 else runs_dir
            clocks.write(f"{number},{impl},{smi('clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used')}\n")
            clocks.flush()
            r = run([py, "bench/sampler.py", "--impl", impl, "--run", str(number), "--no-cupti", "--capture", capture,
                     "--output", os.path.join(target, f"{impl.replace(':', '-')}-{number}.jsonl")],
                    env=env, cwd=repo, timeout=900)
            if r["rc"] != 0:
                failures.append({"run": number, "impl": impl, "rc": r["rc"], "seconds": r["seconds"],
                                 "err_tail": r["err"].strip().splitlines()[-12:]})
        round_s.append(time.time() - t)
        rounds = number
        print(f"round {number}: {round_s[-1]:.0f} s", flush=True)
        if number == 0 and len(failures) == len(IMPLS):
            break
    clocks.close()
    out["rounds"] = rounds
    out["round_seconds"] = [round(x) for x in round_s]
    out["failures"] = failures
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(runs_dir, arcname="runs")
    emit("sampler-runs.tar.gz", buf.getvalue())
except Exception as e:
    out["error"] = repr(e)[:800]
out["payload_seconds"] = round(time.time() - T0, 1)
print("RESULT " + json.dumps(out), flush=True)
