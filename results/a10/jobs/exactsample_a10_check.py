"""exactsample on one GPU of a Snowflake compute pool: the test suite with the CUDA tests required, the
agreement check on both Qwen3-8B captures with the raw outputs saved, and the exhaustive Gumbel bound check.

Inputs (checked against SHA256SUMS): wheelhouse-vllm-0.31.0.tar (the frozen venv, installed with --no-index
--no-deps), cuda-compat-13-2.tar (put on LD_LIBRARY_PATH only when torch cannot use the node's driver),
exactsample-d5322a9.tar (git archive of d5322a9 plus exactsample_cuda.so and gumbel_bound, both built for
sm_86 and sm_89 on the RTX 4090 host), exactsample-testdeps-py312.tar (pytest and scipy from that host's GPU
env) and exactsample-qwen3-8b-logits.tar (the 512-row capture). Every output file is printed base64-encoded
between FILE/END markers, then one RESULT line. PF_LOCAL_ASSETS=<dir> PF_WORK=<dir on disk>: local dry run.
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

WORK = os.environ.get("PF_WORK", "/tmp/pf-exactsample-check")
LOCAL = os.environ.get("PF_LOCAL_ASSETS")
NAMES = ["wheelhouse-vllm-0.31.0.tar", "cuda-compat-13-2.tar", "exactsample-d5322a9.tar",
         "exactsample-testdeps-py312.tar", "exactsample-qwen3-8b-logits.tar"]
VENV = os.path.join(WORK, "venv")
T0 = time.time()

TORCH_CHECK = r"""
import json, torch
r = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
r["libcuda"] = sorted({x.split()[-1] for x in open("/proc/self/maps") if "libcuda.so" in x})
if r["available"]:
    r["device"] = torch.cuda.get_device_name(0); r["cc"] = list(torch.cuda.get_device_capability(0))
    p = torch.cuda.get_device_properties(0)
    r["sms"] = p.multi_processor_count; r["memory_gib"] = round(p.total_memory / 2**30, 1)
    r["driver_cuda"] = torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None
else:
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        r["error"] = str(e).splitlines()[0][:300]
print(json.dumps(r))
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
    if isinstance(data, str):
        data = data.encode()
    text = base64.b64encode(data).decode()
    print(f"=== FILE {name} {len(data)} {hashlib.sha256(data).hexdigest()} ===", flush=True)
    for i in range(0, len(text), 76):
        print(text[i:i + 76])
    print(f"=== END {name} ===", flush=True)


def emit_file(path, name=None):
    if os.path.exists(path):
        emit(name or os.path.basename(path), open(path, "rb").read())
        return True
    return False


def smi():
    r = run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,"
             "temperature.gpu,power.draw,power.limit,memory.total,memory.used", "--format=csv,noheader"], timeout=60)
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
    out["sha256"] = {n: sha256(os.path.join(WORK, n)) for n in NAMES}
    out["sha256_mismatch"] = [n for n in NAMES if out["sha256"][n] != want.get(n)]
    wheels, compat = os.path.join(WORK, "wheels"), os.path.join(WORK, "compat")
    for d, n in ((wheels, NAMES[0]), (compat, NAMES[1]), (WORK, NAMES[2]), (WORK, NAMES[4])):
        os.makedirs(d, exist_ok=True)
        untar(os.path.join(WORK, n), d)
        os.remove(os.path.join(WORK, n))
    t = time.time()
    run([sys.executable, "-m", "venv", VENV], timeout=300)
    whls = sorted(os.path.join(wheels, f) for f in os.listdir(wheels) if f.endswith(".whl"))
    r = run([os.path.join(VENV, "bin", "pip"), "install", "-q", "--no-index", "--no-deps", *whls], timeout=1800)
    site = os.path.join(VENV, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}", "site-packages")
    untar(os.path.join(WORK, NAMES[3]), site)
    os.remove(os.path.join(WORK, NAMES[3]))
    shutil.rmtree(wheels)
    out["install"] = {"rc": r["rc"], "wheels": len(whls), "seconds": round(time.time() - t, 1),
                      "err_tail": (r.get("err") or "").strip().splitlines()[-4:]}
    py = os.path.join(VENV, "bin", "python")
    repo = os.path.join(WORK, "exactsample")
    out["commit"] = open(os.path.join(repo, "COMMIT")).read().strip()
    out["library_sha256"] = sha256(os.path.join(repo, "python", "exactsample", "exactsample_cuda.so"))
    base = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                XDG_CACHE_HOME=os.path.join(WORK, "cache"))
    base.pop("PYTHONPATH", None)
    node = last_json(run([py, "-c", TORCH_CHECK], env=base, timeout=300))
    out["torch_node_driver"] = node
    env = base
    if not node.get("available"):
        env = dict(base, LD_LIBRARY_PATH=compat + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        out["torch_compat"] = last_json(run([py, "-c", TORCH_CHECK], env=env, timeout=300))
    out["compat_driver_used"] = env is not base
    out["nvidia_smi_before"] = smi()
    cuda_env = dict(env, EXACTSAMPLE_REQUIRE_CUDA="1")

    r = run([py, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={WORK}/pytest.xml"],
            env=cuda_env, cwd=repo, timeout=1200)
    open(os.path.join(WORK, "pytest.log"), "w").write(r["out"] + ("\n--- stderr ---\n" + r["err"] if r["err"] else ""))
    out["pytest"] = {"rc": r["rc"], "seconds": r["seconds"],
                     "summary": [x for x in r["out"].splitlines() if " passed" in x or " failed" in x][-1:]}
    r = run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_cuda.py", "tests/test_fused.py"],
            env=cuda_env, cwd=repo, timeout=900)
    out["pytest_cuda_files"] = {"rc": r["rc"], "summary": r["out"].strip().splitlines()[-1:]}

    for tag, capture, ref in (("8x8", "results/logits/qwen3-8b-8x8.npz", "results/agreement-8x8.json"),
                              ("full", os.path.join(WORK, "logits", "qwen3-8b.npz"), "results/agreement.json")):
        name = "agreement-8x8-a10" if tag == "8x8" else "agreement-a10"
        res = os.path.join(WORK, name + ".json")
        r = run([py, "bench/agreement.py", "--capture", capture, "--output", res,
                 "--outputs", os.path.join(WORK, f"outputs-{tag}-a10.npz")], env=env, cwd=repo, timeout=900)
        open(os.path.join(WORK, name + ".log"), "w").write(r["out"] + r["err"])
        entry = {"rc": r["rc"], "seconds": r["seconds"]}
        if os.path.exists(res):
            cells = json.load(open(res))["cells"]
            entry["identical_requests"] = sum(c["identical_requests"] for c in cells)
            entry["requests"] = sum(c["requests"] for c in cells)
            entry["fp64_fallbacks"] = [c["fp64_fallbacks"] for c in cells]
            entry["json_same_as_rtx4090"] = open(res, "rb").read() == open(os.path.join(repo, ref), "rb").read()
        else:
            entry["err_tail"] = r["err"].strip().splitlines()[-8:]
        out[f"agreement_{tag}"] = entry

    r = run([os.path.join(repo, "bin", "gumbel_bound")], env=env, cwd=repo, timeout=900)
    open(os.path.join(WORK, "gumbel-bound-a10.json"), "w").write(r["out"])
    out["gumbel_bound"] = {"rc": r["rc"], "seconds": r["seconds"], "err_tail": r["err"].strip().splitlines()[-4:],
                           "same_as_rtx4090": r["out"].encode() == open(os.path.join(repo, "results/gumbel-bound.json"),
                                                                         "rb").read()}
    out["nvidia_smi_after"] = smi()
    for f in ("pytest.log", "pytest.xml", "agreement-8x8-a10.json", "agreement-a10.json", "agreement-8x8-a10.log",
              "agreement-a10.log", "outputs-8x8-a10.npz", "outputs-full-a10.npz", "gumbel-bound-a10.json"):
        emit_file(os.path.join(WORK, f))
except Exception as e:
    out["error"] = repr(e)[:800]
out["payload_seconds"] = round(time.time() - T0, 1)
print("RESULT " + json.dumps(out), flush=True)
