"""End-to-end vLLM 0.31 decoding with exactsample on or off.

``run`` starts one engine (one arm) and serves the same seeded requests at
each setting and concurrency: all requests of a cell are admitted together,
each generates exactly --output-len tokens (ignore_eos), and the frontend
timestamps every output. Per cell it records wall time, output throughput,
TPOT ((last token - first token) / (tokens - 1), per request), TTFT and a
hash of every request's tokens, with the GPU state before and after.

``--profile`` also wraps the runner's ``sample`` (logits plus sampling) in
CUDA events and host timers to split each step into its logits and sampling
stages and to time exactsample's bridge call on the host. The wrappers cost a
few microseconds per step, so profiled runs are reported separately.

``summarize`` groups run files by model, speculation, arm, setting and
concurrency, and reports medians and ranges over repetitions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

SETTINGS = {
    # Qwen3's recommended sampling settings, and a temperature-only setting.
    "topk_topp": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
    "temp_only": {"temperature": 1.0},
}
TEXT = ("The river bends past the old mill, where the water slows and the light falls through the "
        "willows onto stones that have not moved in a hundred years. Children come in summer to "
        "count the fish, and in winter the ice holds the leaves where they fell. ")


# --- worker side (loaded through worker_extension_cls) -----------------------

class E2EProbe:
    def e2e_install_timing(self):
        return _install_timing(self)

    def e2e_read_timing(self):
        return _read_timing(self)

    def e2e_bridge_steps(self):
        state = getattr(self.model_runner, "_exactsample_bridge", None)
        return None if state is None else int(state["step"])


def _install_timing(worker):
    import torch
    runner = worker.model_runner
    worker._e2e_steps = steps = []
    current = {}
    original_sample, original_logits = runner.sample, runner.model.compute_logits

    def compute_logits(*args, **kwargs):
        logits = original_logits(*args, **kwargs)
        if current.get("open") and "logits" not in current:
            current["logits"] = torch.cuda.Event(enable_timing=True)
            current["logits"].record()
        return logits

    def sample(hidden_states, input_batch, grammar_output):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        current.clear()
        current["open"] = True
        began = time.perf_counter_ns()
        result = original_sample(hidden_states, input_batch, grammar_output)
        host_ns = time.perf_counter_ns() - began
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        current["open"] = False
        steps.append({"start": start, "logits": current.get("logits"), "end": end, "sample_host_ns": host_ns,
                      "bridge_host_ns": current.get("bridge_ns"), "bridge_used": current.get("bridge_used"),
                      "verify_host_ns": current.get("verify_ns"), "staging_wait_ns": current.get("wait_ns"),
                      "requests": int(input_batch.num_reqs), "rows": int(input_batch.cu_num_logits_np[-1]),
                      "has_prefill": bool(input_batch.has_prefill)})
        return result

    runner.sample = sample
    runner.model.compute_logits = compute_logits
    bridge = None
    if os.environ.get("EXACTSAMPLE_DECODE") == "1":
        from exactsample import cuda as library, vllm_bridge as bridge
        original_try, original_verify = bridge.try_sample, library.verify

        def try_sample(*args):
            current["verify_ns"] = current["wait_ns"] = 0
            began = time.perf_counter_ns()
            result = original_try(*args)
            current["bridge_ns"] = time.perf_counter_ns() - began
            current["bridge_used"] = result is not None
            return result

        def verify(*args, **kwargs):  # the library call, inside the bridge
            began = time.perf_counter_ns()
            result = original_verify(*args, **kwargs)
            current["verify_ns"] += time.perf_counter_ns() - began
            return result

        class TimedEvent:  # the staging ring's events, when this bridge has them
            def __init__(self, event):
                self.event = event

            def synchronize(self):
                began = time.perf_counter_ns()
                self.event.synchronize()
                current["wait_ns"] = current.get("wait_ns", 0) + time.perf_counter_ns() - began

            def record(self, *args):
                self.event.record(*args)

        bridge.try_sample, library.verify = try_sample, verify
        state = getattr(runner, "_exactsample_bridge", None)
        if state is not None and "staging_events" in state:
            state["staging_events"] = [TimedEvent(event) for event in state["staging_events"]]
    return {"installed": True, "bridge_sha256": None if bridge is None else
            hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()}


def _read_timing(worker):
    import torch
    torch.cuda.synchronize()
    steps, worker._e2e_steps[:] = list(worker._e2e_steps), []
    result = []
    for index, step in enumerate(steps):
        item = {key: value for key, value in step.items() if key not in ("start", "logits", "end")}
        item["sample_gpu_ms"] = step["start"].elapsed_time(step["end"])
        if step["logits"] is not None:
            item["logits_gpu_ms"] = step["start"].elapsed_time(step["logits"])
            item["sampling_gpu_ms"] = step["logits"].elapsed_time(step["end"])
        if index + 1 < len(steps):
            item["step_period_gpu_ms"] = step["start"].elapsed_time(steps[index + 1]["start"])
        result.append(item)
    return result


# --- frontend -----------------------------------------------------------------

def gpu_state():
    """NVML snapshot; clocks cannot be locked on this machine, so record them.

    The CPU is shared with other work too, so the load average is recorded.
    """
    load = {"loadavg_1m": os.getloadavg()[0], "nproc": os.cpu_count()}
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        clock = lambda kind: nvml.nvmlDeviceGetClockInfo(handle, kind)
        return {"time": datetime.now(timezone.utc).isoformat(), "pstate": nvml.nvmlDeviceGetPerformanceState(handle),
                "sm_clock_mhz": clock(nvml.NVML_CLOCK_SM), "mem_clock_mhz": clock(nvml.NVML_CLOCK_MEM),
                "temperature_c": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                "power_w": nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                "utilization_percent": nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "memory_used_mib": nvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20,
                "compute_processes": len(nvml.nvmlDeviceGetComputeRunningProcesses(handle))} | load
    except Exception as error:  # observation only
        return {"error": str(error)} | load


def prompts(tokenizer, count, length):
    stream = tokenizer.encode(TEXT * 40, add_special_tokens=False)
    if len(stream) < length + 13 * count:
        raise ValueError("prompt text too short")
    return [stream[13 * index:13 * index + length] for index in range(count)]


def sampling_params(setting, seed, max_tokens):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    return SamplingParams(**SETTINGS[setting], seed=seed, max_tokens=max_tokens, ignore_eos=True,
                          detokenize=False, output_kind=RequestOutputKind.DELTA)


def serve(engine, tag, token_prompts, setting, seed_base, max_tokens):
    """Admit every request at once and timestamp each output on arrival."""
    began = time.perf_counter()
    for index, prompt in enumerate(token_prompts):
        engine.add_request(f"{tag}-{index}", {"prompt_token_ids": prompt},
                           sampling_params(setting, seed_base + index, max_tokens))
    arrivals = defaultdict(list)  # request -> [(time, tokens in this output)]
    tokens = defaultdict(list)
    frontend_steps = 0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        frontend_steps += 1
        for output in outputs:
            new = list(output.outputs[0].token_ids)
            if new:
                arrivals[output.request_id].append((now, len(new)))
                tokens[output.request_id].extend(new)
            if output.finished and output.outputs[0].finish_reason != "length":
                raise RuntimeError(f"{output.request_id} finished with {output.outputs[0].finish_reason}")
    ended = time.perf_counter()
    order = [f"{tag}-{index}" for index in range(len(token_prompts))]
    if any(len(tokens[key]) != max_tokens for key in order):
        raise RuntimeError("a request did not produce exactly max_tokens tokens")
    tpot = [(arrivals[key][-1][0] - arrivals[key][0][0]) / (max_tokens - 1) * 1e3 for key in order]
    ttft = [(arrivals[key][0][0] - began) * 1e3 for key in order]
    per_output = [(max_tokens - arrivals[key][0][1]) / max(1, len(arrivals[key]) - 1) for key in order]
    digests = [hashlib.sha256(json.dumps(tokens[key]).encode()).hexdigest()[:16] for key in order]
    return {"wall_s": ended - began, "output_tokens": max_tokens * len(order),
            "throughput_tok_s": max_tokens * len(order) / (ended - began),
            "tpot_ms": summary(tpot), "ttft_ms": summary(ttft),
            "decode_tokens_per_output": summary(per_output), "frontend_steps": frontend_steps,
            "token_sha256_16": digests, "first_tokens": [tokens[key][:8] for key in order]}


def summary(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "mean": statistics.fmean(ordered),
            "min": ordered[0], "max": ordered[-1], "p90": ordered[min(len(ordered) - 1, int(.9 * len(ordered)))],
            "n": len(ordered)}


def profile_summary(steps):
    decode = [step for step in steps if not step["has_prefill"]]
    keys = ("sample_host_ns", "bridge_host_ns", "verify_host_ns", "staging_wait_ns", "sample_gpu_ms",
            "logits_gpu_ms", "sampling_gpu_ms", "step_period_gpu_ms")
    result = {"steps": len(steps), "decode_steps": len(decode),
              "bridge_fallback_steps": sum(step["bridge_used"] is False for step in steps)}
    result.update({key: summary([step.get(key) for step in decode]) for key in keys})
    if result["sampling_gpu_ms"] and result["step_period_gpu_ms"]:
        result["sampling_share_of_step"] = (result["sampling_gpu_ms"]["median"]
                                            / result["step_period_gpu_ms"]["median"])
    return result


def model_label(path):
    """'Qwen/Qwen3-1.7B@70d244cc86cc' for a Hugging Face cache snapshot directory."""
    if path is None:
        return None
    parts = Path(path).parts
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        return parts[-3].removeprefix("models--").replace("--", "/") + "@" + parts[-1][:12]
    return Path(path).name


def display_path(path):
    """A path relative to the repository, or with the home directory as ~."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path).replace(str(Path.home()), "~", 1)


def provenance(args):
    import torch
    import vllm

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def git(*command):
        return subprocess.run(["git", *command], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()

    record = {"commit": git("rev-parse", "HEAD"), "worktree_clean": git("status", "--porcelain") == "",
              "vllm": vllm.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "model_runner_sha256": sha(Path(vllm.__file__).parent / "v1/worker/gpu/model_runner.py"),
              "python": sys.version.split()[0]}
    try:
        import exactsample.vllm_bridge as bridge
        record.update(bridge_file=display_path(bridge.__file__), bridge_sha256=sha(bridge.__file__))
    except ImportError:
        pass
    record["env"] = {key: os.environ.get(key) for key in (
        "EXACTSAMPLE_DECODE", "EXACTSAMPLE_MODE", "EXACTSAMPLE_DISPATCH", "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_BATCH_INVARIANT", "EXACTSAMPLE_TRACE_PATH", "EXACTSAMPLE_CAPTURE_DIR")}
    return record


def run(args):
    if args.gpu_memory_utilization > .83:
        raise SystemExit("gpu memory utilization is capped at 0.83 on this machine")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    os.environ.update(VLLM_USE_V2_MODEL_RUNNER="1", EXACTSAMPLE_DECODE="1" if args.exactsample == "on" else "0",
                      EXACTSAMPLE_MODE=args.mode, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("EXACTSAMPLE_TRACE_PATH", "EXACTSAMPLE_CAPTURE_DIR"):
        os.environ.pop(key, None)
    from vllm import LLM
    concurrencies = sorted(set(args.concurrency))
    record = {"schema": "exactsample.vllm_e2e.v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "label": args.label, "arm": args.arm, "rep": args.rep, "exactsample": args.exactsample,
              "mode": args.mode, "profile": args.profile, "target": model_label(args.target),
              "draft": model_label(args.draft),
              "prompt_len": args.prompt_len, "output_len": args.output_len, "warmup_len": args.warmup_len,
              "seed_base": args.seed_base, "provenance": provenance(args), "gpu_start": gpu_state()}
    speculative = None if args.draft is None else {
        "model": str(args.draft), "method": "draft_model", "num_speculative_tokens": args.num_speculative_tokens,
        "draft_sample_method": "greedy"}
    engine_args = dict(model=str(args.target), tokenizer=str(args.target), dtype="bfloat16",
                       max_model_len=args.max_model_len, max_num_seqs=max(concurrencies),
                       max_num_batched_tokens=args.max_num_batched_tokens,
                       gpu_memory_utilization=args.gpu_memory_utilization, seed=0, enable_prefix_caching=False,
                       speculative_config=speculative, disable_log_stats=True,
                       worker_extension_cls="bench.vllm_e2e.E2EProbe")
    record["engine_args"] = engine_args | {"model": model_label(args.target), "tokenizer": model_label(args.target),
                                           "speculative_config": speculative and speculative | {
                                               "model": model_label(args.draft)}}
    started = time.perf_counter()
    llm = LLM(**engine_args)
    record["engine_start_s"] = time.perf_counter() - started
    config = llm.llm_engine.vllm_config
    blocks = config.cache_config.num_gpu_blocks
    capacity = None if blocks is None else blocks * config.cache_config.block_size
    needed = max(concurrencies) * (args.prompt_len + args.output_len + args.num_speculative_tokens + 1)
    record["engine"] = {"async_scheduling": config.scheduler_config.async_scheduling,
                        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
                        "compilation_mode": str(config.compilation_config.mode),
                        "kv_cache_tokens": capacity, "kv_tokens_needed": needed}
    if capacity is not None and capacity < needed:
        raise SystemExit(f"KV cache holds {capacity} tokens, the largest cell needs {needed}: it would preempt")
    engine = llm.llm_engine
    token_prompts = prompts(llm.get_tokenizer(), max(concurrencies), args.prompt_len)
    if args.profile:
        record["timing_hooks"] = llm.collective_rpc("e2e_install_timing")[0]
    record["cells"] = []
    for setting in args.settings:
        for concurrency in concurrencies:
            cell = {"setting": setting, "concurrency": concurrency, "gpu_before": gpu_state()}
            selected = token_prompts[:concurrency]
            serve(engine, f"warm-{setting}-{concurrency}", selected, setting, args.seed_base, args.warmup_len)
            if args.profile:
                llm.collective_rpc("e2e_read_timing")  # discard the warm-up steps
            steps_before = llm.collective_rpc("e2e_bridge_steps")[0]
            cell.update(serve(engine, f"run-{setting}-{concurrency}", selected, setting, args.seed_base,
                              args.output_len))
            steps_after = llm.collective_rpc("e2e_bridge_steps")[0]
            cell["exactsample_steps"] = None if steps_before is None else steps_after - steps_before
            if args.profile:
                cell["profile"] = profile_summary(llm.collective_rpc("e2e_read_timing")[0])
            cell["gpu_after"] = gpu_state()
            record["cells"].append(cell)
            print(json.dumps({key: cell[key] for key in ("setting", "concurrency", "throughput_tok_s")}
                             | {"tpot_ms": cell["tpot_ms"]["median"]}), flush=True)
    record["gpu_end"] = gpu_state()
    record["state"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(record, output, indent=1)
        output.write("\n")


# --- summary --------------------------------------------------------------------

def spread(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def summarize(args):
    groups = defaultdict(list)
    profiles = defaultdict(list)
    for path in sorted(args.runs.glob("**/*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "exactsample.vllm_e2e.v1" or record.get("state") != "passed":
            continue
        model, draft = record["target"], record["draft"]
        for cell in record["cells"]:
            key = (model, draft, cell["setting"], cell["concurrency"])
            entry = {"arm": record["arm"], "rep": record["rep"], "file": str(path.relative_to(args.runs)),
                     "throughput": cell["throughput_tok_s"], "tpot": cell["tpot_ms"]["median"],
                     "ttft": cell["ttft_ms"]["median"], "tokens_per_output": cell["decode_tokens_per_output"]["median"],
                     "exactsample_steps": cell["exactsample_steps"], "frontend_steps": cell["frontend_steps"],
                     "digests": cell["token_sha256_16"],
                     "sm_clock": [cell["gpu_before"].get("sm_clock_mhz"), cell["gpu_after"].get("sm_clock_mhz")]}
            (profiles if record["profile"] else groups)[key].append(entry | ({"profile": cell["profile"]}
                                                                             if record["profile"] else {}))
    cells = []
    for key in sorted(groups, key=str):
        model, draft, setting, concurrency = key
        by_arm = defaultdict(list)
        for entry in groups[key]:
            by_arm[entry["arm"]].append(entry)
        row = {"model": model, "draft": draft, "setting": setting, "concurrency": concurrency, "arms": {}}
        for arm, entries in sorted(by_arm.items()):
            row["arms"][arm] = {
                "reps": len(entries), "throughput_tok_s": spread([e["throughput"] for e in entries]),
                "tpot_ms": spread([e["tpot"] for e in entries]), "ttft_ms": spread([e["ttft"] for e in entries]),
                "decode_tokens_per_output": spread([e["tokens_per_output"] for e in entries]),
                "exactsample_steps": sorted({e["exactsample_steps"] for e in entries}, key=str),
                "identical_tokens_across_reps": len({tuple(e["digests"]) for e in entries}) == 1,
                "sm_clock_mhz": [min((c for e in entries for c in e["sm_clock"] if c), default=None),
                                 max((c for e in entries for c in e["sm_clock"] if c), default=None)],
                "files": sorted(e["file"] for e in entries)}
        if "off" in row["arms"]:
            off = row["arms"]["off"]
            for arm in row["arms"]:
                if arm != "off":
                    row["arms"][arm]["throughput_vs_off"] = (row["arms"][arm]["throughput_tok_s"]["median"]
                                                             / off["throughput_tok_s"]["median"])
                    row["arms"][arm]["tpot_vs_off"] = row["arms"][arm]["tpot_ms"]["median"] / off["tpot_ms"]["median"]
        cells.append(row)
    profiled = []
    for key in sorted(profiles, key=str):
        model, draft, setting, concurrency = key
        for entry in sorted(profiles[key], key=lambda e: e["arm"]):
            profile = entry["profile"]
            pick = lambda name, field="median": None if profile.get(name) is None else profile[name][field]
            profiled.append({"model": model, "draft": draft, "setting": setting, "concurrency": concurrency,
                             "arm": entry["arm"], "file": entry["file"], "decode_steps": profile["decode_steps"],
                             "bridge_fallback_steps": profile["bridge_fallback_steps"],
                             "bridge_host_us": None if pick("bridge_host_ns") is None else pick("bridge_host_ns") / 1e3,
                             "bridge_host_us_p90": None if pick("bridge_host_ns", "p90") is None
                             else pick("bridge_host_ns", "p90") / 1e3,
                             "verify_host_us": None if pick("verify_host_ns") is None else pick("verify_host_ns") / 1e3,
                             "staging_wait_us": None if pick("staging_wait_ns") is None
                             else pick("staging_wait_ns") / 1e3,
                             "sample_host_us": pick("sample_host_ns") / 1e3,
                             "logits_gpu_ms": pick("logits_gpu_ms"), "sampling_gpu_ms": pick("sampling_gpu_ms"),
                             "step_period_gpu_ms": pick("step_period_gpu_ms"),
                             "sampling_share_of_step": profile.get("sampling_share_of_step"),
                             "tpot_ms": entry["tpot"], "throughput_tok_s": entry["throughput"]})
    result = {"schema": "exactsample.vllm_e2e_summary.v1", "runs_dir": str(args.runs),
              "timed": cells, "profiled": profiled}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for row in cells:
        arms = " | ".join(f"{arm}: {v['throughput_tok_s']['median']:.0f} tok/s, TPOT {v['tpot_ms']['median']:.2f} "
                          f"[{v['tpot_ms']['min']:.2f}, {v['tpot_ms']['max']:.2f}] ms (n={v['reps']})"
                          for arm, v in row["arms"].items())
        print(f"{row['model']} draft={row['draft']} {row['setting']} c={row['concurrency']}: {arms}")
    for row in profiled:
        print(f"PROFILE {row['model']} draft={row['draft']} {row['setting']} c={row['concurrency']} {row['arm']}: "
              f"bridge {row['bridge_host_us']} us, sample host {row['sample_host_us']:.0f} us, sampling "
              f"{row['sampling_gpu_ms']:.3f} ms of step {row['step_period_gpu_ms']:.3f} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("--target", type=Path, required=True)
    one.add_argument("--draft", type=Path)
    one.add_argument("--num-speculative-tokens", type=int, default=4)
    one.add_argument("--exactsample", choices=("on", "off"), required=True)
    one.add_argument("--mode", choices=("coupled", "ratio"), default="coupled")
    one.add_argument("--arm", required=True, help="label of this configuration in the summary (off, on, ...)")
    one.add_argument("--rep", type=int, default=0)
    one.add_argument("--label", default="")
    one.add_argument("--settings", nargs="+", choices=sorted(SETTINGS), default=["topk_topp", "temp_only"])
    one.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    one.add_argument("--prompt-len", type=int, default=128)
    one.add_argument("--output-len", type=int, default=256)
    one.add_argument("--warmup-len", type=int, default=16)
    one.add_argument("--seed-base", type=int, default=1000)
    one.add_argument("--max-model-len", type=int, default=512)
    one.add_argument("--max-num-batched-tokens", type=int, default=8192)
    one.add_argument("--gpu-memory-utilization", type=float, default=.8)
    one.add_argument("--profile", action="store_true")
    one.add_argument("--output", type=Path, required=True)
    both = commands.add_parser("summarize")
    both.add_argument("runs", type=Path)
    both.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    (run if args.command == "run" else summarize)(args)


if __name__ == "__main__":
    main()
