"""Do a request's tokens depend on what else the engine is serving?

``run`` starts one engine (sampler: exactsample or vLLM's own; with or without
VLLM_BATCH_INVARIANT) and serves the same probe requests in several
scenarios. A worker hook hashes each probe's target logits rows as they are
computed, before any sampling code sees them, and records admissions (with
vLLM's per-slot seed) and preemptions. Comparing a probe's run in a scenario
with its run alone then separates the two sources of a token difference: if
the logits row that produced the first differing token is bitwise identical
to the one in the alone run, the sampler made the difference; otherwise the
model's logits already differed.

The probes are six seeded requests (two each at T=0.6 top-k 20 top-p 0.95,
T=1.0, and T=0.8 top-p 0.9) on chat prompts from bench/capture_logits.py, and
two unseeded ones. Requests are added while the scheduler is paused, so each
group joins the same step.

Scenarios with --kv large (seeded probes): ``alone`` (each probe by itself),
``mixed_first`` (probes, then 15 other requests, admitted together),
``mixed_last`` (31 other requests first, probes last, so they get other
slots), ``staggered`` (probes join 3 steps after 15 others; 16 more join 5
steps later). With --kv small: ``preempt`` (all eight probes join four long
requests in a KV cache too small for all of them as they grow: the scheduler
preempts the latest admitted and recomputes them later), then ``alone``, in
which each unseeded probe gets, as an explicit seed, the seed vLLM gave it at
its first admission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

QWEN3 = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
TEMPERATURE = {"temperature": 1.0}
NUCLEUS = {"temperature": 0.8, "top_p": 0.9}
PROBES = {  # name: (index into bench/capture_logits.py PROMPTS, setting, seed)
    "p0": (0, QWEN3, 101), "p1": (16, QWEN3, 202),
    "p2": (27, TEMPERATURE, 303), "p3": (10, TEMPERATURE, 404),
    "p4": (3, NUCLEUS, 505), "p5": (21, NUCLEUS, 606),
}
UNSEEDED = {"u0": (5, QWEN3, None), "u1": (28, QWEN3, None)}
FILLER_SETTINGS = (QWEN3, TEMPERATURE, NUCLEUS)


# --- worker side ------------------------------------------------------------

class DeterminismProbe:
    def det_install(self):
        return _install(self)

    def det_read(self):
        log = self._det_log
        result = {key: list(value) for key, value in log.items()}
        for value in log.values():
            value.clear()
        return result


def _install(worker):
    import torch
    runner = worker.model_runner
    log = worker._det_log = {"rows": [], "admissions": [], "preemptions": [], "steps": []}
    current = {}
    original_sample, original_logits = runner.sample, runner.model.compute_logits
    original_add, original_finish = runner.sampler.add_request, runner.finish_requests

    def compute_logits(*args, **kwargs):
        logits = original_logits(*args, **kwargs)
        batch = current.get("batch")
        if batch is None:
            return logits
        cu = batch.cu_num_logits_np
        # vLLM's own sampler uses FlashInfer's fused kernel (process-wide RNG)
        # for top-k/top-p batches without any explicitly seeded request.
        seeded = bool(runner.sampler.sampling_states.seeds_set[batch.idx_mapping_np].any())
        for index, request in enumerate(batch.req_ids):
            if not request.startswith(("probe-", "unseeded-")):
                continue
            start, end = int(cu[index]), int(cu[index + 1])
            positions = batch.positions[batch.logits_indices[start:end]].tolist()
            rows = logits[start:end].contiguous().view(torch.int16 if logits.element_size() == 2 else torch.int32)
            for position, row in zip(positions, rows.cpu().numpy()):
                log["rows"].append({"request": request, "position": int(position),
                                    "sha": hashlib.sha256(row.tobytes()).hexdigest()[:20],
                                    "dtype": str(logits.dtype), "batch_requests": len(batch.req_ids),
                                    "batch_has_explicit_seed": seeded})
        return logits

    def sample(hidden_states, input_batch, grammar_output):
        current["batch"] = input_batch
        current.pop("bridge_used", None)
        try:
            return original_sample(hidden_states, input_batch, grammar_output)
        finally:
            current["batch"] = None
            log["steps"].append({"requests": list(input_batch.req_ids), "bridge_used": current.get("bridge_used")})

    def add_request(index, params):
        result = original_add(index, params)
        request = next(key for key, value in runner.req_states.req_id_to_index.items() if value == index)
        log["admissions"].append({"request": request, "slot": int(index), "requested_seed": params.seed,
                                  "native_seed": int(runner.sampler.sampling_states.seeds.np[index])})
        return result

    def finish_requests(scheduler_output):
        preempted = sorted(scheduler_output.preempted_req_ids or [])
        if preempted:
            log["preemptions"].append(preempted)
        return original_finish(scheduler_output)

    runner.sample, runner.model.compute_logits = sample, compute_logits
    runner.sampler.add_request, runner.finish_requests = add_request, finish_requests
    if os.environ.get("EXACTSAMPLE_DECODE") == "1":
        from exactsample import vllm_bridge
        original_try = vllm_bridge.try_sample

        def try_sample(*args):
            result = original_try(*args)
            current["bridge_used"] = result is not None
            return result

        vllm_bridge.try_sample = try_sample
    return {"installed": True}


# --- frontend ---------------------------------------------------------------

def make_params(setting, seed, max_tokens):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    return SamplingParams(**setting, seed=seed, max_tokens=max_tokens, ignore_eos=True, detokenize=False,
                          output_kind=RequestOutputKind.DELTA)


class Server:
    def __init__(self, llm, max_tokens):
        from bench.capture_logits import PROMPTS
        self.llm, self.engine, self.max_tokens, self.texts = llm, llm.llm_engine, max_tokens, PROMPTS
        self.tokenizer = llm.get_tokenizer()
        self.tokens, self.prompt_lens = {}, {}

    def chat(self, text):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def admit(self, *groups):
        """Add requests to a paused scheduler, so they all join the same step."""
        self.engine.engine_core.call_utility("pause_scheduler", "keep", False)
        for add, arguments in groups:
            add(*arguments)
        self.engine.engine_core.call_utility("resume_scheduler")

    def add(self, request_id, text, setting, seed, max_tokens=None):
        prompt = self.chat(text)
        self.tokens[request_id], self.prompt_lens[request_id] = [], len(prompt)
        self.engine.add_request(request_id, {"prompt_token_ids": prompt},
                                make_params(setting, seed, max_tokens or self.max_tokens))

    def add_probes(self, scenario, names=PROBES, seeds=None):
        for name in names:
            index, setting, seed = (PROBES | UNSEEDED)[name]
            prefix = "unseeded" if name in UNSEEDED else "probe"
            # A probe's prompt is the same in every scenario.
            self.add(f"{prefix}-{name}-{scenario}", self.texts[index], setting,
                     seeds.get(name, seed) if seeds else seed)

    def add_fillers(self, scenario, count, first=0, long=False):
        for index in range(first, first + count):
            # Short: one prompt and 8 to 96 tokens; long: ten prompts in one message and 200 tokens.
            text = " ".join(self.texts[(index + k) % len(self.texts)] for k in range(10 if long else 1))
            self.add(f"filler-{scenario}-{index}", text, FILLER_SETTINGS[index % 3], 5000 + index,
                     max_tokens=200 if long else 8 + (23 * index) % 89)

    def step(self, count=None):
        steps = 0
        while self.engine.has_unfinished_requests() and (count is None or steps < count):
            for output in self.engine.step():
                self.tokens[output.request_id].extend(output.outputs[0].token_ids)
                if output.finished and output.outputs[0].finish_reason != "length":
                    raise RuntimeError(f"{output.request_id}: {output.outputs[0].finish_reason}")
            steps += 1


def internal_to_external(internal, external_ids):
    found = [key for key in external_ids if internal == key or (
        internal.startswith(key + "-") and len(internal) == len(key) + 9)]
    if len(found) != 1:
        raise RuntimeError(f"ambiguous request identity {internal}")
    return found[0]


def collect(server, scenario, log):
    """Per probe of this scenario: tokens, logits hashes by position, admissions."""
    probes = {key: value for key, value in server.tokens.items()
              if key.startswith(("probe-", "unseeded-")) and key.endswith("-" + scenario)}
    result = {}
    for external, tokens in probes.items():
        name = external.split("-")[1]
        rows, batches = {}, {}  # a position computed again after preemption keeps its last row
        for row in log["rows"]:
            if internal_to_external(row["request"], probes) == external:
                rows[str(row["position"])] = row["sha"]
                batches[str(row["position"])] = [row["batch_requests"], row["batch_has_explicit_seed"]]
        admissions = [entry for entry in log["admissions"] if entry["request"].startswith(external + "-")]
        preempted = sum(any(request.startswith(external + "-") for request in event) for event in log["preemptions"])
        result[name] = {"request_id": external, "tokens": tokens, "logits_sha_by_position": rows,
                        "batch_by_position": batches,
                        "admissions": admissions, "times_preempted": preempted,
                        "prompt_len": server.prompt_lens[external]}
    return result


def compare(alone, other):
    """First differing token, and whether that token's logits row differed."""
    tokens_a, tokens_b = alone["tokens"], other["tokens"]
    divergence = next((i for i, (a, b) in enumerate(zip(tokens_a, tokens_b)) if a != b), None)
    prompt = alone["prompt_len"]
    rows_a, rows_b = alone["logits_sha_by_position"], other["logits_sha_by_position"]
    # Output token i is drawn from the logits row at position prompt_len - 1 + i.
    shared = [i for i in range(len(tokens_a)) if str(prompt - 1 + i) in rows_a and str(prompt - 1 + i) in rows_b]
    differing = [i for i in shared if rows_a[str(prompt - 1 + i)] != rows_b[str(prompt - 1 + i)]]
    upto = shared if divergence is None else [i for i in shared if i <= divergence]
    first_logits = next((i for i in upto if rows_a[str(prompt - 1 + i)] != rows_b[str(prompt - 1 + i)]), None)
    if divergence is None:
        cause = None
    elif str(prompt - 1 + divergence) not in rows_a or str(prompt - 1 + divergence) not in rows_b:
        cause = "unknown (no logits row recorded)"
    else:
        cause = "model logits" if first_logits is not None else "sampler (identical logits)"
    batch = None if divergence is None else other["batch_by_position"].get(str(prompt - 1 + divergence))
    return {"tokens_equal": tokens_a == tokens_b, "first_token_divergence": divergence,
            "batch_at_divergence": None if batch is None else {"requests": batch[0], "has_explicit_seed": batch[1]},
            "logits_rows_compared_before_divergence": len(upto),
            "first_logits_difference": first_logits, "logits_rows_differing_total": len(differing),
            "logits_rows_compared_total": len(shared), "cause": cause,
            "times_preempted": other["times_preempted"]}


def run(args):
    from bench.vllm_e2e import gpu_state, model_label, provenance
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    os.environ.update(VLLM_USE_V2_MODEL_RUNNER="1", EXACTSAMPLE_DECODE="1" if args.exactsample == "on" else "0",
                      EXACTSAMPLE_MODE="coupled", VLLM_BATCH_INVARIANT="1" if args.batch_invariant else "0",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("EXACTSAMPLE_TRACE_PATH", "EXACTSAMPLE_CAPTURE_DIR"):
        os.environ.pop(key, None)
    from vllm import LLM
    record = {"schema": "exactsample.vllm_determinism.v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "exactsample": args.exactsample, "batch_invariant": args.batch_invariant, "kv": args.kv,
              "target": model_label(args.target), "max_tokens": args.max_tokens, "provenance": provenance(args),
              "gpu_start": gpu_state()}
    engine_args = dict(model=str(args.target), tokenizer=str(args.target), dtype="bfloat16", max_model_len=512,
                       max_num_seqs=64, max_num_batched_tokens=2048, gpu_memory_utilization=.5, seed=0,
                       enable_prefix_caching=False, disable_log_stats=True, enforce_eager=args.enforce_eager,
                       kv_cache_memory_bytes=134217728 if args.kv == "small" else 2 * 2**30,
                       worker_extension_cls="integration.vllm_determinism.DeterminismProbe")
    record["engine_args"] = engine_args | {"model": model_label(args.target), "tokenizer": model_label(args.target)}
    llm = LLM(**engine_args)
    config = llm.llm_engine.vllm_config
    record["engine"] = {"kv_cache_tokens": None if config.cache_config.num_gpu_blocks is None
                        else config.cache_config.num_gpu_blocks * config.cache_config.block_size,
                        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
                        "async_scheduling": config.scheduler_config.async_scheduling}
    server = Server(llm, args.max_tokens)
    llm.collective_rpc("det_install")
    scenarios, steps = {}, {}

    def finish(name):
        server.step()
        log = llm.collective_rpc("det_read")[0]
        scenarios[name] = collect(server, name, log)
        steps[name] = {"steps": len(log["steps"]),
                       "bridge_fallback_steps": sum(step["bridge_used"] is False for step in log["steps"]),
                       "preemption_events": len(log["preemptions"])}

    def alone(seeds=None, names=PROBES):
        for name in names:
            server.admit((server.add_probes, ("alone", [name], seeds)))
            server.step()
        finish("alone")

    if args.kv == "large":
        alone()
        server.admit((server.add_probes, ("mixed_first",)), (server.add_fillers, ("mixed_first", 15)))
        finish("mixed_first")
        server.admit((server.add_fillers, ("mixed_last", 31)), (server.add_probes, ("mixed_last",)))
        finish("mixed_last")
        server.admit((server.add_fillers, ("staggered", 15)))
        server.step(3)
        server.admit((server.add_probes, ("staggered",)))
        server.step(5)
        server.admit((server.add_fillers, ("staggered", 16, 15)))
        finish("staggered")
    else:
        # The probes join two steps after four long requests. The cache holds
        # all of them at first but not as they grow; the scheduler then
        # preempts the latest admitted (the probes) and recomputes them later.
        server.admit((server.add_fillers, ("preempt", 4, 0, True)))
        server.step(2)
        server.admit((server.add_probes, ("preempt", [*PROBES, *UNSEEDED])))
        finish("preempt")
        # vLLM gave each unseeded probe a seed at its first admission; the
        # bridge keeps it across preemption, vLLM's own sampler draws anew.
        seeds = {name: scenarios["preempt"][name]["admissions"][0]["native_seed"] for name in UNSEEDED}
        alone(seeds=seeds, names=[*PROBES, *UNSEEDED])
        record["unseeded_probes"] = {name: {"first_admission_seed": seeds[name], "native_seeds_at_admissions": [
            entry["native_seed"] for entry in scenarios["preempt"][name]["admissions"]]} for name in UNSEEDED}
    record["scenarios"], record["steps"] = scenarios, steps
    record["comparisons"] = {
        name: {probe: compare(scenarios["alone"][probe], value) for probe, value in probes.items()}
        for name, probes in scenarios.items() if name != "alone"}
    record["gpu_end"] = gpu_state()
    record["state"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(record, output, indent=1)
        output.write("\n")
    for name, probes in record["comparisons"].items():
        for probe, item in probes.items():
            print(name, probe, json.dumps(item))


def summarize(args):
    rows = []
    for path in sorted(args.runs.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "exactsample.vllm_determinism.v1" or record.get("state") != "passed":
            continue
        for scenario, probes in record["comparisons"].items():
            rows.append({"file": path.name, "sampler": "exactsample" if record["exactsample"] == "on" else "vllm",
                         "batch_invariant": record["batch_invariant"], "kv": record["kv"], "scenario": scenario,
                         "probes": len(probes),
                         "tokens_equal": sorted(p for p, item in probes.items() if item["tokens_equal"]),
                         "logits_equal": sorted(p for p, item in probes.items()
                                                if item["logits_rows_differing_total"] == 0),
                         "causes": {p: item["cause"] for p, item in probes.items() if item["cause"]},
                         "first_token_divergence": {p: item["first_token_divergence"] for p, item in probes.items()},
                         "times_preempted": {p: item["times_preempted"] for p, item in probes.items()},
                         "bridge_fallback_steps": record["steps"][scenario]["bridge_fallback_steps"],
                         "unseeded_probes": record.get("unseeded_probes") if scenario == "preempt" else None})
    args.output.write_text(json.dumps({"schema": "exactsample.vllm_determinism_summary.v1", "rows": rows},
                                      indent=1) + "\n")
    for row in rows:
        print(f"{row['sampler']:11s} BI={int(row['batch_invariant'])} {row['scenario']:11s} tokens equal "
              f"{len(row['tokens_equal'])}/{row['probes']}, logits equal {len(row['logits_equal'])}/{row['probes']}, "
              f"causes {row['causes']}, preempted {row['times_preempted']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("--target", type=Path, required=True)
    one.add_argument("--exactsample", choices=("on", "off"), required=True)
    one.add_argument("--batch-invariant", action="store_true")
    one.add_argument("--kv", choices=("large", "small"), required=True)
    one.add_argument("--max-tokens", type=int, default=64)
    one.add_argument("--enforce-eager", action="store_true")
    one.add_argument("--output", type=Path, required=True)
    both = commands.add_parser("summarize")
    both.add_argument("runs", type=Path)
    both.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    (run if args.command == "run" else summarize)(args)


if __name__ == "__main__":
    main()
