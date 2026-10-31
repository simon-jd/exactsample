"""Real public abort and low-KV preemption/recompute checks on a local engine.

The worker extension only observes admission, scheduler preemptions and cleanup.
It does not force scheduler transitions or replace model/sampling outputs.
"""
import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

if __package__:
    from .gpu_observer import GpuObserver
else:
    from gpu_observer import GpuObserver


def install_observer(worker):
    from exactsample import vllm_bridge as bridge
    events = worker._exactsample_lifecycle_events = []
    runner = worker.model_runner
    original_register = bridge.register_request
    original_finish = runner.finish_requests

    def register(runner, request_id, request_index, params):
        prior = runner._exactsample_bridge["request_seeds"].get(str(request_id))
        original_register(runner, request_id, request_index, params)
        events.append({"event": "register", "request_id": str(request_id),
                       "slot": int(request_index), "requested_seed": params.seed,
                       "step": int(runner._exactsample_bridge["step"]),
                       "prior_effective_seed": prior,
                       "native_seed": int(runner.sampler.sampling_states.seeds.np[int(request_index)]),
                       "effective_seed": runner._exactsample_bridge["request_seeds"][str(request_id)]})

    def finish(scheduler_output):
        preempted = sorted(scheduler_output.preempted_req_ids or [])
        finished = sorted(scheduler_output.finished_req_ids)
        if preempted or finished:
            events.append({"event": "scheduler_cleanup", "preempted": preempted,
                           "finished": finished,
                           "before_seeds": dict(runner._exactsample_bridge["request_seeds"])})
        result = original_finish(scheduler_output)
        if preempted or finished:
            events[-1]["after_seeds"] = dict(runner._exactsample_bridge["request_seeds"])
        return result

    bridge.register_request = register
    runner.finish_requests = finish
    return {"installed": True, "scope": "CPU observations only; public low KV causes scheduler preemption"}


def read_observer(worker):
    return {"events": list(worker._exactsample_lifecycle_events),
            "remaining_seeds": dict(worker.model_runner._exactsample_bridge["request_seeds"])}


def validate_cancelled(cancelled, max_tokens):
    """Require an active abort with no token delivered after its recorded prefix."""
    tokens = cancelled.get("token_ids", [])
    if (cancelled.get("abort_called") is not True
            or cancelled.get("finish_reason") != "abort"
            or not tokens
            or len(tokens) >= max_tokens
            or tokens != cancelled.get("tokens_at_abort")):
        raise AssertionError("abort must finish as abort before the limit with exactly its nonempty pre-abort prefix")


async def run(args, record):
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    from transformers import AutoTokenizer

    if vllm.__version__ != "0.31.0":
        raise RuntimeError("lifecycle test requires pinned vLLM 0.31.0")
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True, trust_remote_code=False)
    prompt = tokenizer.encode("Describe the small river and its trees in a calm, simple story. " * 30,
                              add_special_tokens=False)[:args.prompt_tokens]
    record["prompt_token_ids"] = prompt
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=str(args.target.resolve()), tokenizer=str(args.target.resolve()),
        dtype="bfloat16", trust_remote_code=False, enforce_eager=True,
        max_model_len=256, max_num_seqs=4, max_num_batched_tokens=512,
        gpu_memory_utilization=.28, kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        enable_prefix_caching=False, disable_log_stats=False, seed=0,
        speculative_config={"model": str(args.draft.resolve()), "method": "draft_model",
                            "num_speculative_tokens": 4, "draft_sample_method": "greedy"},
        worker_extension_cls="integration.vllm_worker_probe.WorkerProbe"))
    root = Path(vllm.__file__).parent
    record["model_runner_sha256"] = hashlib.sha256((root / "v1/worker/gpu/model_runner.py").read_bytes()).hexdigest()
    try:
        record["observer"] = await engine.collective_rpc("exactsample_install_lifecycle_observer")

        async def generate(identity, seed, max_tokens, *, abort_after_first=False, input_tokens=None):
            params = SamplingParams(seed=seed, temperature=.6, top_k=20, top_p=.95,
                                    max_tokens=max_tokens, ignore_eos=True,
                                    output_kind=RequestOutputKind.DELTA, stream_interval=1)
            result = {"request_id": identity, "seed": seed, "token_ids": [], "abort_called": False}
            async for output in engine.generate({"prompt_token_ids": input_tokens or prompt}, params, identity):
                for completion in output.outputs:
                    result["token_ids"].extend(completion.token_ids)
                    if completion.finish_reason is not None:
                        result["finish_reason"] = completion.finish_reason
                        result["stop_reason"] = completion.stop_reason
                if abort_after_first and result["token_ids"] and not result["abort_called"]:
                    result["tokens_at_abort"] = list(result["token_ids"])
                    await engine.abort(identity)
                    result["abort_called"] = True
            return result

        record["cancelled"] = await generate("cancel-me", None, args.max_tokens, abort_after_first=True)
        validate_cancelled(record["cancelled"], args.max_tokens)
        # Reuse the external ID after cancellation. vLLM assigns a new internal
        # ID; the old seed must be removed before this independent admission.
        record["reused_external_id"] = await generate("cancel-me", 9191, 4)
        if record["reused_external_id"].get("finish_reason") != "length":
            raise AssertionError("request after cancellation failed")

        record["stress"] = await asyncio.gather(*(
            generate(f"stress-{index}", None, args.max_tokens) for index in range(4)))
        if any(item.get("finish_reason") != "length" or len(item["token_ids"]) != args.max_tokens
               for item in record["stress"]):
            raise AssertionError("stress requests did not complete their requested lengths")
        # A later request lets the existing scheduler send final cleanup; this
        # does not synthesize cleanup or mutate bridge identity state.
        record["drain"] = await generate("drain-cleanup", 1, 1, input_tokens=prompt[:8])
        observed, = await engine.collective_rpc("exactsample_read_lifecycle_observer")
        record["lifecycle"] = observed
        registers = defaultdict(list)
        preempted = set()
        for event in observed["events"]:
            if event["event"] == "register":
                registers[event["request_id"]].append(event)
            else:
                preempted.update(event["preempted"])
                for identity in event["preempted"]:
                    if event["before_seeds"].get(identity) != event["after_seeds"].get(identity):
                        raise AssertionError("preemption changed or deleted the admission key")
                for identity in event["finished"]:
                    if identity in event["after_seeds"]:
                        raise AssertionError("true finish did not clear the admission key")
        resumed = {identity: entries for identity, entries in registers.items()
                   if identity in preempted and len(entries) > 1}
        if not resumed:
            raise AssertionError("no actual scheduler preemption and re-admission; this is not a passing stress test")
        for identity, entries in resumed.items():
            if len({entry["effective_seed"] for entry in entries}) != 1:
                raise AssertionError(f"admission key changed after recompute: {identity}")
        record["preempted_request_ids"] = sorted(preempted)
        record["resumed_request_ids"] = sorted(resumed)
        record["native_seed_changed_on_resume"] = {
            identity: len({entry["native_seed"] for entry in entries}) > 1
            for identity, entries in resumed.items()}
        cancelled_admissions = [entry for identity, entries in registers.items()
                                if identity.startswith("cancel-me-") for entry in entries]
        if (len(cancelled_admissions) != 2
                or cancelled_admissions[0]["request_id"] == cancelled_admissions[1]["request_id"]
                or cancelled_admissions[1]["effective_seed"] != 9191
                or cancelled_admissions[1]["prior_effective_seed"] is not None):
            raise AssertionError("external-ID reuse retained the cancelled identity")
        record["cancelled_and_reused_admissions"] = cancelled_admissions
        verifies = [json.loads(line) for line in args.output.with_suffix(".trace.jsonl").read_text().splitlines()
                    if json.loads(line).get("event") == "verify"]
        record["verified_resumed_steps"] = {}
        for identity, entries in resumed.items():
            after_resume = []
            for event in verifies:
                if identity in event["requests"]:
                    index = event["requests"].index(identity)
                    if event["seeds_uint64"][index] != (entries[0]["effective_seed"] & ((1 << 64) - 1)) or event["status"][index] != 0:
                        raise AssertionError("actual resumed verifier received a changed key or failed status")
                    if event["step"] >= entries[1]["step"]:
                        after_resume.append(event["step"])
            if not after_resume:
                raise AssertionError("no real exactsample verifier step observed after resume")
            record["verified_resumed_steps"][identity] = after_resume
        def external(internal):
            found = [item for item in record["stress"] if internal == item["request_id"] or (
                internal.startswith(item["request_id"] + "-") and len(internal) == len(item["request_id"]) + 9)]
            if len(found) != 1:
                raise AssertionError("ambiguous stress request identity")
            return found[0]
        baseline = []
        for index, (identity, entries) in enumerate(resumed.items()):
            result = await generate(f"isolated-{index}", entries[0]["effective_seed"], args.max_tokens)
            stressed = external(identity)
            divergence = next((i for i, pair in enumerate(zip(result["token_ids"], stressed["token_ids"]))
                               if pair[0] != pair[1]), None)
            equal = result["token_ids"] == stressed["token_ids"]
            baseline.append({"internal_request_id": identity, "isolated": result,
                             "tokens_equal": equal, "first_divergence": divergence})
        record["isolated_recompute_comparisons"] = baseline
        if not all(item["tokens_equal"] for item in baseline):
            raise AssertionError("recomputed sequence differs from isolated execution with the same admission seed")
        if any(identity in observed["remaining_seeds"] for identity in registers if identity.startswith(("stress-", "cancel-me-"))):
            raise AssertionError("completed or cancelled request identity remained after cleanup")
        record["state"] = "passed"
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=134217728)
    parser.add_argument("--prompt-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.prompt_tokens + args.max_tokens > 252 or min(args.prompt_tokens, args.max_tokens) < 1:
        parser.error("positive prompt/output lengths must leave room for drafts within max-model-len 256")
    trace = args.output.with_suffix(".trace.jsonl")
    if args.output.exists() or trace.exists():
        parser.error("refusing to overwrite lifecycle evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(VLLM_USE_V2_MODEL_RUNNER="1", EXACTSAMPLE_DECODE="1", EXACTSAMPLE_MODE="coupled",
                      VLLM_BATCH_INVARIANT="1", EXACTSAMPLE_TRACE_PATH=str(trace.resolve()),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.pop("EXACTSAMPLE_CAPTURE_DIR", None)
    record = {"schema": "exactsample.vllm_lifecycle.v1", "state": "started",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "scope": "public abort and genuine scheduler preemption under small fixed KV; observation-only worker hooks"}
    observer = GpuObserver()
    observer.start()
    try:
        asyncio.run(asyncio.wait_for(run(args, record), timeout=240))
    except BaseException as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["gpu_observation"] = observer.stop()
        with args.output.open("x") as output:
            json.dump(record, output, indent=2)
    print(json.dumps({key: value for key, value in record.items() if key != "gpu_observation"}, indent=2))


if __name__ == "__main__":
    main()
