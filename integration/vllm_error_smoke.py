"""Explicit fault injection: one all-masked request beside a healthy request.

This installs a temporary worker-local wrapper using vLLM's public worker
extension and named collective_rpc hook. No product fault-injection switch exists. The
wrapper restores the original function as soon as one mixed batch is injected.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

if __package__:
    from .gpu_observer import GpuObserver
else:
    from gpu_observer import GpuObserver


def install_fault(worker, min_draft):
    import numpy as np
    from types import SimpleNamespace
    from exactsample import vllm_bridge as bridge
    original = bridge.try_sample
    bridge._exactsample_test_injection = None

    def one_fault(runner, logits, batch, grammar):
        counts = np.diff(batch.cu_num_logits_np) - 1
        eligible = np.flatnonzero(counts >= 1 if min_draft else counts == 0)
        if batch.num_reqs < 2 or not len(eligible):
            return original(runner, logits, batch, grammar)
        request = int(eligible[0])
        if grammar is not None and grammar.structured_output_request_ids:
            mask = bridge.grammar_mask_rows(batch.req_ids, batch.cu_num_logits_np,
                                            grammar.structured_output_request_ids,
                                            grammar.grammar_bitmask, logits.shape[1])
        else:
            mask = np.full((logits.shape[0], (logits.shape[1] + 31) // 32), -1, np.int32)
        start, end = map(int, batch.cu_num_logits_np[request:request + 2])
        mask[start:end] = 0
        bridge._exactsample_test_injection = {"request_id": batch.req_ids[request],
            "other_requests": [key for index, key in enumerate(batch.req_ids) if index != request],
            "draft_count": int(counts[request]),
            "step": int(runner._exactsample_bridge["step"])}
        bridge.try_sample = original
        return original(runner, logits, batch, SimpleNamespace(
            structured_output_request_ids=list(batch.req_ids), grammar_bitmask=mask))

    bridge.try_sample = one_fault
    return {"installed": True, "min_draft": min_draft}


def get_fault(worker):
    from exactsample import vllm_bridge
    return vllm_bridge._exactsample_test_injection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-draft", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.28)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    args = parser.parse_args(argv)
    trace = args.output.with_suffix(".trace.jsonl")
    if args.output.exists() or trace.exists():
        parser.error("refusing to overwrite existing evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(VLLM_USE_V2_MODEL_RUNNER="1", EXACTSAMPLE_DECODE="1", EXACTSAMPLE_MODE="coupled",
                      EXACTSAMPLE_TRACE_PATH=str(trace.resolve()), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.pop("EXACTSAMPLE_CAPTURE_DIR", None)
    from vllm import LLM, SamplingParams
    record = {"schema": "exactsample.vllm_request_error.v1", "state": "started",
              "started_at": datetime.now(timezone.utc).isoformat(), "min_draft": args.min_draft,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "kv_cache_memory_bytes": args.kv_cache_memory_bytes}
    observer = GpuObserver()
    observer.start()
    try:
        llm = LLM(model=str(args.target.resolve()), tokenizer=str(args.target.resolve()),
                  trust_remote_code=False, dtype="bfloat16", max_model_len=256,
                  max_num_seqs=4, max_num_batched_tokens=256, enforce_eager=True,
                  gpu_memory_utilization=args.gpu_memory_utilization, seed=0,
                  kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                  worker_extension_cls="integration.vllm_worker_probe.WorkerProbe",
                  tensor_parallel_size=1, disable_log_stats=True, speculative_config={
                      "model": str(args.draft.resolve()), "method": "draft_model",
                      "num_speculative_tokens": 4, "draft_sample_method": "greedy"})
        llm.collective_rpc("exactsample_install_fault", args=(args.min_draft,))
        params = [SamplingParams(temperature=.6, top_k=20, top_p=.95, seed=seed,
                                 max_tokens=12, ignore_eos=True) for seed in (7, 19)]
        outputs = llm.generate(["Complete briefly: The sky is", "Write a sentence about rain."],
                               params, use_tqdm=False)
        injection, = llm.collective_rpc("exactsample_get_fault")
        assert injection is not None, "no mixed batch met the injection condition"
        record["injection"] = injection
        record["outputs"] = [{"request_id": item.request_id,
                               "token_ids": list(item.outputs[0].token_ids),
                               "finish_reason": item.outputs[0].finish_reason,
                               "stop_reason": item.outputs[0].stop_reason}
                              for item in outputs]
        by_id = {item["request_id"]: item for item in record["outputs"]}
        def external_id(internal):
            # Pinned InputProcessor.assign_request_id appends '-' plus eight
            # random characters; output_processor returns the external ID.
            candidates = [key for key in by_id if internal == key or (
                internal.startswith(key + "-") and len(internal) == len(key) + 9)]
            if len(candidates) != 1:
                raise AssertionError(f"ambiguous internal/external ID mapping: {internal}")
            return candidates[0]
        record["internal_to_external_ids"] = {key: external_id(key) for key in
            [injection["request_id"], *injection["other_requests"]]}
        bad = by_id[external_id(injection["request_id"])]
        assert bad["finish_reason"] == "error", bad
        assert bad["stop_reason"] == "exactsample_status:1", bad
        if args.min_draft == 0:
            assert bad["token_ids"] == [], bad
        for identity in injection["other_requests"]:
            output = by_id[external_id(identity)]
            assert output["finish_reason"] == "length", output
            assert len(output["token_ids"]) == 12, output
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        verifies = [item for item in events if item["event"] == "verify"]
        failed = next(item for item in verifies if item["step"] == injection["step"])
        index = failed["requests"].index(injection["request_id"])
        assert failed["status"][index] == 1 and all(token == -1 for token in failed["tokens"][index]), failed
        assert all(status == 0 for i, status in enumerate(failed["status"]) if i != index), failed
        committed_before = []
        for item in verifies:
            if item["step"] >= injection["step"] or injection["request_id"] not in item["requests"]:
                continue
            i = item["requests"].index(injection["request_id"])
            committed_before += item["tokens"][i][:item["accepted_len"][i] + 1]
        assert bad["token_ids"] == committed_before, (bad, committed_before)
        record.update(state="passed", status=1, failed_step_tokens_committed=0,
                      healthy_requests_completed=len(injection["other_requests"]), trace_file=str(trace))
    except Exception as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error))
        record["gpu_observation"] = observer.stop()
        with args.output.open("x") as output:
            json.dump(record, output, indent=2)
        raise
    record["gpu_observation"] = observer.stop()
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
