"""Run a pinned local vLLM MRv2 batch and retain actual exactsample/JSON evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from gpu_observer import GpuObserver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--exactsample", choices=("on", "off"), default="on")
    parser.add_argument("--mode", choices=("coupled", "ratio"), default="coupled")
    parser.add_argument("--no-spec", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--seed-only", action="store_true", help="single seeded request for original/off diagnosis")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.repeats != 1 and not args.seed_only):
        parser.error("repeats must be positive; multiple repeats require --seed-only")
    for directory in (args.target, args.draft):
        if not directory.is_dir():
            parser.error(f"local model directory does not exist: {directory}")
    trace = args.output.with_suffix(".trace.jsonl")
    if args.output.exists() or trace.exists() or (args.capture_dir and args.capture_dir.exists()):
        parser.error("refusing to overwrite existing evidence paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update({"VLLM_USE_V2_MODEL_RUNNER": "1", "EXACTSAMPLE_DECODE": "1" if args.exactsample == "on" else "0",
                       "EXACTSAMPLE_MODE": args.mode, "EXACTSAMPLE_TRACE_PATH": str(trace.resolve()),
                       "VLLM_BATCH_INVARIANT": "1" if args.batch_invariant else "0",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if args.capture_dir:
        os.environ["EXACTSAMPLE_CAPTURE_DIR"] = str(args.capture_dir.resolve())
    else:
        os.environ.pop("EXACTSAMPLE_CAPTURE_DIR", None)
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    if vllm.__version__ != "0.31.0":
        raise RuntimeError(f"requires locked vLLM 0.31.0, found {vllm.__version__}")
    runner_source = Path(vllm.__file__).parent / "v1/worker/gpu/model_runner.py"
    if args.exactsample == "on" and "exactsample_output = try_sample(" not in runner_source.read_text():
        raise RuntimeError("the exactsample local fork patch has not been applied")
    record = {"schema": "exactsample.vllm_smoke.v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "vllm": vllm.__version__, "exactsample": args.exactsample, "mode": args.mode,
              "speculative": not args.no_spec, "source_file": str(runner_source),
              "source_sha256": hashlib.sha256(runner_source.read_bytes()).hexdigest(),
              "target": str(args.target.resolve()), "draft": str(args.draft.resolve()),
              "batch_invariant": args.batch_invariant, "seed_only": args.seed_only,
              "repeats": args.repeats, "gpu_memory_utilization": args.gpu_memory_utilization,
              "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
              "trace_file": str(trace.resolve()), "state": "started"}
    observer = GpuObserver()
    observer.start()
    try:
        speculative = None if args.no_spec else {
            "model": str(args.draft.resolve()), "method": "draft_model",
            "num_speculative_tokens": 4, "draft_sample_method": "greedy",
        }
        llm = LLM(model=str(args.target.resolve()), tokenizer=str(args.target.resolve()),
                  trust_remote_code=False, dtype="bfloat16", max_model_len=256,
                  max_num_seqs=4, max_num_batched_tokens=256, enforce_eager=True,
                  gpu_memory_utilization=args.gpu_memory_utilization, seed=0,
                  kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                  tensor_parallel_size=1, speculative_config=speculative,
                  disable_log_stats=True)
        schema = {"type": "object", "properties": {"answer": {"type": "integer", "enum": [7]}},
                  "required": ["answer"], "additionalProperties": False}
        prompts = ["Complete briefly: The sky is", "Reply with a short sentence about rain.",
                   'Output only a JSON object whose "answer" is the integer 7.']
        shared = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "max_tokens": args.max_tokens}
        params = [SamplingParams(**shared), SamplingParams(**shared, seed=7),
                  SamplingParams(**shared, seed=19, structured_outputs=StructuredOutputsParams(json=schema))]
        if args.seed_only:
            prompts = [prompts[1]]
            params = [SamplingParams(**shared, seed=7, ignore_eos=True)]
        record["outputs"] = []
        for repetition in range(args.repeats):
            outputs = llm.generate(prompts, params, use_tqdm=False)
            record["outputs"].extend({"request_id": output.request_id, "prompt": prompts[index],
                               "repetition": repetition,
                               "text": output.outputs[0].text,
                               "token_ids": list(output.outputs[0].token_ids),
                               "finish_reason": output.outputs[0].finish_reason}
                              for index, output in enumerate(outputs))
        if any(output["finish_reason"] == "error" for output in record["outputs"]):
            raise AssertionError("a smoke request ended with an engine error")
        if not args.seed_only:
            parsed = json.loads(record["outputs"][2]["text"])
            if parsed != {"answer": 7}:
                raise AssertionError(f"unexpected constrained output: {parsed}")
        events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        verified = [event for event in events if event["event"] == "verify"]
        if args.exactsample == "on":
            if any(status for event in verified for status in event["status"]):
                raise AssertionError("a smoke step returned a non-OK exactsample status")
            if not verified or (not args.seed_only and not any(event["grammar_rows"] > 0 for event in verified)):
                raise AssertionError("no evidence of exactsample consuming live logits and grammar bitmasks")
            if not args.no_spec and not any(any(event["draft_counts"]) for event in verified):
                raise AssertionError("no exactsample speculative verification step ran")
        record.update(state="passed", valid_json=None if args.seed_only else True, exactsample_steps=len(verified),
                      fallback_events=[event for event in events if event["event"] == "fallback"])
    except Exception as error:
        record.update(state="failed", error_type=type(error).__name__, error=str(error))
        record["gpu_observation"] = observer.stop()
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(record, output, ensure_ascii=False, indent=2)
        raise
    record["gpu_observation"] = observer.stop()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
