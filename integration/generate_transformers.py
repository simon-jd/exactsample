"""Explicit local-model fallback: real Transformers forwards -> exactsample CUDA.

This is a single-request, intentionally unoptimized generation loop. It does
not claim vLLM integration, JSON grammar support, or service performance.
Models must already exist in local directories; no downloads are requested.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--target", type=Path, required=True, help="local target model directory")
    result.add_argument("--draft", type=Path, required=True, help="local draft model directory")
    result.add_argument("--prompt", required=True)
    result.add_argument("--max-new-tokens", type=int, default=32)
    result.add_argument("--draft-tokens", type=int, default=4)
    result.add_argument("--mode", choices=("coupled", "ratio"), default="coupled")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--stream-id", type=int, default=0)
    result.add_argument("--temperature", type=float, default=0.6)
    result.add_argument("--top-k", type=int, default=20)
    result.add_argument("--top-p", type=float, default=0.95)
    result.add_argument("--min-p", type=float, default=0.0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--dispatch", choices=("auto", "reference", "fused"), default="auto")
    result.add_argument("--output", type=Path, required=True, help="new JSON evidence file")
    result.add_argument("--save-logits", type=Path, help="new directory for per-step NPZ inputs and JSON metadata")
    return result


def _hash_array(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")


def run(args, record):
    # Heavy dependencies are imported only on an explicit execution. --help
    # works without CUDA, torch, Transformers, or model files.
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from exactsample import Status
    from exactsample.cuda import CudaRequestParams, CudaRngKey, plan, verify

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required; this script has no CPU substitute")
    if torch.cuda.get_device_capability(device) not in ((8, 6), (8, 9)):
        raise RuntimeError("exactsample is built and tested for sm_86 and sm_89 only")
    record["environment"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True, trust_remote_code=False)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft, local_files_only=True, trust_remote_code=False)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target and draft token-to-ID vocabularies must match exactly")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16).to(device).eval()
    vocab = int(target.config.vocab_size)
    if int(draft.config.vocab_size) != vocab:
        raise ValueError("target and draft output vocab sizes must match")
    record["models"] = {
        "target": {"directory": str(args.target.resolve()), "config": target.config.to_dict()},
        "draft": {"directory": str(args.draft.resolve()), "config": draft.config.to_dict()},
    }
    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    if ids.shape[1] == 0:
        raise ValueError("prompt must encode to at least one token")
    prompt_length = int(ids.shape[1])
    if prompt_length + args.max_new_tokens >= (1 << 32):
        raise ValueError("absolute source positions exceed uint32 RNG address space")
    record["prompt_token_ids"] = ids[0].tolist()
    decode_plan = plan(max_rows=args.draft_tokens + 1, max_requests=1,
                       max_draft=args.draft_tokens, vocab=vocab, device=device)
    out = decode_plan.allocate_output()
    params = CudaRequestParams(
        temperature=torch.tensor([args.temperature], dtype=torch.float64, device=device),
        top_k=torch.tensor([args.top_k], dtype=torch.int64, device=device),
        top_p=torch.tensor([args.top_p], dtype=torch.float64, device=device),
        min_p=torch.tensor([args.min_p], dtype=torch.float64, device=device),
        draft_kind=torch.zeros(1, dtype=torch.int64, device=device),
    )
    # The CUDA ABI carries all uint64 seed bits in an int64 tensor.
    signed_seed = args.seed if args.seed < (1 << 63) else args.seed - (1 << 64)
    seed = torch.tensor([signed_seed], dtype=torch.int64, device=device)
    stream_id = torch.tensor([args.stream_id], dtype=torch.int64, device=device)
    padded_drafts = torch.full((1, args.draft_tokens), -1, dtype=torch.int64, device=device)
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    generated = []
    if args.save_logits:
        args.save_logits.mkdir(parents=True, exist_ok=False)
        record["saved_logits_directory"] = str(args.save_logits.resolve())
    started = time.perf_counter()
    with torch.inference_mode():
        while len(generated) < args.max_new_tokens:
            count = min(args.draft_tokens, args.max_new_tokens - len(generated) - 1)
            draft_context = ids
            proposals = []
            for _ in range(count):
                proposal_logits = draft(input_ids=draft_context, use_cache=False).logits[0, -1]
                if torch.isnan(proposal_logits).any().item() or torch.isposinf(proposal_logits).any().item():
                    raise RuntimeError("draft model produced NaN/+inf")
                if not torch.isfinite(proposal_logits).any().item():
                    raise RuntimeError("draft model produced no finite proposal")
                proposal = torch.argmax(proposal_logits).reshape(1, 1)
                proposals.append(int(proposal.item()))
                draft_context = torch.cat((draft_context, proposal), dim=1)
            target_output = target(input_ids=draft_context, use_cache=False)
            logits = target_output.logits[0, -(count + 1):].contiguous()
            del target_output
            first_position = int(ids.shape[1]) - 1
            positions = torch.arange(first_position, first_position + count + 1,
                                     dtype=torch.int64, device=device)
            offsets = torch.tensor([0, count + 1], dtype=torch.int64, device=device)
            padded_drafts.fill_(-1)
            if count:
                padded_drafts[0, :count] = torch.tensor(proposals, dtype=torch.int64, device=device)
            result = verify(logits, offsets, padded_drafts, params=params,
                            rng=CudaRngKey(seed=seed, stream_id=stream_id, positions=positions),
                            mode=args.mode, plan=decode_plan, out=out, dispatch=args.dispatch)
            status = int(result.status[0].item())
            accepted = int(result.accepted_len[0].item())
            raw_tokens = result.tokens[0].tolist()
            # NumPy has no native bf16. Float32 stores bf16/f16/f32 exactly;
            # preserve float64 as float64 if a model explicitly returns it.
            storage_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            saved_logits = logits.to(dtype=storage_dtype).cpu().numpy()
            step = {
                "index": len(record["steps"]), "input_token_count": int(ids.shape[1]),
                "positions": positions.tolist(), "draft_ids": proposals,
                "status": status, "status_name": Status(status).name,
                "accepted_len": accepted, "tokens": raw_tokens,
                "path_used": int(result.path_used[0].item()),
                "logits_dtype": str(logits.dtype), "saved_logits_dtype": str(saved_logits.dtype),
                "logits_sha256": _hash_array(saved_logits),
            }
            record["steps"].append(step)
            if args.save_logits:
                saved_path = args.save_logits / f"step{step['index']:04d}.npz"
                arrays = {
                    "target_logits": saved_logits,
                    "positions": positions.cpu().numpy(),
                    "draft_ids": padded_drafts.cpu().numpy().copy(),
                    "cu_rows": offsets.cpu().numpy(),
                    "seeds": np.array([args.seed], dtype=np.uint64),
                    "stream_ids": np.array([args.stream_id], dtype=np.int64),
                    "temperature": np.array([args.temperature], dtype=np.float64),
                    "top_k": np.array([args.top_k], dtype=np.int64),
                    "top_p": np.array([args.top_p], dtype=np.float64),
                    "min_p": np.array([args.min_p], dtype=np.float64),
                    "draft_kind": np.array([0], dtype=np.int64),
                }
                with saved_path.open("xb") as saved_file:
                    np.savez_compressed(saved_file, **arrays)
                step["saved_logits"] = str(saved_path.resolve())
                step["saved_logits_file_sha256"] = hashlib.sha256(saved_path.read_bytes()).hexdigest()
                _write_record(saved_path.with_suffix(".json"), {
                    "schema": "exactsample.logits.v1", "source": "real Transformers model forwards",
                    "mode": args.mode, "step": step,
                    "model_revisions": {
                        "target": getattr(target.config, "_commit_hash", None),
                        "draft": getattr(draft.config, "_commit_hash", None),
                    },
                    "model_directories": {"target": str(args.target.resolve()), "draft": str(args.draft.resolve())},
                    "original_logits_dtype": str(logits.dtype),
                    "stored_logits_dtype": str(saved_logits.dtype),
                    "conversion": f"exact {logits.dtype} values represented in {saved_logits.dtype}",
                    "array_hashes": {key: _hash_array(array) for key, array in arrays.items()},
                    "environment": record["environment"],
                })
            if status != int(Status.OK):
                raise RuntimeError(f"exactsample rejected request: {Status(status).name}; no step tokens committed")
            committed = raw_tokens[:accepted + 1]
            if not committed or any(token < 0 or token >= vocab for token in committed):
                raise RuntimeError("exactsample returned invalid successful output")
            stop = next((index + 1 for index, token in enumerate(committed) if token in eos_ids), None)
            if stop is not None:
                committed = committed[:stop]
            step["committed_tokens"] = committed
            generated.extend(committed)
            ids = torch.cat((ids, torch.tensor([committed], dtype=torch.int64, device=device)), dim=1)
            if stop is not None:
                break
    torch.cuda.synchronize(device)
    record.update({"status": "completed", "generated_token_ids": generated,
                   "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                   "elapsed_seconds_unoptimized": time.perf_counter() - started})


def main(argv=None):
    args = parser().parse_args(argv)
    for model_dir in (args.target, args.draft):
        if not model_dir.is_dir():
            raise SystemExit(f"local model directory does not exist: {model_dir}")
    if args.max_new_tokens < 1 or args.draft_tokens < 0:
        raise SystemExit("max-new-tokens must be positive and draft-tokens nonnegative")
    if not 0 <= args.seed < (1 << 64) or not 0 <= args.stream_id < (1 << 32):
        raise SystemExit("seed must be uint64 and stream-id uint32")
    for path in (args.output, args.save_logits):
        if path is not None and path.exists():
            raise SystemExit(f"refusing to overwrite existing evidence: {path}")
    record = {
        "schema": "exactsample.transformers_generation.v1", "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "integration": "standalone Transformers fallback, not vLLM",
        "position_convention": "absolute source-logit position: prompt_length - 1 + emitted_offset",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(), "steps": [],
    }
    try:
        run(args, record)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write_record(args.output, record)
        raise
    _write_record(args.output, record)
    print(record["generated_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
