"""Carry exactsample request statuses through vLLM's existing asynchronous D2H event."""
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput, async_copy_to_np, stream
from .vllm_protocol import ExactSampleModelRunnerOutput


def write_diagnostics(diagnostics, copied, status, tokens):
    """Write explicitly requested evidence from already completed CPU copies."""
    record = dict(diagnostics["record"])
    record.update(status=status.tolist(), tokens=tokens.tolist(),
                  accepted_len=copied["accepted_len"].tolist(),
                  path_used=copied["path_used"].tolist(), positions=copied["positions"].tolist())
    trace_path = diagnostics.get("trace_path")
    if trace_path:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        item = {"schema": "exactsample.vllm_trace.v1", "pid": os.getpid(), "time_ns": time.time_ns(), **record}
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(item) + "\n")
    if "target_logits" not in copied:
        return
    directory = Path(diagnostics["capture_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {**diagnostics["cpu_arrays"], **{key: copied[key] for key in ("target_logits", "draft_ids", "positions")}}
    path = directory / f"worker{os.getpid()}-step{record['step']:04d}.npz"
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)
    metadata = {"schema": "exactsample.logits.v1", "source": "vLLM v0.31.0 live Model Runner V2 forwards",
                "mode": record["mode"], "original_logits_dtype": record["raw_logits_dtype"],
                "stored_logits_dtype": str(arrays["target_logits"].dtype),
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "step": record,
                "environment": diagnostics["provenance"]}
    with path.with_suffix(".json").open("x", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)


class ExactSampleAsyncOutput(AsyncOutput):
    """Append status to the original copy stream and reuse its completion event.

    Construction never reads a numerical status on the host. `get_output`
    delegates to the original, single event synchronization before consuming
    status. Sampler status and diagnostic tensors are independent snapshots,
    held alive by the SamplerOutput until this copy completes.
    """

    def __init__(self, *, model_runner_output, sampler_output, num_sampled_tokens,
                 main_stream, copy_stream, check_ep_fault, pending_aux_output):
        super().__init__(model_runner_output=model_runner_output, sampler_output=sampler_output,
                         num_sampled_tokens=num_sampled_tokens, main_stream=main_stream,
                         copy_stream=copy_stream, check_ep_fault=check_ep_fault,
                         pending_aux_output=pending_aux_output)
        self._exactsample_diagnostics = getattr(sampler_output, "exactsample_diagnostics", None)
        self._exactsample_copied = {}
        with stream(copy_stream, main_stream):
            # The superclass already made this copy stream wait for main_stream.
            self._exactsample_status = async_copy_to_np(sampler_output.exactsample_status)
            if self._exactsample_diagnostics is not None:
                self._exactsample_copied = {name: async_copy_to_np(tensor)
                                    for name, tensor in self._exactsample_diagnostics["gpu_arrays"].items()}
            # Re-record the SAME event after all exactsample copies; no second host wait.
            self.copy_event.record(copy_stream)

    def get_output(self):
        result = super().get_output()
        if self._exactsample_status.shape != (len(result.req_ids),):
            raise RuntimeError("exactsample status length differs from model runner request identities")
        statuses = dict(zip(result.req_ids, map(int, self._exactsample_status)))
        # Defense in depth: a failed request can never carry a commit token even
        # if a future caller violates the GPU num_sampled=0 convention.
        for request_id, status in statuses.items():
            if status:
                result.sampled_token_ids[result.req_id_to_index[request_id]] = []
        if self._exactsample_diagnostics is not None:
            write_diagnostics(self._exactsample_diagnostics, self._exactsample_copied,
                              self._exactsample_status, self.sampled_token_ids)
        values = {field.name: getattr(result, field.name) for field in fields(ModelRunnerOutput)}
        return ExactSampleModelRunnerOutput(**values, exactsample_status=statuses)
