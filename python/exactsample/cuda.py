"""Allocation-free CUDA decode calls with explicit, reusable workspace ownership.

Creating a plan, output, and parameter tensors is a setup operation. ``verify``
uses those existing device tensors and never copies parameters from the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import threading
from typing import Any


@dataclass(frozen=True)
class CudaRequestParams:
    temperature: Any  # CUDA float64 [R]
    top_k: Any  # CUDA int64 [R], 0 disables
    top_p: Any  # CUDA float64 [R]
    min_p: Any  # CUDA float64 [R]
    draft_kind: Any  # CUDA int64 [R], 0=one_hot, 1=full


@dataclass(frozen=True)
class CudaRngKey:
    seed: Any  # CUDA int64 [R]; signed storage of uint64 key bits
    stream_id: Any  # CUDA int64 [R], values in [0, 2**32)
    positions: Any  # CUDA int64 [rows], values in [0, 2**32)


@dataclass
class CudaOutput:
    tokens: Any
    accepted_len: Any
    status: Any
    path_used: Any
    num_requests: int = 0


def load_library(path: str | os.PathLike[str] | None = None) -> None:
    """Load the separately built TORCH_LIBRARY shared object (no JIT build)."""
    import torch

    if hasattr(torch.ops.exactsample, "verify_out"):
        return
    requested = path or os.environ.get("EXACTSAMPLE_LIBRARY")
    library = Path(requested) if requested else Path(__file__).with_name("exactsample_cuda.so")
    if not library.is_file():
        raise RuntimeError(
            f"exactsample CUDA library is missing: {library}. Build with CMake and set "
            "EXACTSAMPLE_LIBRARY, or install exactsample_cuda.so alongside exactsample/cuda.py."
        )
    torch.ops.load_library(str(library.resolve()))


class CudaPlan:
    """One workspace. Calls on one stream queue safely; concurrent streams fail.

    Inputs and outputs must not be mutated until ``synchronize`` or normal CUDA
    stream ordering establishes completion. Tensor allocator lifetimes are
    recorded on the consuming stream. CUDA graph capture is not supported in v1.
    """

    def __init__(self, max_rows: int, max_requests: int, max_draft: int, vocab: int,
                 *, device: Any = "cuda", caps: dict | None = None, reference: bool = False):
        import torch

        if caps:
            raise ValueError("caps is reserved; v1 supports the default capabilities only")
        for name, value in (("max_rows", max_rows), ("max_requests", max_requests),
                            ("max_draft", max_draft), ("vocab", vocab)):
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "vocab" else 0):
                raise ValueError(f"{name} must be a nonnegative integer (vocab must be positive)")
            if value >= 2**31:
                raise ValueError(f"{name} exceeds kernel int32 indexing")
        self.max_rows, self.max_requests = max_rows, max_requests
        # Fused path: top-k sets up to this size are processed as an exact FP64
        # list. Tests lower it to exercise the full-row path on small inputs.
        self.list_max = 256
        self.max_draft, self.vocab = max_draft, vocab
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CudaPlan requires a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        # The fused path needs 32 bytes per row; dispatch="reference" also needs
        # 20 bytes per row and vocabulary entry, so it is allocated only on request.
        self.reference = reference
        # Matches exactsample::workspace_size: FUSED_ROW_BYTES = 12288 of per-row scratch.
        self.workspace_bytes = 8 + max_rows * (32 + max(12288, 20 * vocab if reference else 0))
        self.workspace = torch.zeros(self.workspace_bytes, dtype=torch.uint8, device=self.device)
        self._lock = threading.Lock()
        self._event = torch.cuda.Event(enable_timing=False)
        self._last_stream = None
        self._references = ()
        # Instantiate the event during setup, outside the decode hot path.
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()

    def allocate_output(self) -> CudaOutput:
        import torch

        return CudaOutput(
            torch.empty((self.max_requests, self.max_draft + 1), device=self.device, dtype=torch.int64),
            torch.empty(self.max_requests, device=self.device, dtype=torch.int32),
            torch.empty(self.max_requests, device=self.device, dtype=torch.int32),
            torch.empty(self.max_requests, device=self.device, dtype=torch.int32),
        )

    def fallbacks(self) -> int:
        """FP64 recomputations made by the fused path in the last call (synchronizes)."""
        import torch

        self.synchronize()
        return int(self.workspace[4:8].view(torch.int32).item())

    def synchronize(self) -> None:
        """Wait for this plan's last call, allowing workspace reuse on any stream."""
        with self._lock:
            self._event.synchronize()
            self._references = ()
            self._last_stream = None


def plan(max_rows: int, max_requests: int, max_draft: int, vocab: int,
         caps: dict | None = None, *, device: Any = "cuda", reference: bool = False) -> CudaPlan:
    return CudaPlan(max_rows, max_requests, max_draft, vocab, device=device, caps=caps, reference=reference)


def verify(target_logits, cu_rows, draft_ids, *, params: CudaRequestParams,
           rng: CudaRngKey, plan: CudaPlan, out: CudaOutput,
           draft_logits=None, mask=None, mode: str = "ratio", dispatch: str = "auto",
           top_k_top_p: bool = True) -> CudaOutput:
    """Verify ragged proposals in existing buffers and return ``out``.

    All inputs must already be CUDA tensors. ``draft_logits`` are log weights of
    the *actual* proposal distribution, after its own processors and filters;
    no target filters are applied to q. Bonus q rows are ignored. Only the first
    ``out.num_requests`` output rows belong to the call; capacity tails are not
    touched. Invalid device values yield a per-request status asynchronously.

    ``top_k_top_p=False`` tells the kernels that no request has a top-k below
    the vocabulary or a top-p below 1, which a caller that keeps its sampling
    parameters on the host knows (vLLM does): the three kernels that only serve
    those filters are then not launched. It never changes a result; a request
    that does have one is processed on the full-row path.
    """
    import torch

    modes = {"ratio": 0, "coupled": 1}
    paths = {"auto": 0, "reference": 1, "fused": 2}
    if mode not in modes or dispatch not in paths:
        raise ValueError("mode must be ratio/coupled; dispatch must be auto/reference/fused")
    if dispatch == "reference" and not plan.reference:
        raise ValueError("dispatch='reference' needs a plan made with reference=True")
    if not isinstance(params, CudaRequestParams) or not isinstance(rng, CudaRngKey):
        raise TypeError("CUDA verify requires CudaRequestParams and CudaRngKey containing device tensors")
    if target_logits.ndim != 2 or cu_rows.ndim != 1 or cu_rows.numel() < 1:
        raise ValueError("target_logits must be 2D and cu_rows a nonempty 1D tensor")
    requests = cu_rows.numel() - 1
    if target_logits.shape[0] > plan.max_rows or requests > plan.max_requests or target_logits.shape[1] != plan.vocab:
        raise ValueError("input exceeds plan capacity or has a different vocabulary")
    if target_logits.device != plan.device:
        raise ValueError("input and plan must be on the same CUDA device")
    load_library()
    with torch.cuda.device(plan.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("exactsample v1 does not support CUDA graph capture")
        stream = torch.cuda.current_stream(plan.device)
        if not plan._lock.acquire(blocking=False):
            raise RuntimeError("exactsample workspace is already in use by another host call")
        try:
            if plan._last_stream is not None and plan._last_stream != stream.cuda_stream and not plan._event.query():
                raise RuntimeError("exactsample workspace is in flight on another stream; synchronize or use a separate plan")
            tensors = (
                target_logits, cu_rows, draft_ids, draft_logits, params.temperature, params.top_k,
                params.top_p, params.min_p, params.draft_kind, rng.seed, rng.stream_id,
                rng.positions, mask, plan.workspace, out.tokens, out.accepted_len, out.status, out.path_used,
            )
            if any(t is not None and not isinstance(t, torch.Tensor) for t in tensors):
                raise TypeError("all CUDA arguments must be preallocated tensors")
            if any(t is not None and t.requires_grad for t in tensors):
                raise ValueError("exactsample is an inference operator and does not support autograd")
            try:
                torch.ops.exactsample.verify_out(*tensors, modes[mode], paths[dispatch], plan.max_draft,
                                         plan.list_max if top_k_top_p else 0)
            finally:
                # A launch error can follow partially enqueued work. Retain its
                # ownership too, so another stream cannot reuse the workspace.
                for tensor in tensors:
                    if tensor is not None and tensor.device == plan.device:
                        tensor.record_stream(stream)
                plan._references = tensors
                plan._last_stream = stream.cuda_stream
                plan._event.record(stream)
            out.num_requests = requests
        finally:
            plan._lock.release()
    return out
