"""Named worker RPCs for the integration scripts, never used by product decode.

vLLM's worker_extension_cls mechanism adds these methods to the worker, so the
scripts call them by name through collective_rpc instead of serializing
callables. Each method stays dormant until a script requests it. Load it with
the repository root on PYTHONPATH (``integration.vllm_worker_probe.WorkerProbe``).
"""


class WorkerProbe:
    def exactsample_install_fault(self, min_draft):
        from integration.vllm_error_smoke import install_fault
        return install_fault(self, min_draft)

    def exactsample_get_fault(self):
        from integration.vllm_error_smoke import get_fault
        return get_fault(self)

    def exactsample_install_lifecycle_observer(self):
        from integration.vllm_lifecycle import install_observer
        return install_observer(self)

    def exactsample_read_lifecycle_observer(self):
        from integration.vllm_lifecycle import read_observer
        return read_observer(self)
