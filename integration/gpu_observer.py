"""Best-effort NVML evidence; observation does not establish GPU exclusivity."""
import threading
import time


class GpuObserver:
    def __init__(self, interval=.5):
        self.interval = interval
        self.samples = []
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            self.nvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as error:
            self.error = str(error)

    def snapshot(self):
        if self.error:
            return
        try:
            api, handle = self.nvml, self.handle
            utilization = api.nvmlDeviceGetUtilizationRates(handle)
            sample = {"time_ns": time.time_ns(), "gpu_utilization_percent": utilization.gpu,
                      "memory_used_bytes": api.nvmlDeviceGetMemoryInfo(handle).used,
                      "processes": [{"pid": process.pid, "used_gpu_memory_bytes": process.usedGpuMemory}
                                    for process in api.nvmlDeviceGetComputeRunningProcesses(handle)]}
            self.samples.append(sample)
        except Exception as error:
            self.error = str(error)

    def start(self):
        self.snapshot()
        def observe():
            while not self._stop.wait(self.interval):
                self.snapshot()
        self._thread = threading.Thread(target=observe, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.snapshot()
        return {"interval_seconds": self.interval, "samples": self.samples, "error": self.error,
                "scope": "NVML GPU 0, includes unrelated co-resident processes; no exclusivity claim"}
