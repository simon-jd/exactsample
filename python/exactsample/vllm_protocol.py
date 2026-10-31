"""Typed worker-to-scheduler status extension for the pinned local vLLM fork.

The single-device executor returns ModelRunnerOutput directly. The pinned
multiprocess executor's shared-memory queue uses pickle, which also preserves
this declared dataclass field. This module intentionally does not import worker
or CUDA code when the scheduler reconstructs the object.
"""
from dataclasses import dataclass, field
from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class ExactSampleModelRunnerOutput(ModelRunnerOutput):
    exactsample_status: dict[str, int] = field(default_factory=dict)
