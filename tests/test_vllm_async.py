"""CPU execution of pinned native AsyncOutput with simulated deferred copies.

No CUDA execution is claimed here. The native event/copy boundaries are replaced
by a deterministic queue, so status consumption before completion is detectable.
"""
import ast
import __future__
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import numpy as np


def extract_class(source, name, namespace):
    node = next(item for item in ast.parse(source).body if isinstance(item, ast.ClassDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-native-class>", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@unittest.skipUnless(os.environ.get("EXACTSAMPLE_VLLM_SOURCE") and importlib.util.find_spec("torch"),
                     "set EXACTSAMPLE_VLLM_SOURCE and install optional torch for the native async checks")
class AsyncStatusTest(unittest.TestCase):
    @contextmanager
    def environment(self):
        import torch
        native = Path(os.environ["EXACTSAMPLE_VLLM_SOURCE"]) / "vllm/v1"
        repo = Path(__file__).resolve().parents[1]
        copies, events = [], []
        class Event:
            def __init__(self, **kwargs):
                self.record_count = self.sync_count = 0
                events.append(self)
            def record(self, stream):
                self.record_count += 1
                self.fence = len(copies)
            def synchronize(self):
                self.sync_count += 1
                for source, target in copies[:self.fence]:
                    target[:] = source.numpy()
        @contextmanager
        def stream(copy_stream, main_stream):
            yield
        def async_copy(tensor):
            target = np.empty(tuple(tensor.shape), dtype=tensor.numpy().dtype)
            copies.append((tensor, target))
            return target
        outputs = ModuleType("vllm.v1.outputs")
        utils = ModuleType("vllm.v1.worker.gpu.async_utils")
        protocol = ModuleType("exactsample.vllm_protocol")
        helper = ModuleType("exactsample.vllm_async")
        modules = {module.__name__: module for module in (outputs, utils, protocol, helper)}
        with patch.dict(sys.modules, modules):
            outputs.__dict__.update(dataclass=dataclass, field=field)
            output_type = extract_class((native / "outputs.py").read_text(), "ModelRunnerOutput", outputs.__dict__)
            utils.__dict__.update(AsyncModelRunnerOutput=object, torch=NS(cuda=NS(Event=Event)),
                stream=stream, async_copy_to_np=async_copy, envs=NS(VLLM_RAISE_ON_LOGIT_NANS=False))
            extract_class((native / "worker/gpu/async_utils.py").read_text(), "AsyncOutput", utils.__dict__)
            for module, name in ((protocol, "vllm_protocol.py"), (helper, "vllm_async.py")):
                module.__package__ = "exactsample"
                exec(compile((repo / "python/exactsample" / name).read_text(), name, "exec"), module.__dict__)
            yield NS(torch=torch, output_type=output_type, helper=helper, protocol=protocol,
                     events=events, copies=copies)

    def test_status_waits_for_same_event_survives_pickle_and_preserves_healthy_tokens(self):
        with self.environment() as env:
            torch = env.torch
            model = env.output_type(req_ids=["good", "bad"], req_id_to_index={"good": 0, "bad": 1})
            sampler = NS(sampled_token_ids=torch.tensor([[41, 42], [-1, -1]]),
                         num_rejected=torch.tensor([0, 1]), sampling_mask_tensors=None,
                         logprobs_tensors=None, num_nans=None, exactsample_status=torch.tensor([0, 1]))
            status_ref = weakref.ref(sampler.exactsample_status)
            pending = env.helper.ExactSampleAsyncOutput(model_runner_output=model, sampler_output=sampler,
                num_sampled_tokens=torch.tensor([2, 0]), main_stream=object(),
                copy_stream=NS(wait_stream=lambda stream: None), check_ep_fault=False,
                pending_aux_output=None)
            del sampler
            self.assertIsNotNone(status_ref(), "sampler status must outlive the queued copy")
            self.assertEqual(len(env.events), 1)
            self.assertEqual(env.events[0].record_count, 2)
            self.assertEqual(env.events[0].sync_count, 0)
            self.assertEqual(len(env.copies), 3)  # native tokens/count plus exactsample status
            result = pending.get_output()
            self.assertEqual(env.events[0].sync_count, 1)
            self.assertEqual(result.sampled_token_ids, [[41, 42], []])
            self.assertEqual(result.exactsample_status, {"good": 0, "bad": 1})
            restored = pickle.loads(pickle.dumps(result))
            self.assertIsInstance(restored, env.protocol.ExactSampleModelRunnerOutput)
            self.assertEqual(restored.exactsample_status, result.exactsample_status)
            self.assertEqual(restored.sampled_token_ids, result.sampled_token_ids)

    def test_failed_request_cannot_leak_token_even_if_count_is_wrong(self):
        with self.environment() as env:
            torch = env.torch
            model = env.output_type(req_ids=["bad", "good"], req_id_to_index={"bad": 0, "good": 1})
            sampler = NS(sampled_token_ids=torch.tensor([[999], [17]]),
                         num_rejected=torch.tensor([0, 0]), sampling_mask_tensors=None,
                         logprobs_tensors=None, num_nans=None, exactsample_status=torch.tensor([3, 0]))
            pending = env.helper.ExactSampleAsyncOutput(model_runner_output=model, sampler_output=sampler,
                num_sampled_tokens=torch.tensor([1, 1]), main_stream=object(),
                copy_stream=NS(wait_stream=lambda stream: None), check_ep_fault=False,
                pending_aux_output=None)
            self.assertEqual(pending.get_output().sampled_token_ids, [[], [17]])


if __name__ == "__main__":
    unittest.main()
