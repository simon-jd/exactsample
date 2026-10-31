"""Independent CPU review of bridge mapping and patched dispatch boundaries.

Set EXACTSAMPLE_VLLM_SOURCE to a pinned vLLM checkout or site-packages to additionally
exercise functions extracted from the actual original and patched source.
These tests do not claim to execute the inference engine or CUDA kernels.
"""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import __future__

import numpy as np
from exactsample.vllm_bridge import (_runner_reason, finish_requests, grammar_mask_rows,
                            register_request, unsupported_request)


class Sampler:
    def __init__(self):
        self.logits_processors = [object(), object(), object()]
        self.return_sampling_mask = False
        self.trace_replay_state = None
        self.compute_nans = False
        self.sampling_states = NS(seeds=NS(np=np.arange(32, dtype=np.int64) + 100))


def runner_fixture():
    return NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                                data_parallel_size=1, decode_context_parallel_size=1,
                                prefill_context_parallel_size=1),
              batch_sharder=None, adaptive_verification=None,
              model_state=NS(num_new_sampled_tokens_per_step=1), sampler=Sampler(),
              vllm_config=NS(watermark_config=None), speculative_config=None,
              _exactsample_bridge={"requests": {}, "request_seeds": {}})


class BridgeEligibilityReview(unittest.TestCase):
    def test_request_permutation_and_slot_reuse(self):
        runner = runner_fixture()
        register_request(runner, "alice", 7, NS())
        register_request(runner, "bob", 3, NS())
        for slots, ids in (([7, 3], ["alice", "bob"]), ([3, 7], ["bob", "alice"])):
            self.assertIsNone(_runner_reason(runner, NS(idx_mapping_np=slots, req_ids=ids)))
        register_request(runner, "carol", 7, NS())
        self.assertEqual(_runner_reason(runner, NS(idx_mapping_np=[7], req_ids=["alice"])),
                         "unregistered_or_reused_request_slot")
        self.assertIsNone(_runner_reason(runner, NS(idx_mapping_np=[7], req_ids=["carol"])))

    def test_one_unsupported_request_falls_back_entire_batch(self):
        runner = runner_fixture()
        register_request(runner, "supported", 1, NS())
        register_request(runner, "penalty", 2, NS(repetition_penalty=1.2))
        self.assertEqual(_runner_reason(runner, NS(idx_mapping_np=[1, 2], req_ids=["supported", "penalty"])),
                         "repetition_penalty")
        self.assertIsNone(_runner_reason(runner, NS(idx_mapping_np=[1], req_ids=["supported"])))

    def test_preempted_unseeded_identity_survives_slot_recycling(self):
        runner = runner_fixture()
        register_request(runner, "continued", 7, NS(seed=None))
        initial = runner._exactsample_bridge["request_seeds"]["continued"]
        register_request(runner, "other", 7, NS(seed=None))
        runner.sampler.sampling_states.seeds.np[3] = 987654
        register_request(runner, "continued", 3, NS(seed=None))
        self.assertEqual(runner._exactsample_bridge["request_seeds"]["continued"], initial)
        self.assertEqual(runner._exactsample_bridge["requests"][7]["id"], "other")
        finish_requests(runner, ["continued"])
        self.assertNotIn("continued", runner._exactsample_bridge["request_seeds"])
        self.assertNotIn(3, runner._exactsample_bridge["requests"])
        self.assertIn(7, runner._exactsample_bridge["requests"])
        register_request(runner, "continued", 3, NS(seed=None))
        self.assertEqual(runner._exactsample_bridge["request_seeds"]["continued"], 987654)

    def test_runner_features_cannot_bypass_native_processing(self):
        batch = NS(idx_mapping_np=[], req_ids=[])
        for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                      "decode_context_parallel_size", "prefill_context_parallel_size"):
            runner = runner_fixture()
            setattr(runner.parallel_config, field, 2)
            self.assertEqual(_runner_reason(runner, batch), field)
        changes = [
            ("batch_sharder", object(), "sharding_or_adaptive_verification"),
            ("adaptive_verification", object(), "sharding_or_adaptive_verification"),
        ]
        for name, value, reason in changes:
            runner = runner_fixture()
            setattr(runner, name, value)
            self.assertEqual(_runner_reason(runner, batch), reason)
        for field, value, reason in (("return_sampling_mask", True, "sampling_mask_or_trace_replay"),
                                     ("trace_replay_state", object(), "sampling_mask_or_trace_replay"),
                                     ("compute_nans", True, "requested_nan_metrics"),
                                     ("logits_processors", [1, 2, 3, 4], "custom_sampler_or_processors")):
            runner = runner_fixture()
            setattr(runner.sampler, field, value)
            self.assertEqual(_runner_reason(runner, batch), reason)
        runner = runner_fixture()
        runner.model_state.num_new_sampled_tokens_per_step = 2
        self.assertEqual(_runner_reason(runner, batch), "multi_bonus_model")
        runner = runner_fixture()
        runner.vllm_config.watermark_config = object()
        self.assertEqual(_runner_reason(runner, batch), "watermark")

    def test_draft_modes_and_metadata_fallback(self):
        runner = runner_fixture()
        runner.speculative_config = NS(method="draft_model", draft_sample_method="greedy",
                                       rejection_sample_method="standard", enable_adaptive_verification=False,
                                       use_heterogeneous_vocab=False)
        batch = NS(idx_mapping_np=[], req_ids=[])
        self.assertIsNone(_runner_reason(runner, batch))
        for field, value, reason in (("method", "ngram", "requires_greedy_draft_model"),
                                     ("draft_sample_method", "probabilistic", "requires_greedy_draft_model"),
                                     ("rejection_sample_method", "block", "requires_standard_verification"),
                                     ("enable_adaptive_verification", True, "requires_standard_verification"),
                                     ("use_heterogeneous_vocab", True, "heterogeneous_vocab")):
            original = getattr(runner.speculative_config, field)
            setattr(runner.speculative_config, field, value)
            self.assertEqual(_runner_reason(runner, batch), reason)
            setattr(runner.speculative_config, field, original)
        for field in ("logprob_token_ids", "prompt_logprob_token_ids", "allowed_token_ids",
                      "_bad_words_token_ids", "trace_decode_token_ids", "extra_args"):
            self.assertEqual(unsupported_request(NS(**{field: [1]})), field)

    def test_grammar_reordering_preserves_every_position(self):
        requests = ["short", "plain", "long", "ordinary"]
        offsets = np.array([0, 2, 3, 7, 8], np.int32)
        packed = np.arange(12, dtype=np.int32).reshape(6, 2)
        actual = grammar_mask_rows(requests, offsets, ["long", "short"], packed, 35)
        np.testing.assert_array_equal(actual[:2], packed[4:])
        np.testing.assert_array_equal(actual[3:7], packed[:4])
        np.testing.assert_array_equal(actual[[2, 7]], -np.ones((2, 2), np.int32))
        for ids, grammar in ((["a", "a"], ["a"]), (["a", "b"], ["a", "a"])):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                grammar_mask_rows(ids, [0, 1, 2], grammar, np.ones((2, 1), np.int32), 8)


def extracted_function(source, name, *, class_name=None, globals_=None):
    module = ast.parse(source)
    scope = module.body
    if class_name is not None:
        scope = next(node.body for node in scope if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = copy.deepcopy(next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name))
    function.decorator_list = []
    namespace = dict(globals_ or {})
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pinned-vllm-function>", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@unittest.skipUnless(os.environ.get("EXACTSAMPLE_VLLM_SOURCE"), "set EXACTSAMPLE_VLLM_SOURCE for pinned-source review")
class PinnedPatchReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ["EXACTSAMPLE_VLLM_SOURCE"])
        repo = Path(__file__).resolve().parents[1]
        manifest = json.loads((repo / "vllm-patches/manifest.json").read_text())
        cls.temporary = tempfile.TemporaryDirectory(prefix="exactsample-bridge-review-")
        cls.root = Path(cls.temporary.name)
        actual_hashes = set()
        for name, record in manifest["files"].items():
            actual = cls.source / name
            digest = hashlib.sha256(actual.read_bytes()).hexdigest()
            if digest == record["before_sha256"]:
                actual_hashes.add("original")
            elif digest == record["after_sha256"]:
                actual_hashes.add("patched")
            else:
                raise AssertionError(f"unpinned source hash: {actual}")
            target = cls.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(actual, target)
        if len(actual_hashes) != 1:
            raise AssertionError("mixed original/patched source files")
        patch_file = repo / "vllm-patches" / manifest["patch"]
        if "patched" in actual_hashes:
            subprocess.run(["git", "apply", "--reverse", str(patch_file)], cwd=cls.root, check=True)
        cls.original = {name: (cls.root / name).read_text() for name in manifest["files"]}
        subprocess.run(["git", "apply", str(patch_file)], cwd=cls.root, check=True)
        cls.patched = {name: (cls.root / name).read_text() for name in manifest["files"]}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_apply_script_patches_a_root_inside_another_work_tree(self):
        # A virtualenv's site-packages under a Git checkout (.envs/ here).
        repo = Path(__file__).resolve().parents[1]
        manifest = json.loads((repo / "vllm-patches/manifest.json").read_text())
        with tempfile.TemporaryDirectory(prefix="exactsample-apply-") as directory:
            outer = Path(directory)
            subprocess.run(["git", "init", "-q", str(outer)], check=True)
            root = outer / "env" / "site-packages"
            for name, text in self.original.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            for arguments in (["--check"], [], ["--reverse"], []):
                subprocess.run([sys.executable, str(repo / "vllm-patches/apply.py"), "--root", str(root),
                                *arguments], check=True, capture_output=True)
            for name, record in manifest["files"].items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), record["after_sha256"])

    def test_grammar_mapping_matches_pinned_native_layout(self):
        source = (self.source / "vllm/v1/worker/gpu/structured_outputs.py").read_text()
        native = extracted_function(source, "_build_grammar_mapping", globals_={"np": np})
        ids, constrained, offsets = ["c", "a", "b"], ["b", "c"], np.array([0, 2, 3, 7])
        masks = np.arange(6, dtype=np.int32).reshape(6, 1)
        actual = grammar_mask_rows(ids, offsets, constrained, masks, 8)
        mapping = native(ids, constrained, offsets, np.diff(offsets) - 1, 1, 5)
        expected = np.full((7, 1), -1, np.int32)
        for index, encoded in enumerate(mapping):
            request, local = divmod(encoded, 5)
            expected[offsets[request] + local] = masks[index]
        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch CPU dependency missing")
    def test_disabled_patch_executes_identical_native_sample_paths(self):
        import torch
        name = "vllm/v1/worker/gpu/model_runner.py"
        ordinary = extracted_function(self.original[name], "sample", class_name="GPUModelRunner")
        modified = extracted_function(self.patched[name], "sample", class_name="GPUModelRunner", globals_={"os": os})
        real_import = builtins.__import__

        def forbid_exactsample(name, *args, **kwargs):
            if name == "exactsample" or name.startswith("exactsample."):
                raise AssertionError("disabled patch imported exactsample")
            return real_import(name, *args, **kwargs)

        def invoke(function, speculative, grammar):
            calls = []
            result = NS(num_sampled=torch.tensor([2 if speculative else 1]), num_rejected=torch.tensor([0]))
            def sampler(logits, batch, *args):
                calls.append((logits.clone(), args))
                return result
            def mask(logits, *_):
                logits[:, 0] = -torch.inf
            runner = NS(batch_sharder=None, model=NS(compute_logits=lambda hidden: hidden.clone()),
                        sampler=sampler, rejection_sampler=sampler if speculative else None,
                        speculator=NS(draft_logits=None), structured_outputs_worker=NS(apply_grammar_bitmask=mask))
            batch = NS(logits_indices=torch.tensor([1, 2] if speculative else [1]), num_reqs=1,
                       num_draft_tokens=int(speculative))
            constraints = NS(structured_output_request_ids=["request"], grammar_bitmask=np.array([[2]], np.int32)) if grammar else None
            answer = function(runner, torch.arange(12, dtype=torch.float32).reshape(3, 4), batch, constraints)
            self.assertIs(answer[0], result)
            return calls
        for speculative in (False, True):
            for grammar in (False, True):
                with patch.dict(os.environ, {"EXACTSAMPLE_DECODE": "0"}), patch("builtins.__import__", side_effect=forbid_exactsample):
                    before, after = invoke(ordinary, speculative, grammar), invoke(modified, speculative, grammar)
                self.assertEqual(len(before), 1)
                self.assertEqual(len(after), 1)
                torch.testing.assert_close(before[0][0], after[0][0], rtol=0, atol=0)
                self.assertEqual(before[0][1], after[0][1])

    def test_parallel_children_marked_only_when_enabled_and_not_mutated(self):
        name = "vllm/v1/engine/parallel_sampling.py"
        function = extracted_function(self.patched[name], "get_child_info", class_name="ParentRequest",
                                      globals_={"os": os, "copy": copy.copy})
        original_extra = {"caller_option": 4}
        params = NS(n=1, seed=None, extra_args=original_extra)
        parent = NS(request_id="parent", child_requests=set(), n=3,
                    _get_child_sampling_params=lambda index: params)
        with patch.dict(os.environ, {"EXACTSAMPLE_DECODE": "0"}):
            name, disabled = function(parent, 1)
        self.assertEqual(name, "1_parent")
        self.assertIs(disabled, params)
        with patch.dict(os.environ, {"EXACTSAMPLE_DECODE": "1"}):
            _, enabled = function(parent, 2)
        self.assertIsNot(enabled, params)
        self.assertIsNot(enabled.extra_args, original_extra)
        self.assertEqual(enabled.extra_args, {"caller_option": 4, "exactsample_parent_n": 3})
        self.assertEqual(params.extra_args, {"caller_option": 4})
        self.assertIsNotNone(unsupported_request(enabled))

    def test_scheduler_request_error_isolated_and_stale_and_off_ignored(self):
        from collections import defaultdict
        source = self.patched["vllm/v1/core/sched/scheduler.py"]
        statuses = NS(RUNNING=1, FINISHED_ERROR=9, FINISHED_STOPPED=8)
        update = extracted_function(source, "update_from_output", class_name="Scheduler",
                                    globals_={"os": os, "defaultdict": defaultdict,
                                              "RequestStatus": statuses, "EngineCoreOutput": NS,
                                              "EngineCoreOutputs": NS})
        class Request:
            def __init__(self, identity, stale):
                self.__dict__.update(request_id=identity, num_in_flight_tokens=1,
                    num_stale_output_tokens=int(stale), drop_stale_output=False,
                    resumable=True, has_encoder_inputs=False, pooling_params=None,
                    status=1, sampling_params=None, spec_decode_metrics=None,
                    client_index=0, stop_reason=None, trace_headers=None,
                    num_nans_in_logits=0, tokens=[])
            def is_finished(self): return self.status == 9
            def get_finished_reason(self): return "error" if self.status == 9 else None
            def take_prefill_stats(self): return None
            def take_events(self): return []
        for enabled, stale, drop in ((True, False, False), (True, True, False),
                                     (True, True, True), (False, False, False)):
            bad, good = Request("bad", stale), Request("good", False)
            bad.drop_stale_output = drop
            terminated = []
            def finish(ids, status):
                terminated.extend(ids)
                for key in ids:
                    requests[key].status = status
                return [requests[key] for key in ids]
            def accept(request, tokens, **kwargs):
                request.tokens.extend(tokens)
                return tokens, False
            requests = {"bad": bad, "good": good}
            scheduler = NS(requests=requests, defer_block_free=False, perf_metrics=None,
                num_sampled_tokens_per_step=1, is_mm_encoder_only=False,
                structured_output_manager=NS(accept_tokens=lambda *args: True),
                aux_output_connector=None, return_sampling_mask=False,
                grammar_compile_error_reqs=set(), encoder_cache_mismatch_reqs=set(),
                ec_connector=None, connector=None, finished_req_ids_dict={},
                kv_cache_manager=NS(take_events=lambda: None), make_stats=lambda *args, **kwargs: None,
                _update_request_with_output=accept, finish_requests=finish)
            scheduled = NS(num_scheduled_tokens={"bad": 1, "good": 1}, scheduled_spec_decode_tokens={})
            output = NS(sampled_token_ids=[[], [42]], logprobs=None, prompt_logprobs_dict={},
                prompt_token_id_logprobs_dict={}, pooler_output=None, num_nans_in_logits=None,
                kv_connector_output=None, ec_connector_output=None, cudagraph_stats=None,
                req_id_to_index={"bad": 0, "good": 1}, exactsample_status={"bad": 1, "good": 0})
            with patch.dict(os.environ, {"EXACTSAMPLE_DECODE": str(int(enabled))}):
                result = update(scheduler, scheduled, output)
            self.assertEqual(good.tokens, [42])
            self.assertEqual(bad.tokens, [])
            self.assertEqual([bad.num_in_flight_tokens, good.num_in_flight_tokens], [0, 0])
            self.assertEqual(bad.num_stale_output_tokens, 0)
            by_id = {item.request_id: item for item in result[0].outputs}
            self.assertEqual(by_id["good"].new_token_ids, [42])
            if enabled and not stale:
                self.assertEqual(terminated, ["bad"])
                self.assertEqual(bad.status, statuses.FINISHED_ERROR)
                self.assertFalse(bad.resumable)
                self.assertEqual(by_id["bad"].new_token_ids, [])
                self.assertEqual(by_id["bad"].finish_reason, "error")
                self.assertEqual(by_id["bad"].stop_reason, "exactsample_status:1")
            else:
                self.assertEqual(terminated, [])
                self.assertNotIn("bad", by_id)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch CPU dependency missing")
    def test_success_hook_consumes_raw_logits_and_skips_native_mask_sampler(self):
        import torch
        function = extracted_function(self.patched["vllm/v1/worker/gpu/model_runner.py"], "sample",
                                      class_name="GPUModelRunner", globals_={"os": os})
        output = NS(num_sampled=torch.tensor([1]), num_rejected=torch.tensor([0]))
        hidden = torch.tensor([[1., 2., 3.]])
        batch = NS(logits_indices=torch.tensor([0]), num_reqs=1, num_draft_tokens=0)
        grammar = NS(structured_output_request_ids=["x"], grammar_bitmask=np.array([[2]], np.int32))
        def forbidden(*args):
            raise AssertionError("exactsample success executed native mutation/sampling")
        runner = NS(batch_sharder=None, model=NS(compute_logits=lambda state: state.clone()),
                    sampler=forbidden, rejection_sampler=None,
                    structured_outputs_worker=NS(apply_grammar_bitmask=forbidden))
        def bridge(actual_runner, logits, actual_batch, actual_grammar):
            self.assertIs(actual_runner, runner)
            self.assertIs(actual_batch, batch)
            self.assertIs(actual_grammar, grammar)
            torch.testing.assert_close(logits, hidden, rtol=0, atol=0)
            return output
        with patch.dict(os.environ, {"EXACTSAMPLE_DECODE": "1"}), patch("exactsample.vllm_bridge.try_sample", side_effect=bridge):
            result = function(runner, hidden, batch, grammar)
        self.assertIs(result[0], output)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch CPU dependency missing")
class BridgeCpuCallReview(unittest.TestCase):
    """Run the bridge with actual CPU tensors and mocked vLLM/CUDA boundaries."""
    @staticmethod
    def bridge_state(runner, max_draft, requests=3, depth=3):
        """The CPU equivalent of initialize_runner's buffers, with logging events."""
        import torch
        from exactsample.vllm_bridge import _STAGED_FIELDS
        state = runner._exactsample_bridge
        words = requests * (len(_STAGED_FIELDS) + max_draft)
        log = []

        class Event:
            def __init__(self, slot):
                self.slot = slot
            def synchronize(self):
                log.append(("synchronize", self.slot))
            def record(self):
                log.append(("record", self.slot))
        staging = [torch.full((words,), -7, dtype=torch.int64) for _ in range(depth)]
        state.update(plan=NS(max_draft=max_draft), step=0,
                     drafts=torch.empty((requests, max_draft), dtype=torch.int64),
                     offsets=torch.empty(requests + 1, dtype=torch.int64),
                     mask=torch.empty((requests * (max_draft + 1), 1), dtype=torch.int32),
                     staging=staging, staging_np=[slot.numpy() for slot in staging],
                     staging_events=[Event(slot) for slot in range(depth)], staging_next=0,
                     staged=torch.empty(words, dtype=torch.int64),
                     streams=torch.zeros(requests, dtype=torch.int64), kind=torch.zeros(requests, dtype=torch.int64))
        sampling = runner.sampler.sampling_states
        # vLLM keeps these as float32/int32 host arrays.
        for name, value, dtype in (("temperature", .6, np.float32), ("top_p", .95, np.float32),
                                   ("min_p", .1, np.float32), ("top_k", 20, np.int32)):
            setattr(sampling, name, NS(np=np.full(32, value, dtype)))
        runner.sampler.req_states = NS(prefill_len=NS(gpu=torch.zeros(32, dtype=torch.int64)))
        return state, log

    @staticmethod
    def no_host_sync():
        """Patch every tensor-to-host read that would wait for the device."""
        from contextlib import ExitStack
        stack = ExitStack()
        for name in ("cpu", "item", "tolist", "__bool__"):
            stack.enter_context(patch(f"torch.Tensor.{name}", side_effect=AssertionError(f"Tensor.{name} syncs")))
        return stack

    def exercise(self, *, placeholders=False, error=False):
        import torch
        from exactsample.vllm_bridge import try_sample
        runner = runner_fixture()
        state, log = self.bridge_state(runner, max_draft=2)
        state["out"] = NS(tokens=torch.tensor([[11, 8, -1], [7, -1, -1], [9, -1, -1]]),
                          accepted_len=torch.tensor([0 if error else 1, 0, 0], dtype=torch.int32),
                          status=torch.tensor([1 if error else 0, 0, 0], dtype=torch.int32),
                          path_used=torch.ones(3, dtype=torch.int32))
        sampling = runner.sampler.sampling_states
        for request, slot in (("a", 7), ("b", 3), ("c", 1)):
            register_request(runner, request, slot, NS())
        # Native slot seeds change after admission; exactsample must use cached identities.
        sampling.seeds.np[:] = 999
        # A step that starts inside request a's prefill: vLLM marks it
        # prefilling on the host and writes -1 into its draft rows.
        batch = NS(num_reqs=3, req_ids=["a", "b", "c"], idx_mapping_np=np.array([7, 3, 1]),
                   cu_num_logits_np=np.array([0, 3, 4, 6]), cu_num_logits=torch.tensor([0, 3, 4, 6]),
                   is_prefilling_np=np.array([placeholders, False, False]),
                   input_ids=None, positions=None, logits_indices=None, expanded_idx_mapping=None,
                   expanded_local_pos=None, idx_mapping=torch.tensor([7, 3, 1]), seq_lens=torch.tensor([15, 6, 30]))
        gathered = torch.tensor([100, -1 if placeholders else 11, -1 if placeholders else 12, 200, 300, 31])
        positions = torch.tensor([12, 13, 14, 5, 28, 29])
        grammar = NS(structured_output_request_ids=["c", "a"],
                     grammar_bitmask=np.array([[10], [11], [12], [13], [14]], np.int32))
        captured = {}
        def verify(logits, offsets, drafts, **kwargs):
            captured.update(logits=logits.clone(), offsets=offsets.clone(), drafts=drafts.clone(), **kwargs)
            return state["out"]
        modules = {
            "vllm.v1.worker.gpu.input_batch": NS(get_num_sampled_and_rejected=lambda count, *args: (count, torch.tensor([3, 1, 2], dtype=count.dtype) - count)),
            "vllm.v1.worker.gpu.sample.output": NS(SamplerOutput=NS),
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler": NS(gather_draft_sampled=lambda *args: (gathered, positions)),
        }
        logits = torch.arange(6 * 32, dtype=torch.float32).reshape(6, 32)
        with patch.dict(sys.modules, modules), patch("exactsample.cuda.verify", side_effect=verify), \
             patch("torch.cuda.is_current_stream_capturing", return_value=False), self.no_host_sync(), \
             patch.dict(os.environ, {"EXACTSAMPLE_TRACE_PATH": "", "EXACTSAMPLE_CAPTURE_DIR": "", "EXACTSAMPLE_MODE": "coupled"}):
            result = try_sample(runner, logits, batch, grammar)
        if placeholders:
            self.assertIsNone(result)
            self.assertFalse(captured)
            self.assertEqual(log, [])
            return
        self.assertEqual(log, [("synchronize", 0), ("record", 0)])
        torch.testing.assert_close(captured["drafts"], torch.tensor([[11, 12], [-1, -1], [31, -1]]))
        torch.testing.assert_close(captured["rng"].seed, torch.tensor([107, 103, 101]))
        params = captured["params"]
        for name, value in (("temperature", .6), ("top_p", .95), ("min_p", .1)):
            torch.testing.assert_close(getattr(params, name), torch.full((3,), float(np.float32(value)), dtype=torch.float64))
        torch.testing.assert_close(params.top_k, torch.full((3,), 20, dtype=torch.int64))
        torch.testing.assert_close(captured["rng"].positions, positions)
        torch.testing.assert_close(captured["logits"], logits)
        torch.testing.assert_close(captured["mask"], torch.tensor([[12], [13], [14], [-1], [10], [11]], dtype=torch.int32))
        torch.testing.assert_close(result.num_sampled, torch.tensor([0 if error else 2, 1, 1], dtype=torch.int32))
        torch.testing.assert_close(result.exactsample_status, torch.tensor([int(error), 0, 0], dtype=torch.int32))
        torch.testing.assert_close(result.num_rejected, torch.tensor([2 if error else 1, 0, 1], dtype=torch.int32))
        # Status/tokens survive plan reuse while an async D2H copy is pending.
        state["out"].status.fill_(2)
        torch.testing.assert_close(result.exactsample_status, torch.tensor([int(error), 0, 0], dtype=torch.int32))
        self.assertIsNone(result.exactsample_diagnostics)
        snapshot = result.sampled_token_ids.clone()
        state["out"].tokens.fill_(999)
        torch.testing.assert_close(result.sampled_token_ids, snapshot)

    def test_draft_plus_one_shift_identity_raw_mask_and_output_lifetime(self):
        self.exercise()

    def test_prefill_placeholders_fall_back_before_sampling(self):
        self.exercise(placeholders=True)

    def test_non_ok_status_zeroes_only_failed_request_without_host_status_read(self):
        self.exercise(error=True)

    def test_ordinary_steps_rotate_pinned_slots_and_wait_before_reuse(self):
        import torch
        from exactsample.vllm_bridge import try_sample
        runner = runner_fixture()
        state, log = self.bridge_state(runner, max_draft=0)
        state["out"] = NS(tokens=torch.tensor([[5], [6], [7]]), accepted_len=torch.zeros(3, dtype=torch.int32),
                          status=torch.zeros(3, dtype=torch.int32), path_used=torch.ones(3, dtype=torch.int32))
        for request, slot in (("a", 4), ("b", 9), ("c", 2)):
            register_request(runner, request, slot, NS())
        temperature = runner.sampler.sampling_states.temperature.np
        batch = NS(num_reqs=3, req_ids=["c", "a", "b"], idx_mapping_np=np.array([2, 4, 9]),
                   cu_num_logits_np=np.arange(4), cu_num_logits=torch.arange(4),
                   is_prefilling_np=np.zeros(3, bool), input_ids=None, positions=None, logits_indices=None,
                   expanded_idx_mapping=None, expanded_local_pos=None, idx_mapping=torch.tensor([2, 4, 9]),
                   seq_lens=torch.tensor([3, 4, 5]))
        positions = torch.tensor([2, 3, 4])
        calls = []
        def verify(logits, offsets, drafts, **kwargs):
            calls.append({"drafts": drafts.clone(), "temperature": kwargs["params"].temperature.clone(),
                          "seed": kwargs["rng"].seed.clone(), "mask": kwargs["mask"]})
            return state["out"]
        modules = {
            "vllm.v1.worker.gpu.input_batch": NS(get_num_sampled_and_rejected=lambda count, *args: (count, count - 1)),
            "vllm.v1.worker.gpu.sample.output": NS(SamplerOutput=NS),
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler": NS(gather_draft_sampled=lambda *args: (positions, positions)),
        }
        logits = torch.zeros((3, 32))
        with patch.dict(sys.modules, modules), patch("exactsample.cuda.verify", side_effect=verify), \
             patch("torch.cuda.is_current_stream_capturing", return_value=False), self.no_host_sync(), \
             patch.dict(os.environ, {"EXACTSAMPLE_TRACE_PATH": "", "EXACTSAMPLE_CAPTURE_DIR": ""}):
            for step in range(4):
                temperature[[2, 4, 9]] = [step + .25, step + .5, step + .75]
                try_sample(runner, logits, batch, None)
        expected = [(action, step % 3) for step in range(4) for action in ("synchronize", "record")]
        self.assertEqual(log, expected)
        for step, call in enumerate(calls):
            self.assertEqual(tuple(call["drafts"].shape), (3, 0))
            self.assertIsNone(call["mask"])
            torch.testing.assert_close(call["temperature"], torch.tensor([step + .25, step + .5, step + .75],
                                                                         dtype=torch.float64))
            torch.testing.assert_close(call["seed"], torch.tensor([102, 104, 109]))


if __name__ == "__main__":
    unittest.main()
