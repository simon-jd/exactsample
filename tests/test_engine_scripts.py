"""CPU checks of the analysis in the engine scripts (no engine or GPU is started)."""
import unittest

from bench.vllm_e2e import model_label, profile_summary, summary
from integration.vllm_determinism import compare, internal_to_external


def probe(tokens, rows, prompt_len=3, batches=None):
    return {"tokens": tokens, "prompt_len": prompt_len, "times_preempted": 0,
            "logits_sha_by_position": {str(position): sha for position, sha in rows.items()},
            "batch_by_position": batches or {}}


class DeterminismAttribution(unittest.TestCase):
    def test_identical_run(self):
        alone = probe([5, 6, 7], {2: "a", 3: "b", 4: "c"})
        result = compare(alone, probe([5, 6, 7], {2: "a", 3: "b", 4: "c"}))
        self.assertTrue(result["tokens_equal"])
        self.assertIsNone(result["cause"])
        self.assertEqual(result["logits_rows_differing_total"], 0)

    def test_token_from_identical_logits_is_the_sampler(self):
        # Output token i comes from the row at position prompt_len - 1 + i.
        alone = probe([5, 6, 7], {2: "a", 3: "b", 4: "c"})
        other = probe([5, 9, 7], {2: "a", 3: "b", 4: "c"}, batches={"3": [17, True]})
        result = compare(alone, other)
        self.assertEqual(result["first_token_divergence"], 1)
        self.assertEqual(result["cause"], "sampler (identical logits)")
        self.assertEqual(result["batch_at_divergence"], {"requests": 17, "has_explicit_seed": True})

    def test_earlier_logits_difference_is_the_model(self):
        alone = probe([5, 6, 7], {2: "a", 3: "b", 4: "c"})
        result = compare(alone, probe([5, 6, 8], {2: "a", 3: "B", 4: "C"}))
        self.assertEqual(result["first_token_divergence"], 2)
        self.assertEqual(result["first_logits_difference"], 1)
        self.assertEqual(result["cause"], "model logits")
        self.assertEqual(result["logits_rows_differing_total"], 2)

    def test_logits_differences_that_change_no_token(self):
        alone = probe([5, 6], {2: "a", 3: "b"})
        result = compare(alone, probe([5, 6], {2: "a", 3: "x"}))
        self.assertTrue(result["tokens_equal"])
        self.assertEqual(result["logits_rows_differing_total"], 1)

    def test_internal_ids_map_to_one_external_id(self):
        ids = ["probe-p0-alone", "probe-p0-alone2"]
        self.assertEqual(internal_to_external("probe-p0-alone-abcdefgh", ids), "probe-p0-alone")
        with self.assertRaises(RuntimeError):
            internal_to_external("probe-p1-alone-abcdefgh", ids)


class EndToEndSummaries(unittest.TestCase):
    def test_model_label_from_cache_snapshot(self):
        path = "/cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        self.assertEqual(model_label(path), "Qwen/Qwen3-1.7B@70d244cc86cc")
        self.assertEqual(model_label("/models/local-model"), "local-model")
        self.assertIsNone(model_label(None))

    def test_profile_summary_uses_decode_steps_only(self):
        steps = [{"has_prefill": True, "bridge_used": True, "sample_host_ns": 9e6, "bridge_host_ns": 9e6,
                  "sample_gpu_ms": 9.0, "logits_gpu_ms": 1.0, "sampling_gpu_ms": 8.0, "step_period_gpu_ms": 20.0}]
        steps += [{"has_prefill": False, "bridge_used": index != 2, "sample_host_ns": 1e5, "bridge_host_ns": 5e4,
                   "sample_gpu_ms": .5, "logits_gpu_ms": .2, "sampling_gpu_ms": .3, "step_period_gpu_ms": 6.0}
                  for index in range(4)]
        result = profile_summary(steps)
        self.assertEqual((result["steps"], result["decode_steps"], result["bridge_fallback_steps"]), (5, 4, 1))
        self.assertAlmostEqual(result["sampling_share_of_step"], .05)
        self.assertEqual(summary([3, 1, 2])["median"], 2)


if __name__ == "__main__":
    unittest.main()
