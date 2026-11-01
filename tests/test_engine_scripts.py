"""CPU checks of the analysis in the engine scripts (no engine or GPU is started)."""
import unittest

from bench.vllm_e2e import model_label, profile_summary, summary


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
