"""CPU validation of the direct-native large-boundary fixture, not GPU evidence."""
import unittest
import numpy as np
from tests.invariance.vllm_chunk import fixture, description, VOCAB, REQUESTS, DRAFT


class NativeChunkFixture(unittest.TestCase):
    def test_real_boundary_and_shifted_partition(self):
        info = description()
        self.assertGreater(info["fp32_equivalent_bytes"], 1 << 30)
        self.assertEqual(info["native_max_rows"], 1766)
        self.assertLessEqual(1765 * VOCAB * 4, 1 << 30)
        self.assertGreater(1770 * VOCAB * 4, 1 << 30)
        self.assertNotEqual(info["separate_request_split"] * 5, 1765)

    def test_persistent_identity_and_draft_alignment(self):
        x = fixture()
        self.assertEqual(len(np.unique(x["mapping"])), REQUESTS)
        self.assertFalse(np.array_equal(x["mapping"], np.arange(REQUESTS)))
        self.assertEqual(len(np.unique(x["seeds"])), REQUESTS)
        for req in range(REQUESTS):
            lo = req * (DRAFT + 1)
            np.testing.assert_array_equal(x["positions"][lo:lo + 5], x["base_positions"][req] + np.arange(5))
            for step in range(DRAFT):
                token = x["native_drafts"][lo + step + 1]
                invalid = req % 4 == 1 and step == 0 or req % 4 == 2 and step == 1
                self.assertEqual(token not in x["support"][lo + step], invalid)
        self.assertTrue(np.all(np.isfinite(x["weights"][:, 0])))
        self.assertTrue(np.all((x["support"] >= 0) & (x["support"] < VOCAB)))


if __name__ == "__main__":
    unittest.main()
