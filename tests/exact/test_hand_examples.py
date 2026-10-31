import math
import unittest
import numpy as np
from exactsample import RequestParams, RngKey, Status, distribution, verify_reference


class HandCalculatedExamples(unittest.TestCase):
    def check_distribution(self, weights, expected, **kwargs):
        logits = np.log(np.asarray(weights, dtype=np.float64))
        actual, status = distribution(logits, **kwargs)
        self.assertEqual(status, Status.OK)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_equal_probabilities(self):
        self.check_distribution([1, 1, 1, 1], [0.25] * 4)

    def test_dominant_probability(self):
        self.check_distribution([97, 1, 1, 1], [.97, .01, .01, .01])

    def test_nearly_everything_filtered(self):
        self.check_distribution([8, 4, 2, 1], [1, 0, 0, 0], min_p=.75)

    def test_top_p_at_exact_binary_boundary(self):
        # Equal logits make cumulative probabilities binary-exact. Include
        # exactly two tokens at .5; one ULP above .5 requires a third.
        self.check_distribution([1, 1, 1, 1], [.5, .5, 0, 0], top_p=.5)
        self.check_distribution([1, 1, 1, 1], [1 / 3] * 3 + [0],
                                top_p=np.nextafter(.5, 1.0))

    def test_top_k_tie_uses_low_id(self):
        self.check_distribution([4, 2, 2, 1], [2 / 3, 1 / 3, 0, 0], top_k=2)

    def test_reject_first_draft(self):
        out = verify_reference([[0., -math.inf], [0., 0.]], [0, 2], [[1]],
                               rng=RngKey(123, 0, [4, 5]))
        self.assertEqual(out.status.tolist(), [int(Status.OK)])
        self.assertEqual(out.tokens.tolist(), [[0, -1]])
        self.assertEqual(out.accepted_len.tolist(), [0])

    def test_reject_middle_draft(self):
        out = verify_reference([[0., -math.inf], [-math.inf, 0.], [0., 0.]],
                               [0, 3], [[0, 0]], rng=RngKey(456, 0, [4, 5, 6]))
        self.assertEqual(out.status.tolist(), [int(Status.OK)])
        self.assertEqual(out.tokens.tolist(), [[0, 1, -1]])
        self.assertEqual(out.accepted_len.tolist(), [1])

    def test_all_mask(self):
        probs, status = distribution([0., 1., 2.], mask=np.array([0], dtype=np.int32))
        self.assertEqual(status, Status.NO_VALID_TOKEN)
        np.testing.assert_array_equal(probs, [0, 0, 0])

    def test_filter_order_minp_before_topk_before_topp(self):
        # Original weights 8,4,2,1: min-p=.2 removes 1, top-k=2 keeps
        # 8,4; top-p=.7 needs both since 8/12 < .7. Final mass is 12.
        self.check_distribution([8, 4, 2, 1], [2 / 3, 1 / 3, 0, 0],
                                min_p=.2, top_k=2, top_p=.7)

    def test_temperature_changes_ratios(self):
        self.check_distribution([16, 4, 1], [4 / 7, 2 / 7, 1 / 7], temperature=2)
