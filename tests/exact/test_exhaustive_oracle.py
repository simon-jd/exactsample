"""Exhaustive small uniform inputs with a closed-form expected support."""
from fractions import Fraction as F
import math
import unittest
import numpy as np
from exactsample import Status, distribution
from reference.exact import enumerate_ratio_joint
from tests.exact.enumerate import rational_distributions, sequence_joint


class ExhaustiveOracle(unittest.TestCase):
    def test_uniform_vocab_masks_k_and_dyadic_p(self):
        # Uniform allowed tokens have exactly known mass. Enumerating all
        # masks catches IDs leaking through filters and incorrect tie order.
        for vocab in range(1, 9):
            for bits in range(1, 1 << vocab):
                allowed = [i for i in range(vocab) if bits & (1 << i)]
                for k in [0, 1, vocab]:
                    retained = len(allowed) if k == 0 else min(k, len(allowed))
                    for top_p in [.25, .5, .75, 1.]:
                        n = math.ceil(retained * top_p)
                        expected = np.zeros(vocab)
                        expected[allowed[:n]] = 1 / n
                        actual, status = distribution(np.zeros(vocab), top_k=k, top_p=top_p,
                                                       mask=np.array([bits], np.int32))
                        self.assertEqual(status, Status.OK)
                        np.testing.assert_array_equal(actual, expected)

    def test_separately_authored_joint_enumerators_agree(self):
        # Compare both implementations over all denominator-3 p/q on V=3.
        distributions = list(rational_distributions(3, 3))
        for p in distributions:
            for q in distributions:
                expected = sequence_joint([p, (F(1, 3),) * 3], [q])
                actual = enumerate_ratio_joint([p, (F(1, 3),) * 3], [q])
                normalized = {(outcome.tokens, outcome.accepted_len): mass
                              for outcome, mass in actual.items() if mass}
                self.assertEqual(normalized, expected)
