import unittest
from fractions import Fraction as F
from itertools import product
from tests.exact.enumerate import one_step_joint, rational_distributions, sequence_joint


class RationalEnumeration(unittest.TestCase):
    def test_all_small_p_q_pairs_have_target_marginal(self):
        # Exhaust all zero-containing distributions on a 1/4 probability grid.
        # For V = 8 also exhaust a coarser grid, keeping the test bounded.
        for width, denominator in [(2, 4), (3, 4), (4, 4), (8, 2)]:
            distributions = list(rational_distributions(width, denominator))
            for p, q in product(distributions, repeat=2):
                joint = one_step_joint(p, q)
                marginal = [sum(prob for (_, token, _), prob in joint.items()
                                if token == i) for i in range(width)]
                self.assertEqual(marginal, list(p), (p, q))
                self.assertEqual(sum(joint.values()), F(1))
                self.assertEqual(sum(prob for (_, _, accepted), prob in joint.items()
                                     if accepted), sum(min(a, b) for a, b in zip(p, q)))

    def test_hand_computed_full_proposal_joint(self):
        # p=(3/4,1/4), q=(1/4,3/4): draft 0 accepts its 1/4;
        # draft 1 accepts 1/4 and moves its excess 1/2 to token 0.
        self.assertEqual(one_step_joint((F(3, 4), F(1, 4)), (F(1, 4), F(3, 4))),
                         {(0, 0, True): F(1, 4), (1, 1, True): F(1, 4),
                          (1, 0, False): F(1, 2)})

    def test_one_hot_chain_through_three_drafts(self):
        for width in range(2, 9):
            p = tuple(F(1, width) for _ in range(width))
            for length in range(4):
                q = tuple(F(i == 0) for i in range(width))
                joint = sequence_joint([p] * (length + 1), [q] * length)
                self.assertEqual(sum(joint.values()), 1)
                for accepted in range(length):
                    mass = sum(prob for (_, count), prob in joint.items() if count == accepted)
                    self.assertEqual(mass, F(1, width) ** accepted * F(width - 1, width))
                final_mass = sum(prob for (_, count), prob in joint.items() if count == length)
                self.assertEqual(final_mass, F(1, width) ** length)

    def test_reject_middle_exact_example(self):
        p = [(F(1), F(0)), (F(1, 4), F(3, 4)), (F(1, 2), F(1, 2))]
        q = [(F(1), F(0)), (F(1), F(0))]
        self.assertEqual(sequence_joint(p, q), {
            ((0, 1), 1): F(3, 4), ((0, 0, 0), 2): F(1, 8),
            ((0, 0, 1), 2): F(1, 8)})
