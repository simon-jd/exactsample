import math
import unittest
import numpy as np
from exactsample import RequestParams, RngKey, Status, distribution, verify_reference


class ContractEdges(unittest.TestCase):
    def test_k_1_and_vocab(self):
        p, status = distribution([1., 3., 2.], top_k=1)
        self.assertEqual(status, Status.OK)
        np.testing.assert_array_equal(p, [0, 1, 0])
        p, _ = distribution([1., 3., 2.], top_k=3)
        base, _ = distribution([1., 3., 2.])
        np.testing.assert_array_equal(p, base)

    def test_greedy_tie_and_mask(self):
        p, status = distribution([2., 2., 1.], temperature=0)
        self.assertEqual(status, Status.OK)
        np.testing.assert_array_equal(p, [1, 0, 0])
        p, status = distribution([2., 2., 1.], temperature=0, mask=np.array([6], np.int32))
        np.testing.assert_array_equal(p, [0, 1, 0])

    def test_invalid_values_remain_invalid_behind_mask(self):
        for bad in [math.nan, math.inf]:
            p, status = distribution([0., bad], mask=np.array([1], np.int32))
            self.assertEqual(status, Status.INVALID_INPUT)
            np.testing.assert_array_equal(p, [0, 0])

    def test_all_negative_infinity(self):
        _, status = distribution([-math.inf] * 3)
        self.assertEqual(status, Status.NO_VALID_TOKEN)

    def test_minp_boundary_keeps_equal(self):
        p, status = distribution([0., 0., -10.], min_p=1)
        self.assertEqual(status, Status.OK)
        np.testing.assert_array_equal(p, [.5, .5, 0])

    def test_packed_mask_word_boundary(self):
        p, status = distribution(np.zeros(35), mask=np.array([-2147483648, 1], np.int32))
        self.assertEqual(status, Status.OK)
        self.assertEqual(np.flatnonzero(p).tolist(), [31, 32])
        np.testing.assert_array_equal(p[[31, 32]], [.5, .5])

    def test_invalid_parameters(self):
        for params in [dict(temperature=-1), dict(temperature=math.nan),
                       dict(top_p=0), dict(top_p=1.1), dict(min_p=-.1),
                       dict(min_p=1.1), dict(top_k=-1), dict(top_k=1.5)]:
            with self.subTest(params=params):
                _, status = distribution([0., 1.], **params)
                self.assertEqual(status, Status.INVALID_INPUT)

    def test_empty_ragged_segment_is_invalid(self):
        out = verify_reference(np.zeros((1, 2)), [0, 0, 1], np.empty((2, 0), np.int64),
                               rng=RngKey([2, 3], [0, 0], [0]))
        self.assertEqual(out.status.tolist(), [int(Status.INVALID_INPUT), int(Status.OK)])
        self.assertEqual(out.tokens[0].tolist(), [-1])

    def test_zero_proposal_probability_clears_output(self):
        out = verify_reference([[0., 0.], [0., 0.]], [0, 2], [[1]],
                               draft_logits=[[0., -math.inf], [0., 0.]],
                               params=RequestParams(draft_kind='full'),
                               rng=RngKey(3, 0, [0, 1]))
        self.assertEqual(out.status.tolist(), [int(Status.DRAFT_ZERO_PROB)])
        self.assertEqual(out.tokens.tolist(), [[-1, -1]])
        self.assertEqual(out.accepted_len.tolist(), [0])

    def test_invalid_later_row_is_atomic_even_if_first_rejects(self):
        out = verify_reference([[0., -math.inf], [math.nan, 0.]], [0, 2], [[1]],
                               rng=RngKey(5, 0, [0, 1]))
        self.assertEqual(out.status.tolist(), [int(Status.INVALID_INPUT)])
        self.assertEqual(out.tokens.tolist(), [[-1, -1]])
        self.assertEqual(out.accepted_len.tolist(), [0])

    def test_noncontiguous_inputs(self):
        backing = np.arange(48, dtype=np.float64).reshape(4, 12)
        logits = backing[::2, ::3]
        self.assertFalse(logits.flags.c_contiguous)
        args = dict(cu_rows=[0, 2], draft_ids=np.array([[0, -1]])[:, ::2],
                    rng=RngKey(8, 0, np.array([10, -1, 11])[::2]))
        out = verify_reference(logits, **args)
        contiguous = verify_reference(logits.copy(), **args)
        for field in ['tokens', 'accepted_len', 'status']:
            np.testing.assert_array_equal(getattr(out, field), getattr(contiguous, field))

    def test_invalid_draft_id(self):
        for draft in [-1, 2]:
            out = verify_reference(np.zeros((2, 2)), [0, 2], [[draft]],
                                   rng=RngKey(3, 0, [0, 1]))
            self.assertEqual(out.status.tolist(), [int(Status.INVALID_INPUT)])
            self.assertEqual(out.tokens.tolist(), [[-1, -1]])

    def test_mixed_uint64_seed_extremes(self):
        out = verify_reference(np.zeros((2, 2)), [0, 1, 2], np.empty((2, 0), np.int64),
                               rng=RngKey([2**64 - 1, 0], [2**32 - 1, 0], [2**32 - 1, 0]))
        self.assertEqual(out.status.tolist(), [int(Status.OK), int(Status.OK)])

    def test_metadata_error_precedes_no_valid_target(self):
        out = verify_reference([[-float('inf')] * 2] * 2, [0, 2], [[2]],
                               rng=RngKey(0, 0, [0, 1]))
        self.assertEqual(out.status.tolist(), [int(Status.INVALID_INPUT)])

    def test_acceptance_threshold_is_strict(self):
        from unittest.mock import patch
        with patch('exactsample.reference.uniform', return_value=.5):
            out = verify_reference(np.zeros((2, 2)), [0, 2], [[0]],
                                   rng=RngKey(1, 0, [0, 1]))
        self.assertEqual(out.accepted_len.tolist(), [0])
        self.assertEqual(out.tokens.tolist(), [[1, -1]])
        with patch('exactsample.reference.uniform', return_value=np.nextafter(.5, 0)):
            accepted = verify_reference(np.zeros((2, 2)), [0, 2], [[0]],
                                        rng=RngKey(1, 0, [0, 1]))
        self.assertEqual(accepted.accepted_len.tolist(), [1])
