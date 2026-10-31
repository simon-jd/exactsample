import unittest
import numpy as np
from exactsample import RequestParams, RngKey, Status, verify_reference


def ordinary(rows, seed, stream, positions, params=None):
    return verify_reference(rows, np.arange(len(rows) + 1), np.empty((len(rows), 0), np.int64),
                            rng=RngKey([seed] * len(rows), [stream] * len(rows), positions),
                            params=params).tokens[:, 0]


class RequestIdentity(unittest.TestCase):
    def test_slot_permutation_and_batch_insertion(self):
        rows = np.array([[1., 2., 3., 0.], [4., 0., 1., 2.], [0., 0., 0., 0.]])
        seeds = [12, 99, 123]
        streams = [0, 1, 3]
        positions = [100, 5, 12]
        original = verify_reference(rows, [0, 1, 2, 3], np.empty((3, 0), np.int64),
                                    rng=RngKey(seeds, streams, positions))
        order = np.array([2, 0, 1])
        permuted = verify_reference(rows[order], [0, 1, 2, 3], np.empty((3, 0), np.int64),
                                    rng=RngKey(np.array(seeds)[order], np.array(streams)[order],
                                               np.array(positions)[order]))
        np.testing.assert_array_equal(original.tokens[order], permuted.tokens)
        for request in range(3):
            alone = ordinary(rows[request:request + 1], seeds[request], streams[request],
                             [positions[request]])
            np.testing.assert_array_equal(alone, original.tokens[request])

    def test_coupled_spec_on_off_every_draft_sequence(self):
        from itertools import product
        rows = np.array([[0., 1., -1.], [1., 1., 0.], [2., -1., 0.], [-2., 0., 1.]])
        for seed in [0, 1, 0xFFFFFFFFFFFFFFFF]:
            target = ordinary(rows, seed, 17, [10, 11, 12, 13])
            for drafts in product(range(3), repeat=3):
                out = verify_reference(rows, [0, 4], [drafts], rng=RngKey(seed, 17, [10, 11, 12, 13]),
                                       mode='coupled')
                accepted = next((i for i, (x, y) in enumerate(zip(drafts, target)) if x != y), 3)
                self.assertEqual(out.accepted_len.tolist(), [accepted])
                self.assertEqual(out.status.tolist(), [int(Status.OK)])
                self.assertEqual(out.tokens[0, :accepted + 1].tolist(), target[:accepted + 1].tolist())
                self.assertTrue(np.all(out.tokens[0, accepted + 1:] == -1))

    def test_ragged_request_permutation(self):
        rows = np.arange(28, dtype=float).reshape(7, 4) % 7
        drafts = [[0, 1, -1], [2, -1, -1], [1, -1, -1]]
        keys = RngKey([30, 40, 50], [2, 3, 4], [0, 1, 2, 8, 9, 10, 11])
        original = verify_reference(rows, [0, 3, 5, 7], drafts, rng=keys, mode='coupled')
        order = [5, 6, 0, 1, 2, 3, 4]
        swapped = verify_reference(rows[order], [0, 2, 5, 7], [drafts[2], drafts[0], drafts[1]],
                                   rng=RngKey([50, 30, 40], [4, 2, 3], np.array(keys.positions)[order]),
                                   mode='coupled')
        for field in ['tokens', 'accepted_len', 'status']:
            np.testing.assert_array_equal(getattr(original, field)[[2, 0, 1]], getattr(swapped, field))

    def test_cancellation_and_recomputation(self):
        rows = np.array([[0., 1., 2.], [2., 1., 0.]])
        first = ordinary(rows, 89, 4, [99, 100])
        ordinary(rows, 200, 9, [1, 2])  # Unrelated request and slot reuse.
        after = ordinary(rows, 89, 4, [99, 100])
        np.testing.assert_array_equal(first, after)

    def test_logical_split_beyond_1766_rows(self):
        # Checks row identity across a split, not actual 1 GB GPU memory traffic.
        rows = np.zeros((1770, 2))
        whole = ordinary(rows, 1234, 1, np.arange(1770))
        split = np.concatenate([ordinary(rows[:1766], 1234, 1, np.arange(1766)),
                                ordinary(rows[1766:], 1234, 1, np.arange(1766, 1770))])
        np.testing.assert_array_equal(whole, split)
