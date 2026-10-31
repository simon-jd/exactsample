"""Fused-path checks beyond tests/test_cuda.py.

Both kept-set representations are exercised: the FP64 list (list_max=256) and
the FP32 full row (list_max=0). Every result is compared with the FP64 oracle,
including inputs built so that FP32 cannot decide and the FP64 fallback must.
"""
import dataclasses
import os
import unittest
import numpy as np
from exactsample import RequestParams, RngKey, Status, verify_reference
from exactsample.rng import gumbel, uniform
from tests.cuda_helpers import prepared_cuda

full_vocabulary = unittest.skipIf(os.environ.get('EXACTSAMPLE_SANITIZER') == '1', 'too slow under compute-sanitizer')


def packed_mask(allowed):
    """Pack a boolean [rows, V] array into xgrammar int32 words (bit t%32 of word t//32)."""
    rows, vocab = allowed.shape
    bits = np.zeros((rows, (vocab + 31) // 32 * 32), np.uint64)
    bits[:, :vocab] = allowed
    words = (bits.reshape(rows, -1, 32) << np.arange(32, dtype=np.uint64)).sum(-1)
    return words.astype(np.uint32).view(np.int32)


class FusedPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            if os.environ.get('EXACTSAMPLE_REQUIRE_CUDA') == '1':
                raise RuntimeError('CUDA runtime required by qualification command')
            raise unittest.SkipTest('CUDA unavailable; no GPU correctness claim')
        from exactsample.cuda import load_library
        load_library()

    def run_fused(self, rows, offsets, ids, *, list_max=256, dtype=None, mode='ratio', top_k_top_p=True, **kw):
        """Run the fused path, check it against the oracle, return (output, fallbacks)."""
        from exactsample.cuda import verify
        kw.setdefault('rng', RngKey(1234, 3, np.arange(len(rows))))
        args = prepared_cuda(rows, offsets, ids, dtype=dtype, **kw)
        args['plan'].list_max = list_max
        out = verify(**args, mode=mode, dispatch='fused', top_k_top_p=top_k_top_p)
        fallbacks = args['plan'].fallbacks()
        q = args.get('draft_logits')
        expected = verify_reference(args['target_logits'].double().cpu().numpy(), offsets, ids,
                                    params=kw.get('params'), rng=kw['rng'], mask=kw.get('mask'),
                                    draft_logits=None if q is None else q.double().cpu().numpy(), mode=mode)
        for field in ['tokens', 'accepted_len', 'status']:
            np.testing.assert_array_equal(getattr(out, field).cpu().numpy()[:len(offsets) - 1],
                                          getattr(expected, field), err_msg=field)
        self.assertTrue(np.all(out.path_used.cpu().numpy()[:len(offsets) - 1] == 1))
        return out, fallbacks

    def test_random_batches_match_oracle(self):
        import torch
        g = np.random.default_rng(2026)
        for trial in range(16):
            vocab = int(g.choice([37, 300, 2500]))
            counts = g.integers(0, 4, size=6)
            offsets = np.concatenate([[0], np.cumsum(counts + 1)])
            rows = g.normal(scale=float(g.choice([1., 4.])), size=(offsets[-1], vocab))
            if trial % 2:
                rows = np.round(rows * 2) / 2  # many exact ties
            rows[g.random(rows.shape) < .05] = -np.inf
            mask = packed_mask(g.random(rows.shape) < .8) if trial % 3 == 0 else None
            mode = 'coupled' if trial % 4 == 3 else 'ratio'
            kinds = ['one_hot' if mode == 'coupled' or g.random() < .6 else 'full' for _ in counts]
            top_k = [int(g.choice([0, 1, 3, vocab // 3, vocab])) for _ in counts]
            fields = dict(temperature=[float(g.choice([0., .4, 1., 1.7])) for _ in counts], top_k=top_k,
                          min_p=[float(g.choice([0., 0., .02, .3])) for _ in counts], draft_kind=kinds)
            top_p = [float(g.choice([1., .9, .6])) for _ in counts]
            ids = g.integers(0, vocab, size=(len(counts), 3))
            rng = RngKey(g.integers(0, 2**63, size=len(counts), dtype=np.int64).astype(np.uint64) * 2 + 1,
                         g.integers(0, 2**32, size=len(counts)), g.integers(0, 2**32, size=offsets[-1]))
            q = g.normal(scale=2., size=rows.shape) if 'full' in kinds else None
            params = RequestParams(**fields, top_p=top_p)
            for list_max in [256, 0]:
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(trial=trial, list_max=list_max, dtype=dtype):
                        self.run_fused(rows, offsets, ids, list_max=list_max, dtype=dtype, mode=mode,
                                       params=params, rng=rng, mask=mask, draft_logits=q)

    def test_top_k_top_p_hint_never_changes_a_result(self):
        # top_k_top_p=False only skips the list kernels: a request that does
        # filter, a greedy one and a failed one are finished on the full-row path.
        import torch
        g = np.random.default_rng(7)
        offsets = np.array([0, 1, 3, 4, 6, 7, 8])
        rows = g.normal(scale=4., size=(offsets[-1], 2500))
        rows[7] = -np.inf  # no valid token
        params = RequestParams(temperature=[1., 0., .7, .6, 1., .8], top_k=[0, 0, 20, 0, 3, 0],
                               top_p=[1., 1., .95, .9, 1., 1.], min_p=[0., 0., 0., .05, 0., 0.],
                               draft_kind=['one_hot', 'full', 'one_hot', 'one_hot', 'one_hot', 'one_hot'])
        ids = g.integers(0, 2500, size=(len(offsets) - 1, 1))
        q = g.normal(scale=2., size=rows.shape)
        for mode in ['ratio', 'coupled']:  # coupled mode takes one-hot drafts only
            kw = dict(params=params, draft_logits=q) if mode == 'ratio' else dict(
                params=dataclasses.replace(params, draft_kind=['one_hot'] * 6))
            for hint in [True, False]:
                with self.subTest(mode=mode, top_k_top_p=hint):
                    self.run_fused(rows, offsets, ids, dtype=torch.bfloat16, mode=mode, top_k_top_p=hint, **kw)

    def test_near_tie_scores_are_settled_in_fp64(self):
        import torch
        seed, stream, position = 99, 5, 1000
        g0, g1 = gumbel(seed, stream, position, 0, 0), gumbel(seed, stream, position, 0, 1)
        rng = RngKey(seed, stream, [position])
        for eps in [1e-10, -1e-10]:
            # Scores of tokens 0 and 1 differ by eps; FP32 cannot order them.
            out, fallbacks = self.run_fused(np.array([[0., g0 - g1 + eps]]), [0, 1], np.empty((1, 0), np.int64),
                                            list_max=0, dtype=torch.float64, rng=rng)
            self.assertEqual(out.tokens[0, 0].item(), 1 if eps > 0 else 0)
            self.assertGreater(fallbacks, 0)
        out, fallbacks = self.run_fused(np.array([[0., g0 - g1]], np.float32), [0, 1], np.empty((1, 0), np.int64),
                                        list_max=0, dtype=torch.float32, rng=rng)
        self.assertGreater(fallbacks, 0)

    def test_near_threshold_acceptance_is_settled_in_fp64(self):
        import torch
        seed, stream, position = 4321, 2, 50
        u = uniform(seed, stream, position, 1, 0)
        rng = RngKey(seed, stream, [position, position + 1])
        for scale, accepted in [(1 + 1e-12, 1), (1 - 1e-12, 0)]:
            p = u * scale  # target probability of the proposed token 0
            rows = np.array([[0., np.log(1 / p - 1)], [0., 0.]])
            out, fallbacks = self.run_fused(rows, [0, 2], [[0]], list_max=0, dtype=torch.float64, rng=rng)
            self.assertEqual(out.accepted_len[0].item(), accepted)
            self.assertGreater(fallbacks, 0)

    def test_min_p_threshold_is_exact(self):
        import torch
        rows = np.array([[0., -1.25, -3.]])
        for scale in [1 - 1e-9, 1 + 1e-9]:
            # FP32 cannot tell whether token 1 clears min-p; FP64 can.
            params = RequestParams(min_p=float(np.exp(-1.25)) * scale)
            for list_max in [256, 0]:
                for seed in range(20):
                    self.run_fused(rows, [0, 1], np.empty((1, 0), np.int64), list_max=list_max,
                                   dtype=torch.float64, params=params, rng=RngKey(seed, 0, [seed]))

    def test_top_k_selection_with_ties_and_large_k(self):
        import torch
        g = np.random.default_rng(5)
        flat = np.zeros((1, 5000))  # every token ties: radix select and the tie pass
        for k in [1, 100, 2047, 2049, 4999]:
            for list_max in [256, 0]:
                self.run_fused(flat, [0, 1], np.empty((1, 0), np.int64), list_max=list_max,
                               params=RequestParams(top_k=k), dtype=torch.float32)
        steps = np.round(g.normal(size=(3, 3000)) * 8) / 8  # ties at most cut points
        for k in [257, 1000, 2999]:
            self.run_fused(steps, [0, 1, 3], [[-1], [7]], params=RequestParams(top_k=k, min_p=.001),
                           dtype=torch.bfloat16)

    @full_vocabulary
    def test_top_p_over_the_full_row(self):
        import torch
        g = np.random.default_rng(17)
        rows = g.normal(scale=3., size=(2, 151936))
        for k, p in [(0, .9), (1000, .8)]:
            self.run_fused(rows, [0, 2], [[5]], params=RequestParams(top_k=k, top_p=p, temperature=.7),
                           dtype=torch.bfloat16)
        # A flat row: the nucleus spans many sorted chunks.
        flat = g.normal(scale=.5, size=(2, 20000))
        self.run_fused(flat, [0, 1, 2], np.empty((2, 0), np.int64), list_max=0,
                       params=RequestParams(top_p=[.9, .3]), dtype=torch.float32)

    def test_top_p_boundary_is_settled_in_fp64(self):
        import torch
        g = np.random.default_rng(23)
        # 50 tokens of weight exactly 1 and a negligible tail: every prefix sum
        # is an integer, so top_p = count / 50 puts the target on a boundary
        # in both FP64 implementations, and FP32 cannot tell the side.
        rows = g.normal(size=(1, 4096)) - 60
        rows[0, g.choice(4096, 50, replace=False)] = 0
        for count in [3, 40]:
            for list_max in [256, 0]:
                out, fallbacks = self.run_fused(rows, [0, 1], np.empty((1, 0), np.int64), list_max=list_max,
                                                params=RequestParams(top_p=count / 50), dtype=torch.float32)
                if list_max == 0:
                    self.assertGreater(fallbacks, 0)

    @full_vocabulary
    def test_real_vocabulary(self):
        import torch
        g = np.random.default_rng(11)
        rows = g.normal(scale=3., size=(3, 151936))
        mask = packed_mask(g.random((3, 151936)) < .9)
        for k in [300, 5000]:
            self.run_fused(rows, [0, 3], [[17, 4]], params=RequestParams(top_k=k, temperature=.7),
                           mask=mask, dtype=torch.bfloat16)
        self.run_fused(rows[:1], [0, 1], np.empty((1, 0), np.int64), params=RequestParams(temperature=1.3),
                       dtype=torch.bfloat16)

    def test_batch_invariance(self):
        import torch
        from exactsample.cuda import verify
        g = np.random.default_rng(3)
        vocab, counts = 1000, [0, 3, 1, 2, 0]
        params = dict(temperature=[.7, 1., 0., .5, 1.2], top_k=[0, 20, 0, 400, 5], top_p=[.95, .9, 1., 1., .8],
                      min_p=[0., 0., 0., .05, 0.])
        rows = [g.normal(scale=3., size=(c + 1, vocab)) for c in counts]
        drafts = [g.integers(0, vocab, size=c) for c in counts]
        seeds, streams = [11, 22, 33, 44, 55], [1, 2, 3, 4, 5]
        positions = [np.arange(c + 1) + 40 * i for i, c in enumerate(counts)]

        def batch(order):
            sizes = [counts[i] + 1 for i in order]
            ids = np.full((len(order), 3), -1)
            for slot, i in enumerate(order):
                ids[slot, :counts[i]] = drafts[i]
            args = prepared_cuda(np.concatenate([rows[i] for i in order]), np.concatenate([[0], np.cumsum(sizes)]),
                                 ids, dtype=torch.bfloat16,
                                 params=RequestParams(**{k: [v[i] for i in order] for k, v in params.items()}),
                                 rng=RngKey([seeds[i] for i in order], [streams[i] for i in order],
                                            np.concatenate([positions[i] for i in order])))
            out = verify(**args)
            args['plan'].synchronize()
            return [(i, (out.tokens[slot].tolist(), out.accepted_len[slot].item(), out.status[slot].item()))
                    for slot, i in enumerate(order)]

        together = dict(batch([0, 1, 2, 3, 4]))
        self.assertEqual(dict(batch([4, 2, 0, 3, 1])), together)
        for i in range(5):
            self.assertEqual(dict(batch([i]))[i], together[i])
        # 1 to 11 rows run 32 slices per row; 99 rows run 5. The split must not matter.
        for i, output in batch([0, 1, 2, 3, 4] * 9):
            self.assertEqual(output, together[i])


if __name__ == '__main__':
    unittest.main()
