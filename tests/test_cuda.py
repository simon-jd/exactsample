"""Real-device checks: missing CUDA skips; EXACTSAMPLE_REQUIRE_CUDA=1 makes it fail."""
import os
import unittest
import numpy as np
from exactsample import RequestParams, RngKey, Status, verify_reference
from tests.cuda_helpers import prepared_cuda


class CudaDifferential(unittest.TestCase):
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
        load_library()  # A missing/broken library on an available GPU is a failure.

    def compare(self, rows, offsets, ids, *, params=None, rng=None, mask=None,
                draft_logits=None, mode='ratio', dispatch='auto', dtype=None, strided=False):
        from exactsample.cuda import verify
        import torch
        params = params or RequestParams()
        rng = rng or RngKey(1234, 3, np.arange(len(rows)))
        args = prepared_cuda(rows, offsets, ids, params=params, rng=rng, mask=mask,
                             draft_logits=draft_logits, dtype=dtype, strided=strided)
        result = verify(**args, mode=mode, dispatch=dispatch)
        args['plan'].synchronize()
        # Oracle sees precisely the same rounded input dtype, including bf16.
        host_rows = args['target_logits'].double().cpu().numpy()
        host_q = args.get('draft_logits')
        if host_q is not None:
            host_q = host_q.double().cpu().numpy()
        expected = verify_reference(host_rows, offsets, ids, params=params, rng=rng,
                                    mask=mask, draft_logits=host_q, mode=mode)
        for field in ['tokens', 'accepted_len', 'status']:
            np.testing.assert_array_equal(getattr(result, field).cpu().numpy(), getattr(expected, field))
        return result

    def test_dtype_stride_and_forced_paths(self):
        import torch
        generator = np.random.default_rng(42)
        rows = generator.normal(size=(6, 35))
        offsets = [0, 1, 3, 6]
        ids = [[-1, -1], [2, -1], [3, 4]]
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for dispatch in ['reference', 'fused']:
                for k in [0, 1, 20, 35]:
                    with self.subTest(dtype=dtype, dispatch=dispatch, k=k):
                        self.compare(rows, offsets, ids, params=RequestParams(top_k=k),
                                     dtype=dtype, dispatch=dispatch, strided=True)

    def test_boundaries_and_packed_masks(self):
        for dispatch in ['reference', 'fused']:
            for p in [.5, np.nextafter(.5, 1), 1.]:
                self.compare(np.zeros((4, 4)), [0, 4], [[0, 1, 2]],
                             params=RequestParams(top_k=4, top_p=p), dispatch=dispatch)
            mask = np.array([[-2147483648, 1]] * 3, np.int32)
            self.compare(np.zeros((3, 35)), [0, 3], [[31, 32]], mask=mask,
                         params=RequestParams(top_k=20), dispatch=dispatch, strided=True)

    def test_status_and_atomicity(self):
        for dispatch in ['reference', 'fused']:
            for bad in [float('nan'), float('inf'), -float('inf')]:
                self.compare([[0., -float('inf')], [bad, bad]], [0, 2], [[1]], dispatch=dispatch)
            self.compare(np.zeros((2, 2)), [0, 2], [[0]],
                         mask=np.zeros((2, 1), np.int32), dispatch=dispatch)
            self.compare(np.zeros((1, 2)), [0, 0, 1], np.empty((2, 0), np.int64), dispatch=dispatch)

    def test_full_q_actual_distribution(self):
        self.compare(np.log([[3., 1.], [1., 1.]]), [0, 2], [[1]],
                     params=RequestParams(draft_kind='full', top_k=1),
                     draft_logits=np.log([[1., 3.], [1., 1.]]), dispatch='reference')
        self.compare([[0., 0.], [0., 0.]], [0, 2], [[1]],
                     params=RequestParams(draft_kind='full'),
                     draft_logits=[[0., -float('inf')], [0., 0.]], dispatch='reference')

    def test_future_target_error_takes_priority_over_q_zero(self):
        self.compare([[0., 0.], [float('nan'), 0.]], [0, 2], [[1]],
                     params=RequestParams(draft_kind='full'),
                     draft_logits=[[0., -float('inf')], [0., 0.]], dispatch='reference')

    def test_coupled_matches_ordinary_and_reordered_batch(self):
        rows = np.array([[0., 1., 2.], [1., 0., 2.], [2., 1., 0.]])
        rng = RngKey(777, 7, [20, 21, 22])
        expected = verify_reference(rows, [0, 1, 2, 3], np.empty((3, 0), np.int64),
                                    rng=RngKey([777] * 3, [7] * 3, [20, 21, 22]))
        drafts = expected.tokens[:2, 0].reshape(1, 2)
        for dispatch in ['reference', 'fused']:
            result = self.compare(rows, [0, 3], drafts, rng=rng, mode='coupled', dispatch=dispatch)
            self.assertEqual(result.tokens.cpu().tolist(), [expected.tokens[:, 0].tolist()])
            self.compare(rows[[2, 0, 1]], [0, 1, 2, 3], np.empty((3, 0), np.int64),
                         rng=RngKey([777] * 3, [7] * 3, [22, 20, 21]), dispatch=dispatch)

    def test_workspace_conflict_and_separate_plans(self):
        import torch
        from exactsample.cuda import verify
        if not hasattr(torch.cuda, '_sleep'):
            self.skipTest('torch.cuda._sleep unavailable for deterministic in-flight fixture')
        args = prepared_cuda(np.zeros((1, 4)), [0, 1], np.empty((1, 0), np.int64))
        other = prepared_cuda(np.zeros((1, 4)), [0, 1], np.empty((1, 0), np.int64))
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        first.wait_stream(torch.cuda.current_stream())
        second.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(first):
            torch.cuda._sleep(100000000)
            verify(**args)
        with torch.cuda.stream(second):
            with self.assertRaisesRegex(RuntimeError, 'in flight'):
                verify(**args)
            verify(**other)
        args['plan'].synchronize()
        other['plan'].synchronize()
        self.assertEqual(args['out'].tokens.cpu().tolist(), other['out'].tokens.cpu().tolist())

    def test_capacity_failure_and_large_k_on_fused_path(self):
        from exactsample.cuda import verify
        import torch
        result = self.compare(np.zeros((1, 100)), [0, 1], np.empty((1, 0), np.int64),
                              params=RequestParams(top_k=65), dispatch='fused')
        self.assertEqual(result.path_used.cpu().tolist(), [1])
        args = prepared_cuda(np.zeros((1, 100)), [0, 1], np.empty((1, 0), np.int64))
        args['target_logits'] = torch.zeros((2, 100), device='cuda')
        with self.assertRaisesRegex(ValueError, 'capacity'):
            verify(**args)

    def test_rng_boundaries_and_mixed_dispatch(self):
        rows = np.zeros((3, 80))
        result = self.compare(rows, [0, 1, 2, 3], np.empty((3, 0), np.int64),
                              params=RequestParams(top_k=[0, 20, 65]),
                              rng=RngKey([0, 2**63, 2**64 - 1], [0, 2**32 - 1, 9],
                                         [0, 2**32 - 1, 100]))
        self.assertEqual(result.path_used.cpu().tolist(), [1, 1, 1])

    def test_output_alias_is_rejected(self):
        from exactsample.cuda import verify
        args = prepared_cuda(np.zeros((1, 4)), [0, 1], np.empty((1, 0), np.int64))
        # aliasing is forbidden even if only one output element would be used.
        args['out'].tokens = args['rng'].seed.reshape(1, 1)
        with self.assertRaisesRegex(RuntimeError, 'alias'):
            verify(**args)

    def test_noncontiguous_outputs(self):
        import torch
        from exactsample.cuda import verify
        args = prepared_cuda(np.zeros((3, 4)), [0, 1, 3], [[-1], [2]], strided=True)
        out = args['out']
        backing = torch.full((2, 4), -777, dtype=torch.int64, device='cuda')
        out.tokens = backing[:, ::2]
        for field in ['accepted_len', 'status', 'path_used']:
            setattr(out, field, torch.empty(4, dtype=torch.int32, device='cuda')[::2])
        result = verify(**args)
        args['plan'].synchronize()
        self.assertTrue(torch.all(backing[:, 1::2] == -777).item())
        self.assertEqual(result.status.cpu().tolist(), [0, 0])

    def test_invalid_offsets_do_not_write_tokens(self):
        import torch
        from exactsample.cuda import verify
        for offsets in [[1, 1, 2], [0, 2, 1], [0, -1, 2], [0, 1, 3]]:
            args = prepared_cuda(np.zeros((2, 4)), offsets, np.empty((2, 0), np.int64))
            out = verify(**args)
            args['plan'].synchronize()
            self.assertEqual(out.status.cpu().tolist(), [int(Status.INVALID_INPUT)] * 2)
            self.assertEqual(out.tokens.cpu().tolist(), [[-1], [-1]])

    @unittest.skipIf(os.environ.get('EXACTSAMPLE_SANITIZER') == '1', 'too slow under compute-sanitizer')
    def test_real_vocab_small_k_and_temperature(self):
        # A real Qwen3 vocabulary shape; the values are synthetic, not model logits.
        import torch
        rows = np.random.default_rng(989).normal(size=(2, 151936))
        for k in [0, 20]:
            for dispatch in ['reference', 'fused']:
                self.compare(rows, [0, 2], [[123]], params=RequestParams(top_k=k),
                             dtype=torch.bfloat16, dispatch=dispatch)


if __name__ == '__main__':
    unittest.main()
