"""CPU checks of mapping/fallback decisions; not an engine integration test."""
from types import SimpleNamespace
import unittest
import numpy as np
from exactsample.vllm_bridge import grammar_mask_rows, unsupported_request


class BridgeMapping(unittest.TestCase):
    def test_constraint_order_differs_from_batch_order(self):
        # Request b owns rows 1..3, a owns row 0, c is unconstrained row 4.
        masks = np.array([[10], [11], [12], [13]], np.int32)
        actual = grammar_mask_rows(['a', 'b', 'c'], [0, 1, 4, 5], ['b', 'a'], masks, 8)
        np.testing.assert_array_equal(actual, [[13], [10], [11], [12], [-1]])

    def test_bad_mask_layout_fails_before_use(self):
        for masks in [np.ones((2, 1), np.int32), np.ones((1, 1), np.int64),
                      np.ones((1, 2), np.int32)]:
            with self.assertRaises(ValueError):
                grammar_mask_rows(['a'], [0, 1], ['a'], masks, 8)
        with self.assertRaises(ValueError):
            grammar_mask_rows(['a'], [0, 1], ['missing'], np.ones((1, 1), np.int32), 8)

    def test_unsupported_processors_fall_back(self):
        self.assertIsNone(unsupported_request(SimpleNamespace()))
        for field, value in [('n', 2), ('presence_penalty', .1), ('frequency_penalty', .1),
                             ('repetition_penalty', 1.1), ('min_tokens', 1), ('logprobs', 0),
                             ('prompt_logprobs', 0), ('bad_words', ['bad']),
                             ('logit_bias', {1: 1}), ('thinking_token_budget', 0)]:
            with self.subTest(field=field):
                self.assertEqual(unsupported_request(SimpleNamespace(**{field: value})), field)
