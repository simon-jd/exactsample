"""CPU-only cancellation evidence guards, including Python optimized mode."""
import unittest

from integration.vllm_lifecycle import validate_cancelled


class CancellationGuardTests(unittest.TestCase):
    def setUp(self):
        self.cancelled = {"abort_called": True, "finish_reason": "abort",
                          "token_ids": [264], "tokens_at_abort": [264]}

    def test_exact_nonempty_abort_prefix_passes(self):
        validate_cancelled(self.cancelled, 96)

    def test_token_delivered_after_abort_fails(self):
        self.cancelled["token_ids"] = [264, 42]
        with self.assertRaises(AssertionError):
            validate_cancelled(self.cancelled, 96)

    def test_changed_prefix_fails(self):
        self.cancelled["token_ids"] = [42]
        with self.assertRaises(AssertionError):
            validate_cancelled(self.cancelled, 96)

    def test_non_abort_finish_fails(self):
        for reason in ("length", "stop", "error", None):
            with self.subTest(reason=reason):
                self.cancelled["finish_reason"] = reason
                with self.assertRaises(AssertionError):
                    validate_cancelled(self.cancelled, 96)

    def test_abort_not_called_fails(self):
        self.cancelled["abort_called"] = False
        with self.assertRaises(AssertionError):
            validate_cancelled(self.cancelled, 96)

    def test_empty_prefix_fails(self):
        self.cancelled.update(token_ids=[], tokens_at_abort=[])
        with self.assertRaises(AssertionError):
            validate_cancelled(self.cancelled, 96)

    def test_completed_budget_before_abort_fails(self):
        with self.assertRaises(AssertionError):
            validate_cancelled(self.cancelled, 1)


if __name__ == "__main__":
    unittest.main()
