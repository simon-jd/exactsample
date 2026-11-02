import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
from tests.stat.analysis import g_test, holm, power_bound


class StatisticalAnalysis(unittest.TestCase):
    def test_hand_g_statistic(self):
        import math
        result = g_test([75, 25], [.5, .5])
        self.assertAlmostEqual(result['g'], 2 * (75 * math.log(1.5) + 25 * math.log(.5)))
        self.assertLess(result['p_value'], .00001)

    def test_support_violation(self):
        self.assertEqual(g_test([99, 1], [1, 0])['p_value'], 0)

    def test_holm_stops_at_first_nonrejection(self):
        self.assertEqual(holm([.003, .0049, .0051]), [True, True, True])
        self.assertEqual(holm([.0035, .0001, .5]), [True, True, False])
        self.assertEqual(holm([.004, .0041, .0042]), [False, False, False])

    def test_planned_sample_size(self):
        plan = json.loads((Path(__file__).parent / 'stat_plan.json').read_text())
        self.assertEqual(len(plan['cells']), plan['family_cells'])
        bound = power_bound(plan)
        self.assertGreaterEqual(bound['power_lower_bound'], plan['minimum_power'])

    def test_failed_full_family_cannot_be_reported_qualified(self):
        from tests.stat import run
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'failure.json'
            # Exercise failure accounting without executing a sampler or GPU.
            failed = [{'cell': 'injected-analysis-failure', 'rejected': True}]
            with patch('sys.argv', ['stat', '--backend', 'reference', '--output', str(destination)]), \
                    patch.object(run, 'run_family', return_value=failed) as family:
                self.assertEqual(run.main(), 1)
            report = json.loads(destination.read_text())
            self.assertFalse(report['qualification'])
            self.assertTrue(report['initial_failure'])
            self.assertTrue(report['preregistered_sample_count_used'])
            self.assertEqual(family.call_count, 2)
            self.assertEqual(report['diagnostic_retry'], failed)

    def test_statistics_refuse_to_replace_existing_evidence(self):
        from tests.stat import run
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'old.json'
            destination.write_text('preserve')
            with patch('sys.argv', ['stat', '--output', str(destination)]), \
                    patch.object(run, 'run_family') as family:
                with self.assertRaises(SystemExit):
                    run.main()
            family.assert_not_called()
            self.assertEqual(destination.read_text(), 'preserve')
