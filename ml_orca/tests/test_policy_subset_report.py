import unittest

from ml_orca.experiments.report_policy_subset import arm_metrics, valid_arm


class PolicySubsetReportTest(unittest.TestCase):
    def test_metrics_and_validity_keep_search_work_and_result_gate_separate(self):
        run = {'plan_rc': 0, 'rows_rc': 0, 'fallback': False, 'optimizer': 'pg_orca',
               'error': '', 'server_failure': False,
               'optimizer_progress': [{'cost': 7.0}],
               'dsl_observability': {'rules': {
                   'a': {'binding_attempts': 5, 'generated_alternatives': 2,
                         'duplicate_alternatives': 1, 'memo_inserted_alternatives': 1},
                   'b': {'binding_attempts': 8, 'generated_alternatives': 0,
                         'duplicate_alternatives': 0, 'memo_inserted_alternatives': 0}}}}
        metrics = arm_metrics(run)
        self.assertEqual(metrics['binding_attempts'], 13)
        self.assertEqual(metrics['memo_expanding_rules'], ['a'])
        self.assertTrue(valid_arm(run, True, {'Node Type': 'Result'}, metrics))
        self.assertFalse(valid_arm(run, False, {'Node Type': 'Result'}, metrics))


if __name__ == '__main__':
    unittest.main()
