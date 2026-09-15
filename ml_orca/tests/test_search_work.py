import unittest
from unittest.mock import patch

from ml_orca.objectives.search_work import observed_work_vector


class SearchWorkTest(unittest.TestCase):
    def test_stages_remain_separate_and_incomplete_streams_are_not_zero(self):
        attempts = dict(complete=True, exclusions=[], attempts=10, evaluations=6)
        costs = dict(complete=True, exclusions=[], events=[{}] * 5,
                     status_counts={'costed': 2, 'duplicate_context': 3})
        checks = dict(complete=True, exclusions=[], status_counts={'prune': {'pruned': 4}})
        prefix = 'ml_orca.objectives.search_work.'
        with patch(prefix + 'candidate_evidence', return_value=attempts), \
             patch(prefix + 'cost_evidence', return_value=costs), \
             patch(prefix + 'search_check_evidence', return_value=checks):
            result = observed_work_vector({})
            self.assertTrue(result['complete'])
            self.assertEqual(result['values']['dsl_budget_skips'], 4)
            self.assertEqual(result['values']['cost_entries_by_status']['costed'], 2)
            self.assertEqual(result['values']['cost_entries'], 5)
            self.assertFalse(result['is_scalar_C'])
            for stream in (attempts, costs, checks):
                stream.update(complete=False, exclusions=['truncated_stream'])
                incomplete = observed_work_vector({})
                self.assertIsNone(incomplete['values'])
                self.assertEqual(incomplete['exclusions'], ['truncated_stream'])
                stream.update(complete=True, exclusions=[])


if __name__ == '__main__':
    unittest.main()
