"""Utility contract tests: C matters, interactions are not independent Top-K."""
import unittest

from ml_orca.objectives.rule_utility import net_utility, rank_policy_predictions


CONTRACT = {'plan_cost_scale': 100., 'search_work_scale': 10.}


def prediction(policy, d, f, c, query='q'):
    return dict(query=query, policy=policy, status='complete', direct_gain=d, future_gain=f, search_work=c)


class RuleUtilityTest(unittest.TestCase):
    def test_shared_scale_and_predicted_search_penalty(self):
        score = net_utility(40, 60, 2, **CONTRACT)
        self.assertAlmostEqual(score['utility'], .8)
        self.assertAlmostEqual(net_utility(40, 60, 12, **CONTRACT)['utility'], -.2)
        # Changing physical units with their reference scales preserves utility.
        scaled = net_utility(400, 600, 2000, plan_cost_scale=1000, search_work_scale=10000)
        self.assertEqual(score, scaled)
        self.assertEqual(net_utility(40, 60, 2, future_weight=0, search_weight=0, **CONTRACT)['utility'], .4)

    def test_unknown_invalid_and_overflow_never_become_zero(self):
        for value in (None, True, '1', float('nan'), float('inf'), -1):
            with self.subTest(search_work=value), self.assertRaises(ValueError):
                net_utility(1, 1, value, **CONTRACT)
        for key in CONTRACT:
            with self.assertRaises(ValueError):
                net_utility(1, 1, 1, **{**CONTRACT, key: 0})
        with self.assertRaises(ValueError):
            net_utility(1e308, 1e308, 0, plan_cost_scale=1e-308, search_work_scale=1)

    def test_dependency_bundle_and_order_have_distinct_values(self):
        rows = [prediction('baseline', 0, 0, 0), prediction('A', 0, 0, 2),
                prediction('B', 0, 0, 1), prediction('A_then_B', 10, 90, 3),
                prediction('B_then_A', 10, 0, 3)]
        ranked = rank_policy_predictions(rows, query_weights={'q': 1},
                                         policies=[r['policy'] for r in rows], **CONTRACT)
        self.assertEqual(ranked[0]['policy'], 'A_then_B')
        self.assertGreater(ranked[0]['utility'], 0)
        self.assertLess(next(r['utility'] for r in ranked if r['policy'] == 'A'), 0)

    def test_workload_frequency_not_per_query_oracle(self):
        rows = [prediction('A', 100, 0, 0, 'rare'), prediction('B', 0, 0, 0, 'rare'),
                prediction('A', 0, 0, 0, 'hot'), prediction('B', 20, 0, 0, 'hot')]
        ranked = rank_policy_predictions(rows, query_weights={'rare': 1, 'hot': 9},
                                         policies=['A', 'B'], **CONTRACT)
        self.assertEqual(ranked[0]['policy'], 'B')
        self.assertAlmostEqual(ranked[0]['utility'], .18)

    def test_missing_failed_duplicate_or_unknown_arm_rejected(self):
        rows = [prediction('A', 1, 0, 1), prediction('B', 2, 0, 1)]
        for bad in (rows[:1], rows + rows[:1], [rows[0], {**rows[1], 'status': 'failed'}],
                    [rows[0], {**rows[1], 'query': 'unknown'}]):
            with self.assertRaises(ValueError):
                rank_policy_predictions(bad, query_weights={'q': 1}, policies=['A', 'B'], **CONTRACT)


if __name__ == '__main__':
    unittest.main()
