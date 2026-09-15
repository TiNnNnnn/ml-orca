"""Credit is conserved; provenance ambiguity is not a causal label."""
import copy
import math
import unittest

from ml_orca.objectives.rule_credit import allocate_gain, root_gain_credit


class RuleCreditTest(unittest.TestCase):
    def test_diamond_roles_background_and_duplicate_invariance(self):
        ancestry = {3: [1, 2, 1], 4: [1, 2, 3]}
        result = allocate_gain(12., [3, 4, 3], ancestry, dsl_share=.5)
        self.assertEqual([(c['sequence'], c['role'], c['credit']) for c in result['credits']],
                         [(1, 'F', 1.5), (2, 'F', 1.5), (3, 'D', 1.5), (4, 'D', 1.5)])
        self.assertEqual(result['unattributed'], 6.)
        self.assertEqual(result, allocate_gain(12., [4, 3], ancestry, dsl_share=.5))
        self.assertEqual(allocate_gain(12., [], {}, dsl_share=1.)['unattributed'], 12.)
        for share in (0., .1, .5, 1.):
            for gain in (0., 1e-200, 40., 1e300):
                r = allocate_gain(gain, [3, 4], ancestry, dsl_share=share)
                self.assertTrue(math.isclose(math.fsum(c['credit'] for c in r['credits']) + r['unattributed'],
                                             gain, rel_tol=1e-12, abs_tol=1e-300))
                self.assertTrue(all(c['credit'] >= 0 for c in r['credits']))

    def test_invalid_or_missing_contract_is_not_a_zero_label(self):
        for value in (None, True, float('nan'), float('inf'), -1., 10**1000):
            with self.subTest(gain=value), self.assertRaises(ValueError):
                allocate_gain(value, [], {}, dsl_share=.5)
        for share in (None, True, float('nan'), -1., 1.01):
            with self.subTest(share=share), self.assertRaises(ValueError):
                allocate_gain(1., [], {}, dsl_share=share)
        for direct, ancestry in (([True], {}), ([0], {}), ([3], {}), ([3], {3: [3]})):
            with self.assertRaises(ValueError):
                allocate_gain(1., direct, ancestry, dsl_share=.5)

    def report(self):
        return dict(audits={'cost': {'complete': True}}, cost_origin_audit={'complete': True},
                    physical_plan_source_audit={'complete': True, 'plans': {
                        '10': {'generators': [1, 2], 'recorded_inserters': [1]},
                        '20': {'generators': [2, 3, 4], 'recorded_inserters': [2, 3]}}},
                    instance_work_ledger={'instances': [dict(sequence=i, rule_hash='same_rule',
                                                            ancestors=list(range(1, i))) for i in range(1, 5)]},
                    root_cost_updates=[dict(event_sequence=1, candidate_sequence=10, previous_candidate_sequence=0,
                                            first_feasible=True, cost_reduction=None, exclusions=[]),
                                       dict(event_sequence=2, candidate_sequence=20, previous_candidate_sequence=10,
                                            first_feasible=False, cost_reduction=40., exclusions=[])])

    def test_new_inserters_only_not_alias_growth_or_common_old_sources(self):
        report = self.report()
        result = root_gain_credit(report, dsl_share=.5)
        self.assertTrue(result['complete'])
        self.assertIsNone(result['events'][0]['allocation'])
        self.assertEqual(result['events'][1]['eligible_new_inserters'], [3])
        self.assertEqual([c['sequence'] for c in result['by_instance']], [1, 2, 3])
        self.assertAlmostEqual(sum(c['D'] + c['F'] for c in result['by_instance']), 20.)
        report['physical_plan_source_audit']['plans']['20']['recorded_inserters'] = [2]
        result = root_gain_credit(report, dsl_share=.5)
        self.assertEqual(result['events'][1]['allocation']['unattributed'], 40.)
        self.assertEqual(result['by_instance'], [])

    def test_missing_failure_and_non_improvements_remain_masked(self):
        for patch in ({'cost_reduction': None}, {'cost_reduction': -2.}, {'exclusions': ['changed_statistics']}):
            report = self.report()
            report['root_cost_updates'][1].update(patch)
            result = root_gain_credit(report, dsl_share=.5)
            self.assertIsNone(result['events'][1]['allocation'])
        report = self.report()
        report['audits']['cost']['complete'] = False
        self.assertFalse(root_gain_credit(report, dsl_share=.5)['complete'])
        report = self.report()
        report['root_cost_updates'].append(copy.deepcopy(report['root_cost_updates'][1]))
        with self.assertRaisesRegex(ValueError, 'duplicate gain'):
            root_gain_credit(report, dsl_share=.5)


if __name__ == '__main__':
    unittest.main()
