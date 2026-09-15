from copy import deepcopy
import unittest

from ml_orca.experiments.generate_priority_control import (
    policy_text, prioritized_rules, scaled_orderings, shuffled_rules, uncapped_orderings)
from ml_orca.experiments.prepare_priority_comparison import policy_roundtrip_equal
from ml_orca.experiments.report_priority_scale import optimizer_stage_costs, productive_response_points
from ml_orca.objectives.priority import priority_pair
from ml_orca.objectives.search_work import WORK_CONTRACT


def policy_cell(name, cost=10):
    return dict(unit=dict(workload='w', query='1', query_crc32='12345678', scenario=1, policy=name),
        inputs=dict(query_sql='select 1', graph_snapshot={'crc32': '12345678'}, catalog_snapshot={'crc32': '87654321'},
                    stats_experiment_document=None, stats_experiment_requests=None,
                    candidate_policy=[dict(rule_hash='a', enabled=True, placement='cbo', priority=1,
                                           budget=dict(per_query=4, per_rule=0, per_node=0))]),
        admission=dict(feature_integrity_verified=True, feature_exclusions=[], model_training_eligible=False),
        response=dict(status='incomplete', exclusions=['incomplete_timing_schedule'], comparison_path='/frozen/comparison.json',
            search=dict(validation=dict(complete=True, exclusions=[], runtime_settings='same'),
                terminal_quality=dict(available=True, exclusions=[], optimizer_cost=cost),
                trace_audits={k: dict(complete=True, exclusions=[]) for k in
                    ('attempts', 'origins', 'cost_lifecycle', 'cost_origin_audit', 'physical_plan_source_audit')},
                work=dict(contract=WORK_CONTRACT, complete=True, exclusions=[], values=dict(
                    dsl_dispatched_attempts=10, dsl_evaluated_attempts=6, dsl_budget_skips=4,
                    cost_entries=5, cost_entries_by_status={'costed': 2, 'duplicate_context': 3},
                    search_checks_by_stage_and_status={'prune': {'pruned': 4}})))))


class PriorityControlTest(unittest.TestCase):
    def test_productive_response_normalizes_each_query_without_dropping_zero_output(self):
        report = {'records': [dict(workload='w', query='q', rows=[
            dict(complete=True, order='default', size=8, planning_ms_median=10,
                 binding_attempts=20, memo_inserted_alternatives=0),
            dict(complete=True, order='default', size=16, planning_ms_median=30,
                 binding_attempts=100, memo_inserted_alternatives=7),
            dict(complete=True, order='reverse', size=16, planning_ms_median=99,
                 binding_attempts=999, memo_inserted_alternatives=9)])]}
        self.assertEqual(productive_response_points(report), [
            dict(workload='w', query='q', enabled_rules=8, memo_inserted_alternatives=0,
                 planning_ratio=1, binding_ratio=1),
            dict(workload='w', query='q', enabled_rules=16, memo_inserted_alternatives=7,
                 planning_ratio=3, binding_ratio=5)])

    def test_explicit_ranking_is_high_to_low_and_preserves_other_fields(self):
        rows = [dict(rule_hash=h, enabled=True, placement='cbo', priority=0,
                     budget=dict(per_node=0, per_rule=0, per_query=0))
                for h in ('a', 'b', 'c')]
        result = prioritized_rules(rows, ['c', 'a'])
        self.assertEqual([r['priority'] for r in result], [1, 0, 2])
        self.assertEqual([{**r, 'priority': 0} for r in result], rows)
        with self.assertRaises(ValueError):
            prioritized_rules(rows, ['c', 'c'])

    def test_optimizer_stage_cost_parser_retains_order_and_rejects_nonfinite(self):
        text = '[OPT]: stage 0 completed in 12ms,  plan with cost 8.5 was found\n' \
               '[OPT]: stage 1 completed in 3ms, plan with cost 7e0 was found'
        self.assertEqual(optimizer_stage_costs(text), [
            {'stage': 0, 'elapsed_ms': 12, 'cost': 8.5},
            {'stage': 1, 'elapsed_ms': 3, 'cost': 7.0}])
        with self.assertRaises(ValueError):
            optimizer_stage_costs('[OPT]: stage 0 completed in 1ms, plan with cost 1e309 was found')

    def test_policy_roundtrip_ignores_only_derived_candidate_position(self):
        row = dict(rule_hash='0'*16, source_line=1, enabled=True, placement='cbo', phase='explore',
                   effect='preserves_join_graph', priority=0, order='bottom_up', fixpoint=False,
                   budget=dict(per_node=0, per_rule=0, per_query=0),
                   candidate_list='cbo', candidate_list_position=9)
        actual = [{**row, 'candidate_list_position': 0}]
        self.assertTrue(policy_roundtrip_equal([row], actual))
        self.assertFalse(policy_roundtrip_equal([row], [{**actual[0], 'priority': 1}]))
        self.assertFalse(policy_roundtrip_equal([row], [{**actual[0], 'candidate_list': None}]))
        disabled = {**row, 'enabled': False, 'candidate_list': None, 'candidate_list_position': None}
        self.assertTrue(policy_roundtrip_equal([disabled], [disabled]))

    def test_scaled_controls_use_nested_outcome_blind_membership(self):
        rows = [dict(rule_hash=f'{i:016x}', enabled=i != 1, placement='cbo', priority=0,
                     budget=dict(per_node=1, per_rule=1, per_query=128)) for i in range(7)]
        arms = scaled_orderings(rows, [2, 6, 4], 71)
        self.assertEqual(list(arms), [f's{s:03d}-{a}' for s in (2, 4, 6)
                                     for a in ('default', 'reverse', 'random')])
        selected = []
        for size in (2, 4, 6):
            same = [{r['rule_hash'] for r in arms[f's{size:03d}-{arm}'] if r['enabled']}
                    for arm in ('default', 'reverse', 'random')]
            self.assertTrue(all(value == same[0] for value in same))
            self.assertEqual(len(same[0]), size)
            selected.append(same[0])
        self.assertLess(selected[0], selected[1])
        self.assertLess(selected[1], selected[2])
        for invalid in ([], [0], [7], [2, 2], [True]):
            with self.assertRaises(ValueError):
                scaled_orderings(rows, invalid, 71)

    def test_uncapped_controls_preserve_membership_and_only_order_differs(self):
        rows = [dict(rule_hash=f'{i:016x}', enabled=i != 1, placement='cbo', priority=-7,
                     budget=dict(per_node=1, per_rule=1, per_query=128)) for i in range(5)]
        before = deepcopy(rows)
        arms = uncapped_orderings(rows, 71)
        self.assertEqual(rows, before)
        self.assertEqual(arms, uncapped_orderings(rows, 71))
        for rules in arms.values():
            for old, new in zip(rows, rules):
                self.assertEqual(new['budget'], dict(per_node=0, per_rule=0, per_query=0))
                self.assertEqual({**new, 'budget': old['budget'], 'priority': old['priority']}, old)
        self.assertEqual([r['priority'] for r in arms['reverse']], [1, 0, 3, 4, 5])
        with self.assertRaises(ValueError):
            uncapped_orderings([{**rows[0], 'placement': 'rbo'}], 71)

    def test_cost_priority_supervision_is_paired_and_independent_of_timing(self):
        left, right = policy_cell('a', 10), policy_cell('b', 2)
        right['inputs']['candidate_policy'][0]['priority'] = 99
        before = deepcopy((left, right))
        result = priority_pair(left, right)
        self.assertEqual(result['exclusions'], [])
        self.assertEqual(result['cost']['observed_winner'], 'right')
        self.assertGreater(result['cost']['log1p_margin'], 0)
        self.assertTrue(result['trace_complete'])
        self.assertTrue(all(v == 0 for v in result['work_delta'].values()))
        reverse = priority_pair(right, left)
        self.assertEqual(reverse['cost']['log1p_margin'], -result['cost']['log1p_margin'])
        self.assertEqual((left, right), before)
        for row in (left, right):
            row['response'].update(status='complete', planning_ms_median=999999, execution_ms_median=1)
        self.assertEqual(result, priority_pair(left, right))
        right['response']['search']['terminal_quality']['optimizer_cost'] = 10
        self.assertEqual(priority_pair(left, right)['cost']['observed_winner'], 'tie')
        for row in (left, right):
            row['response']['search']['terminal_quality']['optimizer_cost'] = 0
        self.assertEqual(priority_pair(left, right)['cost']['log1p_margin'], 0)

    def test_priority_pairs_do_not_compare_budgets_membership_inputs_or_failed_arms(self):
        left, right = policy_cell('a'), policy_cell('b')
        for mutate in (
            lambda r: r['unit'].update(scenario=2),
            lambda r: r['response'].update(comparison_path='/other/comparison.json'),
            lambda r: r['inputs']['candidate_policy'][0]['budget'].update(per_query=0),
            lambda r: r['inputs']['candidate_policy'][0].update(enabled=False),
            lambda r: r['inputs']['candidate_policy'][0].update(placement='rbo'),
            lambda r: r['inputs'].update(stats_experiment_document='rows: 8'),
            lambda r: r['response']['search']['validation'].update(runtime_settings='different'),
            lambda r: r['response']['search']['validation'].update(complete=False, exclusions=['plan_timeout']),
            lambda r: r['admission'].update(feature_integrity_verified=False),
        ):
            changed = deepcopy(right)
            mutate(changed)
            result = priority_pair(left, changed)
            self.assertTrue(result['exclusions'])
            self.assertIsNone(result['cost'])
            self.assertIsNone(result['work_delta'])
        self.assertIsNone(priority_pair(left, None)['cost'])
        self.assertIn('missing_policy_cell', priority_pair(left, None)['exclusions'])

    def test_quality_and_work_masks_are_separate_never_zero_fill_incomplete_work(self):
        left, right = policy_cell('a'), policy_cell('b')
        right['response']['search']['work']['values']['cost_entries_by_status'] = {'costed': 5}
        result = priority_pair(left, right)
        self.assertEqual(result['work_delta']['cost_entries'], 0)
        self.assertEqual(result['work_delta']['cost_entries_by_status/costed'], -3)
        self.assertEqual(result['work_delta']['cost_entries_by_status/duplicate_context'], 3)
        right['response']['search']['work'].update(complete=False, values=None, exclusions=['truncated'])
        result = priority_pair(left, right)
        self.assertIsNone(result['work_delta'])
        self.assertIsNotNone(result['cost'])
        for value in (None, True, -1, float('inf'), float('nan'), 10**400):
            right['response']['search']['terminal_quality']['optimizer_cost'] = value
            self.assertIsNone(priority_pair(left, right)['cost'])
        for change in ({'dsl_dispatched_attempts': 11}, {'cost_entries': True},
                       {'cost_entries_by_status': {'costed': 99}}):
            changed = policy_cell('b')
            changed['response']['search']['work']['values'].update(change)
            self.assertIsNone(priority_pair(left, changed)['work_delta'])
        missing = policy_cell('b')
        missing['response']['search']['trace_audits'].pop('origins')
        self.assertFalse(priority_pair(left, missing)['trace_complete'])

    def test_only_enabled_priorities_change_and_budget_is_explicit(self):
        rows = [dict(rule_hash=f'{i:016x}', enabled=i != 1, placement='cbo', phase='explore',
                     effect='preserves_join_graph', priority=-9, order='bottom_up', fixpoint=False,
                     budget=dict(per_node=2, per_rule=3, per_query=4)) for i in range(12)]
        before = deepcopy(rows)
        result = shuffled_rules(rows, 20260914)
        self.assertEqual(rows, before)
        self.assertEqual(result, shuffled_rules(rows, 20260914))
        self.assertEqual(result, list(reversed(shuffled_rules(list(reversed(rows)), 20260914))))
        self.assertEqual(result[1], rows[1])
        self.assertEqual(sorted(r['priority'] for r in result if r['enabled']), list(range(1, 12)))
        for old, new in zip(rows, result):
            self.assertEqual({**new, 'priority': old['priority']}, old)
        text = policy_text(result)
        for field, value in [('per_node', 2), ('per_rule', 3), ('per_query', 4)]:
            self.assertEqual(text.count(f'    {field}: {value}\n'), len(rows))
        self.assertIn('  enabled: false\n', text)
        with self.assertRaises(ValueError):
            shuffled_rules(rows + [rows[0]], 0)
        with self.assertRaises(ValueError):
            shuffled_rules([{**rows[0], 'placement': 'rbo'}], 0)


if __name__ == '__main__':
    unittest.main()
