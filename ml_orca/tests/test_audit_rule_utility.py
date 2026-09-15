"""No cross-context/epoch gains and no fake gain for first feasibility."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from ml_orca.trace.audit_rule_utility import (
    audit_utility_run, best_cost_ledger, comparison_runs, cost_origin_evidence, instance_work_ledger,
    physical_plan_sources, root_search_progress)
from ml_orca.trace.profile_rule_candidates import binding_origin_evidence
from ml_orca.encoding.rule_history_encoding import encode_observations


class RuleUtilityAuditTest(unittest.TestCase):
    def test_progress_uses_update_watermark_and_masks_legacy_or_inconsistent_traces(self):
        run = dict(experiment_outcomes=[dict(cost_progress_version=1, rule_candidates=10,
                   cost_candidates=9, search_checks=20, stats_lifecycle_events=2, optimizer_cost=3)],
                   cost_events=[dict(sequence=1, cost_kind='computed', cost=5, preceding_rule_candidates=1),
                                dict(sequence=2, cost_kind='computed', cost=3, preceding_rule_candidates=2)],
                   cost_lifecycle_events=[dict(sequence=i, status='best_updated', candidate_sequence=i,
                       previous_candidate_sequence=i-1, preceding_rule_candidates=i+3,
                       preceding_cost_candidates=i+5, preceding_search_checks=i+7, stats_lifecycle_sequence=2)
                       for i in (1, 2)])
        updates = [dict(event_sequence=i, candidate_sequence=i, first_feasible=i == 1, exclusions=[])
                   for i in (1, 2)]
        work = dict(complete=True, exclusions=[], values={'cost_entries': 9})
        result = root_search_progress(run, updates, work)
        self.assertTrue(result['complete'], result['exclusions'])
        self.assertEqual([p['work']['cost_entries'] for p in result['points']], [6, 7])
        self.assertEqual([p['candidate_sequence'] for p in result['points']], [1, 2])
        self.assertEqual(result['terminal_work']['cost_entries'], 9)
        for patch_event in ({'preceding_cost_candidates': None}, {'preceding_cost_candidates': 1},
                            {'preceding_cost_candidates': 10}, {'preceding_rule_candidates': True},
                            {'preceding_rule_candidates': None},
                            {'preceding_search_checks': 0}, {'stats_lifecycle_sequence': 1}):
            bad = deepcopy(run)
            bad['cost_lifecycle_events'][1].update(patch_event)
            with self.subTest(patch_event=patch_event):
                self.assertIsNone(root_search_progress(bad, updates, work)['points'])
        for change in ({'cost_progress_version': None}, {'optimizer_cost': 4}):
            bad = deepcopy(run)
            bad['experiment_outcomes'][0].update(change)
            self.assertFalse(root_search_progress(bad, updates, work)['complete'])
        self.assertFalse(root_search_progress(run, [], work)['complete'])
        self.assertFalse(root_search_progress(run, updates, {**work, 'complete': False})['complete'])

    def test_physical_sources_follow_cost_time_children_and_deduplicate_diamond(self):
        def cost(seq, *children):
            return dict(sequence=seq, status='costed', cost_kind='computed', group=seq,
                        optimization_context=0, child_contexts=[
                            dict(cost_candidate_sequence=c, group=c, optimization_context=0) for c in children])
        costs = [cost(1), cost(2, 1), cost(3, 1), cost(4, 2, 3), cost(5)]
        sources = {seq: dict(generators=[], exposed=[], recorded_inserters=[]) for seq in range(1, 6)}
        sources[1] = dict(generators=[10, 11], exposed=[9], recorded_inserters=[10])
        sources[5] = dict(generators=[99], exposed=[], recorded_inserters=[99])
        plans = physical_plan_sources(costs, sources, [4, 4])
        self.assertEqual(list(plans), [4])
        self.assertEqual(plans[4]['cost_candidates'], [1, 2, 3, 4])
        self.assertEqual(plans[4]['generators'], [10, 11])
        self.assertEqual(plans[4]['recorded_inserters'], [10])
        self.assertEqual(plans[4]['exposed_inputs'], [9])
        self.assertEqual(plans[4]['root_only_generators'], [])
        self.assertEqual(plans[4]['no_known_generator_candidates'], [2, 3, 4])
        for bad_child in (0, 4, 5, True, 999):
            bad = costs[:3] + [cost(4, bad_child)] + costs[4:]
            with self.subTest(child=bad_child), self.assertRaises(ValueError):
                physical_plan_sources(bad, sources, [4])
        for changed in ({'child_contexts': None}, {'status': 'pruned', 'cost_kind': 'lower_bound'},
                        {'child_contexts': [dict(cost_candidate_sequence=1, group=1, optimization_context=7)]}):
            bad = costs[:3] + [{**costs[3], **changed}] + costs[4:]
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                physical_plan_sources(bad, sources, [4])

    def test_physical_sources_allow_deep_trees_without_python_recursion(self):
        costs = [dict(sequence=i, status='costed', cost_kind='computed', group=i, optimization_context=0,
                      child_contexts=[] if i == 1 else [dict(cost_candidate_sequence=i-1,
                                                            group=i-1, optimization_context=0)])
                 for i in range(1, 1501)]
        sources = {i: dict(generators=[], exposed=[], recorded_inserters=[]) for i in range(1, 1501)}
        self.assertEqual(len(physical_plan_sources(costs, sources, [1500])[1500]['cost_candidates']), 1500)

    def test_cost_sources_are_at_event_instances_not_latest_group_aliases(self):
        attempts = [dict(sequence=i, rule_hash='a', evaluated=True, status='ready_cbo') for i in (1, 2)]
        origin = dict(candidate_sequence=1, rule_hash='a', origin_depth=1, group=7, group_expression=2,
                      target_path='r/0', relation='memo_consumes', outcome='memo_inserted')
        event = dict(sequence=1, preceding_rule_candidates=2, group=7, group_expression=5,
                     origin_chain=[dict(group=7, group_expression=2)],
                     dsl_origin_instances=[origin, {**origin, 'candidate_sequence': 2, 'outcome': 'memo_duplicate'}])
        run = dict(experiment_outcomes=[dict(cost_origin_trace_version=1)], cost_events=[event])
        self.assertEqual(cost_origin_evidence(run, attempts)['sources'][1]['generators'], [1, 2])
        for changed in ({'candidate_sequence': 3}, {'candidate_sequence': True}, {'candidate_sequence': []},
                        {'rule_hash': 'b'}, {'origin_depth': 2}, {'group_expression': 9},
                        {'target_path': []}, {'relation': 'guess'}, {'outcome': 'unknown'}):
            with self.subTest(changed=changed):
                bad = {**run, 'cost_events': [{**event, 'dsl_origin_instances': [{**origin, **changed}]}]}
                self.assertFalse(cost_origin_evidence(bad, attempts)['complete'])
        duplicate = {**run, 'cost_events': [{**event, 'dsl_origin_instances': [origin, origin]}]}
        self.assertIn('duplicate_cost_origin_position', cost_origin_evidence(duplicate, attempts)['exclusions'])
        missing = {**run, 'cost_events': [{**event, 'dsl_origin_instances': None}]}
        self.assertIsNone(cost_origin_evidence(missing, attempts)['sources'])
        empty = {**run, 'cost_events': [{**event, 'dsl_origin_instances': []}]}
        self.assertTrue(cost_origin_evidence(empty, attempts)['complete'])
        self.assertFalse(cost_origin_evidence({**run, 'experiment_outcomes': [{}]}, attempts)['complete'])

    def test_instance_diamond_work_is_unique_and_failed_attempts_are_retained(self):
        rows = [dict(sequence=i, rule_hash='a' if i < 3 else 'b', evaluated=True,
                     status='ready_cbo' if i < 4 else 'constraint_rejected', match_us=i)
                for i in range(1, 5)]
        edges = [dict(src_candidate_sequence=a, dst_candidate_sequence=b)
                 for a, b in ((1, 2), (1, 3), (2, 4), (3, 4), (1, 2))]
        costs = {10: dict(generators=[2, 3], exposed=[1]), 11: dict(generators=[], exposed=[1]),
                 12: dict(generators=[1, 3], exposed=[])}
        ledger = instance_work_ledger(rows, edges, costs)
        a, b, c, rejected = ledger['instances']
        self.assertEqual(a['descendant_attempts'], [2, 3, 4])
        self.assertEqual(rejected['ancestors'], [1, 2, 3])
        self.assertEqual(rejected['status'], 'constraint_rejected')
        self.assertEqual(a['generator_cost_events'], [12])
        self.assertEqual(a['descendant_cost_events'], [10])  # Event 12 not charged twice.
        self.assertEqual(a['exposed_input_cost_events'], [10, 11])  # Exposure is not generation.
        self.assertIsNone(a['self_diagnostic_us']['constraint_us'])  # Missing is not zero.
        self.assertEqual(ledger['work_accounting']['generator_linked_cost_events'], 2)
        self.assertEqual(ledger['work_accounting']['no_known_generator_cost_events'], 1)
        with self.assertRaisesRegex(ValueError, 'precede'):
            instance_work_ledger(rows, edges + [dict(src_candidate_sequence=4, dst_candidate_sequence=1)], costs)

    def test_v4_retains_distinct_producers_and_rejects_false_source_links(self):
        candidates = [dict(sequence=i, rule_hash='a', status='ready_cbo', evaluated=True) for i in (1, 2)]
        candidates.append(dict(sequence=3, rule_hash='b', status='match_rejected', evaluated=True))
        edge = dict(scheduler='cbo', src_rule='a', dst_rule='b', dst_candidate_sequence=3,
                    candidate_status='match_rejected', src_target_path='r/0', dst_binding_path='r/1',
                    dst_source_path=None, binding_group=1, binding_group_expression=0,
                    producer_relation='memo_consumes', relation='binding_observed', producer_outcome='memo_rehashed')
        edges = [{**edge, 'binding_edge_sequence': i, 'src_candidate_sequence': i} for i in (1, 2)]
        run = dict(experiment_outcomes=[dict(binding_edge_trace_version=4, binding_origin_edges=2)], rule_edges=edges)
        with patch('ml_orca.trace.profile_rule_candidates.candidate_evidence', return_value={
                'rows': candidates, 'complete': True, 'exclusions': []}):
            self.assertTrue(binding_origin_evidence(run)['complete'])
            for bad in (None, 0, True, 3, 4):
                changed = {**run, 'rule_edges': [edges[0], {**edges[1], 'src_candidate_sequence': bad}]}
                with self.subTest(source=bad):
                    self.assertIn('unresolved_binding_edge_producer_instance', binding_origin_evidence(changed)['exclusions'])
            changed = {**run, 'rule_edges': [edges[0], {**edges[1], 'src_candidate_sequence': 1}]}
            self.assertIn('duplicate_binding_position', binding_origin_evidence(changed)['exclusions'])
            candidates[1]['status'] = 'constraint_rejected'
            self.assertIn('unresolved_binding_edge_producer_instance', binding_origin_evidence(run)['exclusions'])

    def test_history_keeps_instances_without_duplicating_equal_trees(self):
        candidates = [dict(sequence=i, rule_hash='a', status='ready_cbo', evaluated=True) for i in (1, 2)]
        edge = dict(src_rule='a', dst_rule='a', src_target_path='r', dst_binding_path='r',
                    producer_relation='memo_consumes', producer_outcome='memo_inserted',
                    src_candidate_sequence=1, dst_candidate_sequence=2)
        tree = {'nodes': [{'path': 'r'}], 'root': 0}
        prefix = 'ml_orca.encoding.rule_history_encoding.'
        with patch(prefix + 'candidate_evidence', return_value={'complete': True, 'rows': candidates}), \
             patch(prefix + 'binding_origin_evidence', return_value={'complete': True, 'edges': [edge],
                   'producer_instance_coverage': 'validated_source_attempts'}), \
             patch(prefix + 'stats_timeline', return_value={'complete': True}), \
             patch(prefix + 'group_expression_tree', return_value=tree):
            encoded = encode_observations({})
        self.assertEqual(len(encoded['trees']), 1)
        self.assertEqual(encoded['contexts'][0]['attempts'], 2)
        self.assertEqual([r['sequence'] for r in encoded['instances']], [1, 2])
        self.assertEqual(encoded['edges'][0]['src_candidate_sequence'], 1)
        self.assertEqual(encoded['edges'][0]['dst_candidate_sequence'], 2)

    def setUp(self):
        first = dict(sequence=1, group=1, optimization_context=2, search_stage=0,
                     stats_lifecycle_sequence=3, required_columns=1, required_order_columns=0,
                     required_order_matching=0, required_distribution_type=7,
                     status='costed', cost_kind='computed', cost=100.)
        self.costs = [first, {**first, 'sequence': 2, 'cost': 60.}]
        self.events = [dict(sequence=1, status='best_updated', candidate_sequence=1,
                            previous_candidate_sequence=0, group=1, optimization_context=2),
                       dict(sequence=2, status='best_updated', candidate_sequence=2,
                            previous_candidate_sequence=1, group=1, optimization_context=2)]

    def test_delayed_comparable_gain_not_infinite_initial_gain(self):
        first, second = best_cost_ledger(self.costs, self.events)
        self.assertTrue(first['first_feasible'])
        self.assertIsNone(first['cost_reduction'])
        self.assertEqual(second['cost_reduction'], 40.)

    def test_context_statistics_and_property_changes_are_masked(self):
        for field in ('optimization_context', 'group', 'search_stage', 'stats_lifecycle_sequence',
                      'required_columns', 'required_order_columns', 'required_order_matching',
                      'required_distribution_type'):
            costs = [self.costs[0], {**self.costs[1], field: self.costs[1][field] + 1}]
            with self.subTest(field=field):
                row = best_cost_ledger(costs, self.events)[1]
                self.assertIsNone(row['cost_reduction'])
                self.assertIn('changed_' + field, row['exclusions'])

    def test_missing_or_lower_bound_cost_cannot_become_gain(self):
        for patch in ({'cost': None}, {'cost': float('nan')}, {'status': 'pruned', 'cost_kind': 'lower_bound'},
                      {'stats_lifecycle_sequence': None}):
            with self.subTest(patch=patch):
                row = best_cost_ledger([self.costs[0], {**self.costs[1], **patch}], self.events)[1]
                self.assertIsNone(row['cost_reduction'])
                self.assertTrue(row['exclusions'])

    def test_non_improvement_is_not_relabelled_as_positive(self):
        row = best_cost_ledger([self.costs[0], {**self.costs[1], 'cost': 120}], self.events)[1]
        self.assertEqual(row['cost_reduction'], -20)

    def test_run_pointer_and_missing_previous_candidate(self):
        run = {'candidate_events': [], 'experiment_outcomes': []}
        self.assertEqual(list(comparison_runs({'a/b': [run]})), [('/a~1b/0', run)])
        row = best_cost_ledger(self.costs[1:], self.events)[1]
        self.assertIsNone(row['cost_reduction'])

    def test_complete_run_resolves_selected_root_through_cost_reference(self):
        costs = [{**c, 'experiment': 'x', 'operator': 'Scan', 'group_expression': c['sequence'],
                  'optimization_request': 0, 'memo_version': 10, 'preceding_rule_candidates': 0}
                 for c in self.costs]
        events = []
        for update in self.events:
            events.extend([{**update, 'status': 'retained_new', 'previous_candidate_sequence': 0}, update])
        events.append(dict(status='selected_plan', candidate_sequence=2, operator='Scan', cost=60.,
                           plan_node=1, parent_plan_node=0))
        events = [{**e, 'experiment': 'x', 'sequence': i} for i, e in enumerate(events, 1)]
        final = dict(experiment='x', optimizer_cost=60., stats_targets=[], candidate_trace_version=2, rule_candidates=0,
                     binding_edge_trace_version=3, binding_origin_edges=0, cost_trace_version=1,
                     cost_candidates=2, cost_lifecycle_version=1, cost_lifecycle_events=len(events))
        run = dict(plan_rc=0, rows_rc=0, optimizer='pg_orca', candidate_events=[], rule_edges=[],
                   experiment_outcomes=[final], cost_events=costs, cost_lifecycle_events=events)
        report = audit_utility_run(run)
        self.assertTrue(all(a['complete'] for a in report['audits'].values()))
        self.assertEqual(report['root_cost_updates'][1]['cost_reduction'], 40.)
        self.assertEqual(report['terminal_plan_quality']['optimizer_cost'], 60.)
        for cost in (None, float('nan'), 61., True):
            bad = {**run, 'experiment_outcomes': [{**final, 'optimizer_cost': cost}]}
            self.assertFalse(audit_utility_run(bad)['terminal_plan_quality']['available'])
        # A single first-feasible plan can be the useful result of a policy;
        # it has terminal quality even without any best-cost decrease.
        single_events = [dict(status='retained_new', candidate_sequence=1, previous_candidate_sequence=0),
                         dict(status='best_updated', candidate_sequence=1, previous_candidate_sequence=0,
                              group=1, optimization_context=2),
                         dict(status='selected_plan', candidate_sequence=1, operator='Scan', cost=100.,
                              plan_node=1, parent_plan_node=0)]
        single_events = [{**e, 'experiment': 'x', 'sequence': i} for i, e in enumerate(single_events, 1)]
        single = {**run, 'cost_events': costs[:1], 'cost_lifecycle_events': single_events,
                  'experiment_outcomes': [{**final, 'optimizer_cost': 100., 'cost_candidates': 1,
                                           'cost_lifecycle_events': 3}]}
        single_report = audit_utility_run(single)
        self.assertEqual(single_report['comparable_updates'], 0)
        self.assertEqual(single_report['terminal_plan_quality']['optimizer_cost'], 100.)
        self.assertFalse(report['dfc_training_ready'])  # Cost observations alone are not credit labels.


if __name__ == '__main__':
    unittest.main()
