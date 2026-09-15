"""Local rule context is ordered, state-dependent, and independent of future outcomes."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from ml_orca.encoding.group_expression_encoding import attempt_samples, group_expression_sequence, stats_timeline
from ml_orca.trace.profile_rule_candidates import candidate_state


def attempt_fixture():
    node = {'operator': 'CLogicalGet', 'arity': 0, 'stats_source': 'memo_group',
            'rows': 10, 'empty': False, 'memo_group_expressions': 2,
            'memo_state': {'group': 4, 'group_expression': 5, 'group_explored': False,
                          'group_implemented': False, 'expression_explored': False,
                          'expression_implemented': False}}
    return {'evaluated': True, 'sequence': 1, 'rule_hash': 'a'*16, 'status': 'match_rejected',
            'input_context': {'capture': 'before_evaluation', 'scope': 'source_before_match_view',
                'root': {**node, 'operator': 'CLogicalInnerJoin', 'arity': 3},
                'children': [{'position': i, 'node': {**node, 'rows': rows}} for i, rows in enumerate((10, 100))],
                'relational_children': 2, 'omitted_children': 0,
                'source_tree': {'complete': True, 'nodes': [
                    {'operator': 'CLogicalInnerJoin', 'arity': 3},
                    {'operator': 'CLogicalSelect', 'arity': 2},
                    {'operator': 'CLogicalGet', 'arity': 0},
                    {'operator': 'CScalarConst', 'arity': 0},
                    {'operator': 'CLogicalGet', 'arity': 0},
                    {'operator': 'CScalarConst', 'arity': 0}]}}}


class GroupExpressionEncodingTest(unittest.TestCase):
    def test_global_query_input_retains_external_request_without_future_statistics(self):
        from ml_orca.encoding.group_expression_encoding import query_input_tree
        row = attempt_fixture()
        context = row['input_context']
        context.update(capture='before_memo_initialization', scope='query_after_preprocessing', stats_lifecycle_sequence=0)
        for node in [context['root'], *[c['node'] for c in context['children']]]:
            node.update(memo_state=None, memo_group_expressions=None, stats_source='expression')
        context['children'][0]['node'].update(operator='CLogicalSelect', arity=2)
        source = context['source_tree']
        source['request_binding'] = 'resolved_operator_only'
        for node in source['nodes']:
            node['request_index'] = None
        source['nodes'][2]['request_index'] = 0
        record = {'schema_version': 1, 'experiment': 'q', 'preceding_rule_candidates': 0,
                  'preceding_cost_candidates': 0, 'preceding_search_checks': 0, 'input_context': context}
        run = {'query_input_contexts': [record], 'experiment_outcomes': [{'experiment': 'q', 'query_input_context_version': 1}]}
        request = {'schema_version': 1, 'scope': 'native_stats_requests_not_runtime_resolution',
                   'experiment': 'q', 'discover': False, 'requests': [
                       {'relations': [], 'expression': 'a'*16, 'operator': 'CLogicalGet', 'requested_rows': 6}]}
        tree = query_input_tree(run, stats_requests=request)
        self.assertEqual(tree['capture'], 'before_memo_initialization')
        self.assertEqual([n['path'] for n, s in zip(tree['nodes'], tree['sequences'])
                          if ('ge:request_direct_target', 1) in s], ['r/0/0'])
        changed = deepcopy(request)
        changed['requests'][0]['requested_rows'] = 1536
        self.assertNotEqual(query_input_tree(run, stats_requests=changed), tree)
        run.update(candidate_events=[{'future': 999}], cost_events=[{'cost': 1}],
                   stats_lifecycle_events=[{'rows': 99999}], final_plan={'operator': 'different'})
        self.assertEqual(query_input_tree(run, stats_requests=request), tree)
        timed_out = deepcopy(run)
        timed_out.update(experiment_outcomes=[], plan_rc=124)
        self.assertEqual(query_input_tree(timed_out, stats_requests=request), tree)
        for field in ('preceding_rule_candidates', 'preceding_cost_candidates', 'preceding_search_checks'):
            invalid = deepcopy(run)
            invalid['query_input_contexts'][0][field] = 1
            with self.assertRaisesRegex(ValueError, 'does not precede'):
                query_input_tree(invalid, stats_requests=request)
        for mutate in (lambda r: r.update(query_input_contexts=[]),
                       lambda r: r['query_input_contexts'].append(record),
                       lambda r: r['query_input_contexts'][0]['input_context'].update(capture='after_search'),
                       lambda r: r['query_input_contexts'][0]['input_context']['root'].update(memo_state={'group': 1}),
                       lambda r: r['query_input_contexts'][0]['input_context']['source_tree']['nodes'][2].update(request_index=None)):
            invalid = deepcopy(run)
            mutate(invalid)
            with self.assertRaises(ValueError):
                query_input_tree(invalid, stats_requests=request)

    def timeline_fixture(self):
        rows = []
        for sequence, cutoff in enumerate((0, 2, 3), 1):
            row = deepcopy(attempt_fixture())
            row['sequence'] = sequence
            row['input_context']['stats_lifecycle_sequence'] = cutoff
            nodes = [row['input_context']['root'], *[c['node'] for c in row['input_context']['children']]]
            for node in nodes:
                node['memo_state']['statistics_owner_group'] = 4
                node.update(stats_source='missing', rows=None, empty=None)
            if sequence == 3:
                nodes[0].update(stats_source='memo_group', rows=42, empty=False)
            rows.append(row)
        events = [{'sequence': i, 'group': 4, 'experiment': 'test',
                   'scope': 'group_cache_after_write', 'status': status, 'rows': value, 'empty': empty,
                   'preceding_rule_candidates': before, 'preceding_cost_candidates': 0}
                  for i, (status, value, empty, before) in enumerate((
                      ('available', 10, False, 1), ('reset', None, None, 1), ('available', 42, False, 2)), 1)]
        run = {'stats_lifecycle_events': events, 'experiment_outcomes': [{
            'experiment': 'test', 'stats_lifecycle_version': 1, 'stats_lifecycle_events': 3,
            'rule_candidates': 3, 'cost_candidates': 1}], 'cost_events': [{'sequence': 1, 'stats_lifecycle_sequence': 3}]}
        return run, {'rows': rows, 'complete': True, 'exclusions': []}

    def test_statistics_timeline_distinguishes_prior_future_and_reset(self):
        run, audit = self.timeline_fixture()
        with patch('ml_orca.encoding.group_expression_encoding.candidate_evidence', return_value=audit):
            result = stats_timeline(run)
        self.assertTrue(result['complete'], result['exclusions'])
        roots = [link for link in result['links'] if link['slot'] == 'root']
        self.assertEqual([(r['prior_event_sequence'], r['next_event_sequence']) for r in roots],
                         [(None, 1), (2, 3), (3, None)])
        self.assertEqual([r['observed_rows'] for r in roots], [None, None, 42])
        self.assertEqual(len(result['links']), 9)

    def test_later_statistics_cannot_modify_previous_attempt_inputs(self):
        run, audit = self.timeline_fixture()
        with patch('ml_orca.encoding.group_expression_encoding.candidate_evidence', return_value=audit):
            before = attempt_samples(run)
            run['stats_lifecycle_events'][0]['rows'] = 9999
            self.assertTrue(stats_timeline(run)['complete'])
            self.assertEqual(before, attempt_samples(run))
            audit['rows'][0]['input_context']['stats_lifecycle_sequence'] = 1
            self.assertEqual(before[0]['inputs'], attempt_samples(run)[0]['inputs'])
            self.assertFalse(stats_timeline(run)['complete'])  # A future watermark cannot masquerade as prior.

    def test_stats_timeline_rejects_missing_invalid_and_truncated_evidence(self):
        run, audit = self.timeline_fixture()
        mutations = [lambda r: r['stats_lifecycle_events'].pop(),
                     lambda r: r['stats_lifecycle_events'][0].update(sequence=2),
                     lambda r: r['stats_lifecycle_events'][0].update(rows=float('nan')),
                     lambda r: r['stats_lifecycle_events'][2].update(rows=43),
                     lambda r: r['stats_lifecycle_events'][1].update(rows=10),
                     lambda r: r['stats_lifecycle_events'][1].update(preceding_rule_candidates=0),
                     lambda r: r['cost_events'][0].update(stats_lifecycle_sequence=4),
                     lambda r: r['experiment_outcomes'][0].pop('stats_lifecycle_version')]
        for mutate in mutations:
            bad = deepcopy(run)
            mutate(bad)
            with patch('ml_orca.encoding.group_expression_encoding.candidate_evidence', return_value=audit):
                self.assertFalse(stats_timeline(bad)['complete'])
        audit['rows'][0]['input_context'].pop('stats_lifecycle_sequence')
        with patch('ml_orca.encoding.group_expression_encoding.candidate_evidence', return_value=audit):
            self.assertIn('missing_or_invalid_pre_evaluation_stats_watermark', stats_timeline(run)['exclusions'])

    def test_local_rows_properties_and_search_state_affect_encoding(self):
        row = attempt_fixture()
        encode = lambda r: group_expression_sequence(candidate_state(r))
        initial = encode(row)
        for field, value in (('rows', 0), ('rows', 1000), ('stats_source', 'missing'),
                             ('memo_group_expressions', 3),
                             ('memo_state', {'expression_explored': True}),
                             ('logical_properties', {'source': 'complete_memo_group', 'output_columns': 3,
                                 'outer_columns': 1, 'not_null_columns': 1, 'key_count': 0, 'join_depth': 2})):
            changed = deepcopy(row)
            changed['input_context']['children'][0]['node'][field] = value
            self.assertNotEqual(initial, encode(changed))
        swapped = deepcopy(row)
        children = swapped['input_context']['children']
        children[0]['node'], children[1]['node'] = children[1]['node'], children[0]['node']
        self.assertNotEqual(initial, encode(swapped))

    def test_identifiers_and_future_labels_do_not_affect_inputs(self):
        row = attempt_fixture()
        before = candidate_state(row)
        row.update(group=99, memo_version=999, rule_hash='b'*16, status='ready_cbo',
                   match_us=999, direct_insertions=999, binding_context={'future': True})
        row['input_context']['root']['memo_state'].update(group=123, group_expression=456)
        after = candidate_state(row)
        self.assertNotEqual(before['provenance'], after['provenance'])
        self.assertEqual(group_expression_sequence(before), group_expression_sequence(after))

    def test_ordered_tree_and_prefix_validation(self):
        row = attempt_fixture()
        before = group_expression_sequence(candidate_state(row))
        tree = row['input_context']['source_tree']
        nodes = tree['nodes']
        tree['nodes'] = [nodes[0], nodes[4], *nodes[1:4], nodes[5]]
        self.assertNotEqual(before, group_expression_sequence(candidate_state(row)))
        tree['nodes'] = nodes[:2]
        with self.assertRaises(ValueError):
            group_expression_sequence(candidate_state(row))
        tree['complete'] = False
        self.assertIn(('ge:tree_complete', 0), group_expression_sequence(candidate_state(row)))
        tree['nodes'] = nodes + [nodes[-1]]
        with self.assertRaises(ValueError):
            group_expression_sequence(candidate_state(row))
        del row['input_context']['source_tree']
        self.assertIn(('ge:tree_available', 0), group_expression_sequence(candidate_state(row)))
        row['input_context']['children'][1]['position'] = 0
        with self.assertRaises(ValueError):
            group_expression_sequence(candidate_state(row))

    def test_uncaptured_and_incomplete_attempts_remain_in_denominator(self):
        rows = [attempt_fixture(), {**attempt_fixture(), 'evaluated': False, 'status': 'budget_skipped'}]
        audit = {'rows': rows, 'complete': False, 'exclusions': ['candidate_sequence_gap']}
        with patch('ml_orca.encoding.group_expression_encoding.candidate_evidence', return_value=audit):
            samples = attempt_samples({})
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(not r['admission']['trace_complete'] for r in samples))
        self.assertEqual(samples[1]['inputs']['group_expression_sequence'], [('ge:captured', 0)])
        self.assertIn('pre_evaluation_context_unavailable', samples[1]['admission']['feature_exclusions'])
        row = attempt_fixture()
        row['input_context']['capture'] = 'after_evaluation'
        self.assertEqual(group_expression_sequence(candidate_state(row)), [('ge:captured', 0)])


if __name__ == '__main__':
    unittest.main()
