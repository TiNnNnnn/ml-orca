"""History time/population gates, concrete root routing, and trainable context fusion."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from ml_orca.encoding.group_expression_encoding import group_expression_tree
from ml_orca.trace.profile_rule_candidates import candidate_state
from ml_orca.encoding.rule_history_encoding import attach_history
from ml_orca.encoding.rule_policy_encoding import encode_sequence, fit_vocabulary
from ml_orca.tests.test_group_expression_encoding import attempt_fixture
from ml_orca.tests.test_rule_policy_encoding import ir_fixture
from ml_orca.tests.test_rule_tree_model import tree_features, torch


def actual_tree():
    row = attempt_fixture()
    row['input_context']['children'][0]['node'].update(operator='CLogicalSelect', arity=2)
    return group_expression_tree(candidate_state(row))


def history_fixture():
    with patch('ml_orca.tests.test_rule_tree_model.encode_sequence', side_effect=lambda s, _: s):
        feature, _ = tree_features([ir_fixture(), ir_fixture()])
    feature['rule_hashes'] = ['0', '1']  # Admission metadata, not model tokens.
    tree = actual_tree()
    history = {'schema_version': 1, 'capture': 'audited_history_frozen', 'frozen_at_utc': '2026-09-12T10:00:00+00:00',
        'runs': [{'source': {'path': 'source'}, 'unit': {'workload': 'dev', 'query': '1'},
            'inputs': {'query_sql': 'select * from a', 'catalog_snapshot': {'path': 'catalog'},
                       'graph_snapshot': {'path': 'graph'}}, 'trees': [tree, None],
            'contexts': [{'rule_hash': '1', 'tree': 0, 'attempts': 2}, {'rule_hash': '1', 'tree': 1, 'attempts': 1}],
            'edges': [{'src_rule': '0', 'dst_rule': '1', 'src_target_path': 'r/0/0/5',
                       'dst_binding_path': 'r/0', 'producer_relation': 'input_exposes',
                       'producer_outcome': 'memo_duplicate', 'tree': 0,
                       'root': next(i for i, n in enumerate(tree['nodes']) if n['path'] == 'r/0')}],
            'denominators': {'attempts': 3, 'observed_edges': 1, 'admitted_edges': 1, 'exclusions': {'context_unavailable': 1}}}]}
    return feature, history


def attach(feature, bundle, timestamp='2026-09-12T11:00:00+00:00', allowed=None):
    documents = {'source': {}, 'catalog': {'catalog': {'captured_at': '2026-09-12T09:00:00+00:00'}},
                 'graph': {'nodes': [{'rule_hash': '0'}, {'rule_hash': '1'}]}}
    with patch('ml_orca.encoding.rule_history_encoding.read_snapshot', side_effect=lambda s: json.dumps(documents[s['path']]).encode()), \
         patch('ml_orca.encoding.rule_history_encoding.input_sequences', return_value=deepcopy(feature)):
        return attach_history(feature, bundle, {}, timestamp, {'dev:1': 'select * from a'} if allowed is None else allowed)


class RuleHistoryEncodingTest(unittest.TestCase):
    def test_training_minimum_counts_complete_dynamic_queries_and_rejects_leakage(self):
        from ml_orca.training.train_policy_baseline import history_training_gate
        run = {'unit': {'workload': 'app', 'query': '1.sql'},
               'inputs': {'query_sql': 'SELECT 1', 'candidate_policy': [{'rule_hash': 'a'}]},
               'trees': [{'nodes': [{'path': 'r'}], 'root': 0}],
               'contexts': [{'attempts': 3, 'rule_hash': 'a', 'tree': 0}],
               'edges': [{'src_rule': 'a', 'dst_rule': 'a', 'tree': 0,
                          'root': 0, 'dst_binding_path': 'r'}],
               'denominators': {'attempts': 3, 'observed_edges': 1,
                                'admitted_edges': 1, 'exclusions': {}}}
        history, allowed = {'runs': [run]}, {'app:1': 'SELECT 1'}
        self.assertEqual(history_training_gate(history, allowed, 1)['qualified_dynamic_queries'], 1)
        with self.assertRaisesRegex(ValueError, 'declared minimum 2000'):
            history_training_gate(history, allowed, 2000)
        with self.assertRaisesRegex(ValueError, 'requires audited history'):
            history_training_gate(None, allowed, 2000)
        with self.assertRaisesRegex(ValueError, 'outside assigned'):
            history_training_gate(history, {}, 1)
        repeated = deepcopy(run)
        repeated['unit']['query'] = '2.sql'
        with self.assertRaisesRegex(ValueError, 'queries 1 <'):
            history_training_gate({'runs': [run, repeated]}, {**allowed, 'app:2': 'SELECT 1'}, 2)
        # Thousands of edge events from one query still count as one query.
        run['edges'] *= 2000
        run['denominators'].update(observed_edges=2000, admitted_edges=2000)
        with self.assertRaisesRegex(ValueError, 'queries 1 <'):
            history_training_gate(history, allowed, 2000)
        run['edges'][0]['dst_binding_path'] = 'r/99'
        with self.assertRaisesRegex(ValueError, 'queries 0 <'):
            history_training_gate(history, allowed, 1)
        report = history_training_gate(history, allowed, 0)
        self.assertFalse(report['required'])  # Legacy pilot is explicitly not a population gate.
        self.assertIn('unresolved_actual_binding_root', report['excluded_graph_reasons'])

    def test_shared_history_equals_independent_admission_and_keeps_time_gate(self):
        from ml_orca.training.train_policy_baseline import reuse_history
        original, bundle = history_fixture()
        prepared = attach(deepcopy(original), bundle)
        later = '2026-09-12T12:00:00+00:00'
        shared = deepcopy(original)
        reuse_history(shared, prepared, later)
        self.assertEqual(shared, attach(deepcopy(original), bundle, later))
        self.assertIs(shared['history_trees'], prepared['history_trees'])
        self.assertIs(shared['history_edges'], prepared['history_edges'])
        self.assertIs(shared['sequences']['history_node'], prepared['sequences']['history_node'])
        self.assertEqual(prepared['history_admission']['prediction_at'], '2026-09-12T11:00:00+00:00')
        with self.assertRaisesRegex(ValueError, 'already attached'):
            reuse_history(shared, prepared, later)
        for timestamp in (bundle['frozen_at_utc'], '2026-09-12T09:00:00+00:00',
                          '2026-09-12T12:00:00'):
            with self.assertRaises(ValueError):
                reuse_history(deepcopy(original), prepared, timestamp)
        wrong = deepcopy(original)
        wrong['rule_trees'].pop()
        with self.assertRaisesRegex(ValueError, 'rule IR mismatch'):
            reuse_history(wrong, prepared, later)

    def test_shared_history_rejects_same_shape_with_different_rule_semantics(self):
        from ml_orca.training.train_policy_baseline import reuse_history
        original, bundle = history_fixture()
        prepared = attach(deepcopy(original), bundle)
        for channel in ('rule_node', 'rule_symbol', 'rule_constraint'):
            changed = deepcopy(original)
            changed['sequences'][channel][0][0] = ('different_semantics', None)
            self.assertEqual(changed['rule_trees'], prepared['rule_trees'])
            with self.assertRaisesRegex(ValueError, 'rule IR mismatch'):
                reuse_history(changed, prepared, '2026-09-12T12:00:00+00:00')
        wrong = deepcopy(original)
        wrong['rule_graph_scope'] = 'runtime'
        with self.assertRaisesRegex(ValueError, 'rule IR mismatch'):
            reuse_history(wrong, prepared, '2026-09-12T12:00:00+00:00')
        wrong = deepcopy(original)
        wrong['rule_hashes'].reverse()
        self.assertEqual(wrong['rule_trees'], prepared['rule_trees'])
        with self.assertRaisesRegex(ValueError, 'rule IR mismatch'):
            reuse_history(wrong, prepared, '2026-09-12T12:00:00+00:00')

    def test_ordered_actual_tree_preserves_statistics_and_missing_deep_nodes(self):
        tree = actual_tree()
        self.assertEqual(tree['nodes'][tree['root']]['path'], 'r')
        indices = {n['path']: i for i, n in enumerate(tree['nodes'])}
        self.assertEqual(tree['nodes'][indices['r']]['children'], [indices['r/0'], indices['r/1'], indices['r/2']])
        self.assertIn(('ge:rows', None), tree['sequences'][indices['r/0/0']])
        self.assertNotIn(('ge:rows', None), tree['sequences'][indices['r/0']])
        row = attempt_fixture()
        with self.assertRaisesRegex(ValueError, 'disagree'):
            group_expression_tree(candidate_state(row))
        row['input_context']['children'][0]['node'].update(operator='CLogicalSelect', arity=2)
        row['input_context']['source_tree']['nodes'] = row['input_context']['source_tree']['nodes'][:3]
        row['input_context']['source_tree']['complete'] = False
        partial = group_expression_tree(candidate_state(row))
        self.assertFalse(partial['complete'])
        self.assertEqual(len(partial['nodes']), 3)
        self.assertNotIn('r/1', [n['path'] for n in partial['nodes']])

    def test_admission_refuses_future_and_unassigned_queries(self):
        feature, bundle = history_fixture()
        for timestamp in ('2026-09-12T09:59:59+00:00', bundle['frozen_at_utc'], '2026-09-12T11:00:00'):
            with self.assertRaises(ValueError):
                attach(deepcopy(feature), bundle, timestamp)
        with self.assertRaisesRegex(ValueError, 'training population'):
            attach(deepcopy(feature), bundle, allowed={})
        bundle['runs'][0]['unit']['query'] = '1.sql'
        ready = attach(feature, bundle)
        self.assertEqual(ready['history_contexts'][1]['count'], 1)
        self.assertEqual(ready['history_edges'][0]['root'], bundle['runs'][0]['edges'][0]['root'])
        # Instantiated producer paths may extend beyond a template's Input.
        self.assertEqual(ready['edge_roots'], [[5, 2]])  # Static ports remain separate.

    def test_wrong_binding_path_cannot_be_replaced_by_a_template_root(self):
        feature, bundle = history_fixture()
        bundle['runs'][0]['edges'][0]['dst_binding_path'] = 'r/0/99'
        with self.assertRaisesRegex(ValueError, 'binding root'):
            attach(feature, bundle)

    def test_duplicate_runs_bad_indices_and_lost_denominators_rejected(self):
        feature, history = history_fixture()
        for mutate in (lambda b: b['runs'].append(deepcopy(b['runs'][0])),
                       lambda b: b['runs'][0]['edges'][0].update(tree=-1),
                       lambda b: b['runs'][0]['contexts'][0].update(attempts=-1),
                       lambda b: b['runs'][0]['denominators'].update(attempts=99)):
            bad = deepcopy(history)
            mutate(bad)
            with self.assertRaises(ValueError):
                attach(deepcopy(feature), bad)


@unittest.skipIf(torch is None, 'PyTorch is optional')
class RuleHistoryModelTest(unittest.TestCase):
    def test_edge_gather_preserves_full_model_gradients_and_adam_updates(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        feature, bundle = history_fixture()
        features = attach(feature, bundle)
        features['history_runs'] *= 2
        edge = features['history_edges'][0]
        features['history_edges'] = [dict(edge, run=i % 2,
            root=i % len(features['history_trees'][0]['nodes'])) for i in range(31)]
        features['sequences']['history_edge'] *= 31
        vocabulary = fit_vocabulary([features['sequences']])
        features['sequences'] = {k: [encode_sequence(s, vocabulary) for s in seqs]
                                for k, seqs in features['sequences'].items()}
        original_select = torch.Tensor.index_select

        def scalar_select(value, dim, index):
            if not len(index):
                return original_select(value, dim, index)
            return torch.stack([value.select(dim, int(i)) for i in index], dim=dim)

        for workers, dtype in ((1, torch.float64), (4, torch.float64), (1, torch.float32), (4, torch.float32)):
            torch.set_num_threads(workers)
            torch.manual_seed(977)
            model = TreePolicyPredictor(len(vocabulary), 8, history=True).to(dtype=dtype)
            reference = deepcopy(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=.003)
            for _ in range(2):
                optimizer.zero_grad()
                reference_optimizer.zero_grad()
                prediction = model(features)
                loss = prediction.square().mean()
                loss.backward()
                with patch.object(torch.Tensor, 'index_select', scalar_select):
                    expected = reference(features)
                    expected_loss = expected.square().mean()
                    expected_loss.backward()
                tolerance = dict(rtol=1e-9, atol=1e-11) if dtype == torch.float64 else dict(rtol=2e-5, atol=2e-7)
                torch.testing.assert_close(prediction, expected, **tolerance)
                torch.testing.assert_close(loss, expected_loss, **tolerance)
                for (name, actual), (_, wanted) in zip(model.named_parameters(), reference.named_parameters()):
                    self.assertEqual(actual.grad is None, wanted.grad is None, name)
                    if actual.grad is not None:
                        torch.testing.assert_close(actual.grad, wanted.grad, **tolerance)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                reference_optimizer.step()
                for actual, wanted in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, wanted, **tolerance)
            for field, bad in (('tree', -1), ('root', -1), ('run', -1)):
                corrupt = deepcopy(features)
                corrupt['history_edges'][0][field] = bad
                with self.assertRaises(ValueError):
                    model(corrupt)
            features_empty = deepcopy(features)
            features_empty['history_edges'] = []
            features_empty['sequences']['history_edge'] = []
            self.assertTrue(torch.isfinite(model(features_empty)).all())

    def test_shared_history_preserves_predictions_and_parameter_gradients(self):
        from ml_orca.training.train_policy_baseline import reuse_history
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        original, bundle = history_fixture()
        prepared = attach(deepcopy(original), bundle)
        shared = deepcopy(original)
        reuse_history(shared, prepared, '2026-09-12T11:00:00+00:00')
        vocabulary = fit_vocabulary([prepared['sequences']])
        encoded_history = {k: [encode_sequence(s, vocabulary) for s in v]
                           for k, v in prepared['sequences'].items() if k.startswith('history_')}
        for feature in (prepared, shared):
            feature['sequences'] = {k: encoded_history[k] if k in encoded_history else
                [encode_sequence(s, vocabulary) for s in v] for k, v in feature['sequences'].items()}
        separate = deepcopy(shared)
        for mode in ('none', 'context', 'graph'):
            model = TreePolicyPredictor(len(vocabulary), 8, history=True, history_mode=mode)
            expected = model(separate)
            expected.sum().backward()
            gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            actual = model(shared)
            actual.sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key, parameter in model.named_parameters():
                if key in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[key], rtol=0, atol=0)
                else:
                    self.assertIsNone(parameter.grad)
            self.assertEqual(shared, separate)

    def test_context_moments_distinguish_spread_and_preserve_counts_and_gradients(self):
        from ml_orca.models.rule_tree_model import context_summary
        # Equal hidden means/counts, different empirical distributions.
        spread = torch.tensor([[-.75, -.25], [.75, .25]], dtype=torch.float64, requires_grad=True)
        flat = torch.zeros_like(spread)
        destinations = torch.tensor([0, 0])
        weights = torch.ones((2, 1), dtype=torch.float64)
        torch.testing.assert_close(context_summary(spread, destinations, weights, 2),
                                   context_summary(flat, destinations, weights, 2))
        first = context_summary(spread, destinations, weights, 2, variance=True)
        second = context_summary(flat, destinations, weights, 2, variance=True)
        self.assertFalse(torch.equal(first, second))
        torch.testing.assert_close(first[0, 2:4], torch.tensor([.5625, .0625], dtype=torch.float64))
        self.assertTrue(torch.equal(first[1], torch.zeros_like(first[1])))  # Unobserved rule.
        # Compression is exact for identical encoded vectors; order is irrelevant.
        weights = torch.tensor([[3.], [2.]], dtype=torch.float64)
        compressed = context_summary(spread, destinations, weights, 2, variance=True)
        expanded = context_summary(spread.repeat_interleave(torch.tensor([3, 2]), dim=0),
                                   torch.zeros(5, dtype=torch.long), torch.ones((5, 1), dtype=torch.float64),
                                   2, variance=True)
        torch.testing.assert_close(compressed, expanded)
        torch.testing.assert_close(torch.autograd.grad(compressed.square().sum(), spread)[0],
                                   torch.autograd.grad(expanded.square().sum(), spread)[0])
        torch.testing.assert_close(compressed, context_summary(spread.flip(0), destinations,
                                                               weights.flip(0), 2, variance=True))
        empty = context_summary(spread[:0], destinations[:0], weights[:0], 2, variance=True)
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))
        # Do not imply that two moments uniquely identify a distribution.
        another = torch.tensor([[-1.], [0.], [0.], [1.]], dtype=torch.float64)
        same_moments = torch.tensor([[-2**-.5], [-2**-.5], [2**-.5], [2**-.5]], dtype=torch.float64)
        args = (torch.zeros(4, dtype=torch.long), torch.ones((4, 1), dtype=torch.float64), 1)
        torch.testing.assert_close(context_summary(another, *args, variance=True),
                                   context_summary(same_moments, *args, variance=True))

    def test_variance_pooling_integrates_without_expanding_rule_nodes_or_edges(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        feature, bundle = history_fixture()
        features = attach(feature, bundle)
        vocabulary = fit_vocabulary([features['sequences']])
        features['sequences'] = {k: [encode_sequence(s, vocabulary) for s in seqs]
                                for k, seqs in features['sequences'].items()}
        original = deepcopy(features)
        model = TreePolicyPredictor(len(vocabulary), 8, history=True, history_pooling='mean_variance')
        before = model(features)
        before.sum().backward()
        self.assertEqual(features, original)
        self.assertEqual(model.history_update.in_features, 17)
        self.assertTrue(torch.isfinite(model.history_update.weight.grad).all())
        self.assertGreater(model.history_update.weight.grad[:, 8:16].abs().sum().item(), 0)
        clone = TreePolicyPredictor(len(vocabulary), 8, history=True, history_pooling='mean_variance')
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(before, clone(features))
        for count in (0, -1, True, float('nan')):
            bad = deepcopy(features)
            bad['history_contexts'][0]['count'] = count
            with self.assertRaisesRegex(ValueError, 'attempt count'):
                model(bad)

    def test_batched_history_projections_match_scalar_values_and_gradients(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        feature, bundle = history_fixture()
        features = attach(feature, bundle)
        # Repeated edges remain repeated observations, not sampled unique pairs.
        features['history_edges'] *= 3
        features['sequences']['history_edge'] *= 3
        vocabulary = fit_vocabulary([features['sequences']])
        features['sequences'] = {k: [encode_sequence(s, vocabulary) for s in seqs] for k, seqs in features['sequences'].items()}
        torch.manual_seed(973)
        model = TreePolicyPredictor(len(vocabulary), 8, history=True)
        prediction = model(features)
        prediction.sum().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()
        attempt, ports = model.history_attempt.forward, model.history_ports.forward
        with patch.object(model.history_attempt, 'forward', side_effect=lambda values: torch.stack([attempt(v) for v in values])), \
             patch.object(model.history_ports, 'forward', side_effect=lambda values: torch.stack([ports(v) for v in values])):
            reference = model(features)
            reference.sum().backward()
        torch.testing.assert_close(prediction, reference)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-5, atol=2e-7)
        # Controls share weights/vocabulary. Each masks only its declared channel.
        for mode in ('none', 'context'):
            model.history_mode = mode
            before = model(features)
            poisoned = deepcopy(features)
            poisoned['history_edges'] = []
            poisoned['sequences']['history_edge'] = []
            if mode == 'none':
                poisoned['history_contexts'] = []
                poisoned['history_trees'] = []
            torch.testing.assert_close(before, model(poisoned), rtol=0, atol=0)

    def test_history_tree_policy_and_actual_roots_train_through_all_rounds(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(963)
        feature, bundle = history_fixture()
        features = attach(feature, bundle)
        vocabulary = fit_vocabulary([features['sequences']])
        features['sequences'] = {k: [encode_sequence(s, vocabulary) for s in seqs] for k, seqs in features['sequences'].items()}
        model = TreePolicyPredictor(len(vocabulary), 8, history=True)
        before = model(features)
        for kind in ('root', 'tree', 'behavior_policy', 'query_statistics', 'attempt_count'):
            changed = deepcopy(features)
            if kind == 'root':
                changed['history_edges'][0]['root'] = changed['history_trees'][0]['root']
            elif kind in ('tree', 'behavior_policy', 'query_statistics'):
                channel = {'tree': 'history_node', 'behavior_policy': 'history_policy', 'query_statistics': 'history_relation'}[kind]
                for seq in changed['sequences'][channel]:
                    seq['numbers'] = [v + 10 if known else v for v, known in zip(seq['numbers'], seq['known_number'])]
            else:
                changed['history_contexts'][0]['count'] += 10
            self.assertFalse(torch.equal(before, model(changed)), kind)
        changed = deepcopy(features)
        changed['response'] = {'execution_ms': 1e20}
        changed['current_memo'] = {'rows': 1e20}
        torch.testing.assert_close(before, model(changed), rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.parameters())
        before.square().sum().backward()
        for name, parameter in model.named_parameters():
            if name.startswith(('history_', 'rounds.')):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model(features)))
        with self.assertRaisesRegex(ValueError, 'explicitly enable'):
            TreePolicyPredictor(len(vocabulary), 8)(features)


if __name__ == '__main__':
    unittest.main()
