"""Tree topology, bound constraints, exact dependency roots and multi-hop reachability."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from ml_orca.encoding.rule_policy_encoding import input_sequences, rule_structure, rule_root_index, encode_sequence, fit_vocabulary
from ml_orca.tests.test_rule_policy_encoding import ir_fixture, catalog_fixture


def tree_features(irs, paths=('r',), vocabulary=None):
    native = {'schema_version': 1, 'nodes': [{'rule_hash': str(i), 'learning_ir': ir} for i, ir in enumerate(irs)],
              'edges': [{'src_rule': '0', 'dst_rule': str(len(irs) - 1), 'evidence': 'static_template',
                         'target_path': p, 'src_target_path': p, 'dst_source_path': 'r'} for p in paths]}
    catalog = catalog_fixture()
    catalog['settings'] = {'search_path': 'public'}
    for relation in catalog['relations']:
        relation['schema'] = 'public'
    documents = {'graph': native, 'static': native, 'catalog': {'catalog': catalog}}
    inputs = {'graph_snapshot': {'path': 'graph'}, 'catalog_snapshot': {'path': 'catalog'},
              'query_sql': 'SELECT a.id FROM a', 'candidate_policy': [
                  {'rule_hash': str(i), 'enabled': True, 'placement': 'cbo'} for i in range(len(irs))]}
    with patch('ml_orca.encoding.rule_policy_encoding.read_snapshot', side_effect=lambda s: json.dumps(documents[s['path']]).encode()):
        feature = input_sequences(inputs, {'path': 'static'}, tree_rules=True)
    vocabulary = vocabulary or fit_vocabulary([feature['sequences']])
    feature['sequences'] = {k: [encode_sequence(s, vocabulary) for s in value] for k, value in feature['sequences'].items()}
    return feature, vocabulary


class RuleTreeLayoutTest(unittest.TestCase):
    def test_exact_source_target_ports_and_constraint_references(self):
        structure = rule_structure(ir_fixture())
        self.assertEqual(len(structure['nodes']), 6)
        self.assertEqual(structure['roots'], {'source': 2, 'target': 5})
        self.assertEqual(rule_root_index(structure, 'target', 'r/0'), 4)
        self.assertEqual(rule_root_index(structure, 'source', 'r/0/0'), 0)
        self.assertEqual(structure['constraint_references'][0], [{'token': 2, 'symbol': 9}, {'token': 3, 'symbol': 4}])
        for side, path in (('source', 'r/1'), ('source', 'r/0/0/0'), ('source', 'r/00'), ('other', 'r')):
            with self.assertRaises(ValueError):
                rule_root_index(structure, side, path)
        features, _ = tree_features([ir_fixture(), ir_fixture()], ('r', 'r/0'))
        self.assertEqual(features['edge_roots'], [[5, 2], [4, 2]])
        self.assertEqual(features['rule_edges'], [[0, 1], [0, 1]])


@unittest.skipIf(torch is None, 'PyTorch is optional')
class RuleTreeModelTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(951)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_expression_trees_use_existing_encoder_and_keep_relational_ports(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        ir = ir_fixture()
        ir.update(schema_version=2, bindings=[{'kind': 'Not', 'mode': 'build', 'symbols': [5, 0]}])
        other = deepcopy(ir)
        other['bindings'][0].update(kind='And', symbols=[5, 0, 2])
        features, vocabulary = tree_features([ir, other], ('r', 'r/0'))
        model = TreePolicyPredictor(len(vocabulary), 8)
        result = model(features)
        self.assertTrue(torch.isfinite(result).all())
        result.square().sum().backward()
        self.assertGreater(model.rule_trees.child_order.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertNotEqual(rule_structure(ir), rule_structure(other))
        structure = rule_structure(ir)
        with self.assertRaises(ValueError):
            rule_root_index(structure, 'target', 'r/1')

    def test_pre_memo_query_tree_is_explicit_trainable_and_preserves_legacy_defaults(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        features, vocabulary = tree_features([ir_fixture(), ir_fixture()], ('r', 'r/0'))
        for token in ('ge:operator:CLogicalGet', 'ge:operator:CLogicalSelect', 'ge:requested_rows'):
            vocabulary.setdefault(token, len(vocabulary))
        baseline = deepcopy(features)
        features.update(decision_point='post_preprocessing_pre_cbo', query_input_tree={
            'capture': 'before_memo_initialization', 'scope': 'pre_memo_query_root_expression', 'complete': True,
            'nodes': [{'path': 'r/0/0', 'children': []}, {'path': 'r/0', 'children': [0]},
                      {'path': 'r', 'children': [1]}], 'root': 2})
        query_sequences = [
            [('ge:operator:CLogicalGet', None), ('ge:requested_rows', 2.)],
            [('ge:operator:CLogicalSelect', None)], [('ge:operator:CLogicalSelect', None)]]
        features['sequences']['query_input_node'] = [encode_sequence(s, vocabulary) for s in query_sequences]
        torch.manual_seed(71)
        old = TreePolicyPredictor(len(vocabulary), 8)
        torch.manual_seed(71)
        model = TreePolicyPredictor(len(vocabulary), 8, query_input=True)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        before = old(baseline)
        with self.assertRaisesRegex(ValueError, 'explicitly enable'):
            old(features)
        with self.assertRaisesRegex(ValueError, 'complete pre-Memo'):
            model(baseline)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(old.state_dict())
        observed = model(features)
        changed = deepcopy(features)
        changed['sequences']['query_input_node'][0]['numbers'][1] = 7.
        self.assertFalse(torch.equal(model.query_inputs(features), model.query_inputs(changed)))
        self.assertFalse(torch.equal(observed, model(changed)))
        poison = deepcopy(features)
        poison.update(final_plan={'rows': 999}, response={'cost': 999}, candidate_events=[{'future': True}])
        torch.testing.assert_close(observed, model(poison), rtol=0, atol=0)
        for mutate in (lambda f: f['query_input_tree'].update(capture='after_search'),
                       lambda f: f['query_input_tree']['nodes'][0].update(children=[2]),
                       lambda f: f['query_input_tree'].update(complete=False)):
            bad = deepcopy(features)
            mutate(bad)
            with self.assertRaises(ValueError):
                model(bad)
        observed.sum().backward()
        self.assertTrue(torch.isfinite(model.query_input_fusion.weight.grad).all())
        self.assertGreater(model.query_input_fusion.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.rule_trees.child_order.weight_ih_l0.grad.abs().sum().item(), 0)
        clone = TreePolicyPredictor(len(vocabulary), 8, query_input=True)
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(observed, clone(features), rtol=0, atol=0)
        torch.testing.assert_close(before, old(baseline), rtol=0, atol=0)

        pooled = TreePolicyPredictor(len(vocabulary), 8, query_input=True, query_pooling='root_mean')
        states = pooled.rule_trees.tree_states(pooled.encoder(features['sequences']['query_input_node']),
                                               features['query_input_tree']['nodes'])
        from ml_orca.models.sequence import query_context_embedding
        expected = torch.tanh(pooled.query_input_fusion(torch.cat((
            query_context_embedding(pooled.encoder, features), states[2:3], states.mean(0, keepdim=True)), 1)))
        torch.testing.assert_close(pooled.query_inputs(features), expected, rtol=0, atol=0)
        pooled(features).square().sum().backward()
        self.assertGreater(pooled.query_input_fusion.weight.grad[:, 16:].abs().sum().item(), 0)
        self.assertGreater(pooled.rule_trees.child_order.weight_ih_l0.grad.abs().sum().item(), 0)
        for params in ({'query_pooling': 'root_mean'}, {'query_input': True, 'query_pooling': 'unknown'}):
            with self.assertRaises(ValueError):
                TreePolicyPredictor(len(vocabulary), 8, **params)
        with self.assertRaises(RuntimeError):
            pooled.load_state_dict(model.state_dict())

        # Query input and admitted historical GE/rooted edges must compose.
        from ml_orca.tests.test_rule_history_encoding import history_fixture, attach
        historical, bundle = history_fixture()
        historical = attach(historical, bundle)
        historical.update(decision_point=features['decision_point'], query_input_tree=features['query_input_tree'])
        historical['sequences']['query_input_node'] = query_sequences
        vocabulary = fit_vocabulary([historical['sequences']])
        historical['sequences'] = {k: [encode_sequence(s, vocabulary) for s in value]
                                   for k, value in historical['sequences'].items()}
        combined = TreePolicyPredictor(len(vocabulary), 8, history=True, query_input=True)
        output = combined(historical)
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        for parameter in (combined.history_update.weight, combined.query_input_fusion.weight,
                          combined.rounds[0].update.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_priority_readout_is_explicit_and_legacy_head_stays_two_channels(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        from ml_orca.objectives.priority import priority_pair
        from ml_orca.training.priority_loss import priority_cost_loss
        from ml_orca.tests.test_priority_control import policy_cell
        features, vocabulary = tree_features([ir_fixture(), ir_fixture()])
        torch.manual_seed(71)
        default = TreePolicyPredictor(len(vocabulary), 8)
        torch.manual_seed(71)
        explicit = TreePolicyPredictor(len(vocabulary), 8, output_size=2)
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        model = TreePolicyPredictor(len(vocabulary), 8, output_size=1)
        changed = deepcopy(features)
        # Same query and DSL/graph; only policy priority changes.
        for sequence in changed['sequences']['policy']:
            index = sequence['tokens'].index(vocabulary['priority'])
            sequence['numbers'][index] = .5
            sequence['known_number'][index] = True
        scores = torch.cat([model(features), model(changed)])
        report = priority_pair(policy_cell('a', 10), policy_cell('b', 2))
        loss = priority_cost_loss(scores, [(0, 1, report)])
        loss.backward()
        self.assertTrue(torch.isfinite(model.readout[-1].weight.grad).all())
        self.assertGreater(model.readout[-1].weight.grad.abs().sum().item(), 0)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(default.state_dict())
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TreePolicyPredictor(len(vocabulary), 8, output_size=size)

    def test_priority_loss_preserves_ties_offsets_and_rejects_unadmitted_pairs(self):
        from ml_orca.objectives.priority import priority_pair
        from ml_orca.training.priority_loss import priority_cost_loss
        from ml_orca.tests.test_priority_control import policy_cell
        report = priority_pair(policy_cell('a', 10), policy_cell('b', 2))
        scores = torch.tensor([1., 0.], requires_grad=True)
        loss = priority_cost_loss(scores, [(0, 1, report)])
        torch.testing.assert_close(loss, priority_cost_loss(scores + 5, [(0, 1, report)]))
        reverse = priority_pair(policy_cell('b', 2), policy_cell('a', 10))
        torch.testing.assert_close(loss, priority_cost_loss(scores, [(1, 0, reverse)]))
        loss.backward()
        self.assertAlmostEqual(scores.grad.sum().item(), 0)
        tie = priority_pair(policy_cell('a'), policy_cell('b'))
        self.assertEqual(priority_cost_loss(torch.zeros(2), [(0, 1, tie)]).item(), 0)
        self.assertGreater(priority_cost_loss(scores, [(0, 1, tie)]).item(), 0)
        for pairs in ([], [(0, 0, report)], [(0, 2, report)], [(0, 1, report), (1, 0, reverse)]):
            with self.assertRaises(ValueError):
                priority_cost_loss(scores, pairs)
        for mutate in (lambda r: r.update(trace_complete=False),
                       lambda r: r.update(work_delta=None),
                       lambda r: r.update(cost_exclusions=['missing']),
                       lambda r: r['cost'].update(log1p_margin=999)):
            invalid = deepcopy(report)
            mutate(invalid)
            with self.assertRaises(ValueError):
                priority_cost_loss(scores, [(0, 1, invalid)])
        with self.assertRaises(ValueError):
            priority_cost_loss(torch.tensor([float('nan'), 0.]), [(0, 1, report)])

    def test_recursive_tree_topology_order_and_arbitrary_child_count(self):
        from ml_orca.models.rule_tree_model import RuleTreeEncoder
        model = RuleTreeEncoder(8)
        values = torch.randn(5, 8)
        first = [{'children': []}, {'children': []}, {'children': [0, 1]}, {'children': [2]}]
        second = [{'children': []}, {'children': [0]}, {'children': []}, {'children': [1, 2]}]
        self.assertFalse(torch.equal(model.tree_states(values[:4], first)[-1], model.tree_states(values[:4], second)[-1]))
        many = [{'children': []} for _ in range(4)] + [{'children': [0, 1, 2, 3]}]
        original = model.tree_states(values, many)[-1]
        many[-1]['children'].reverse()
        self.assertFalse(torch.equal(original, model.tree_states(values, many)[-1]))
        with self.assertRaises(ValueError):
            model.tree_states(values, [{'children': [0]}])

    def test_bound_constraints_affect_roots_and_permutation_is_a_set(self):
        from ml_orca.models.rule_tree_model import RuleTreeEncoder
        from ml_orca.models.rule_policy_model import SharedSequenceEncoder
        ir = ir_fixture()
        features, vocabulary = tree_features([ir])
        encoder, model = SharedSequenceEncoder(len(vocabulary), 8), RuleTreeEncoder(8)
        rule, states = model(features['rule_trees'][0], features['sequences'], encoder)
        changed = deepcopy(ir)
        changed['constraints'][1]['symbols'] = [6, 3]  # Same kind/count, a different source attribute binding.
        other, _ = tree_features([changed], vocabulary=vocabulary)
        rebound, rebound_states = model(other['rule_trees'][0], other['sequences'], encoder)
        self.assertFalse(torch.equal(rule, rebound))
        self.assertFalse(torch.equal(states, rebound_states))
        reordered = deepcopy(ir)
        reordered['constraints'].reverse()
        other, _ = tree_features([reordered], vocabulary=vocabulary)
        same, same_states = model(other['rule_trees'][0], other['sequences'], encoder)
        torch.testing.assert_close(rule, same)
        torch.testing.assert_close(states, same_states)
        empty = deepcopy(ir)
        empty['constraints'] = []
        other, _ = tree_features([empty], vocabulary=vocabulary)
        self.assertFalse(torch.equal(rule, model(other['rule_trees'][0], other['sequences'], encoder)[0]))
        (rule.square().sum() + states.square().sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        corrupt = deepcopy(features['rule_trees'][0])
        corrupt['constraint_references'][0][0]['symbol'] = 4  # Token payload still points to 9.
        with self.assertRaisesRegex(ValueError, 'bound constraint'):
            model(corrupt, features['sequences'], encoder)

    def test_rooted_multi_round_propagation_has_exact_receptive_field(self):
        from ml_orca.models.rule_policy_model import DirectedRuleAggregation
        layers = [DirectedRuleAggregation(8, rooted=True) for _ in range(3)]
        edges = [[0, 1], [1, 2], [2, 3]]
        nodes, positions, roots = torch.randn(4, 8), torch.randn(3, 8), torch.randn(3, 2, 8)
        changed = nodes.clone()
        changed[0] += 1
        for round_index, layer in enumerate(layers, 1):
            nodes, changed = layer(nodes, edges, positions, roots), layer(changed, edges, positions, roots)
            self.assertFalse(torch.equal(nodes[round_index], changed[round_index]))
            if round_index < 3:
                torch.testing.assert_close(nodes[round_index + 1:], changed[round_index + 1:], rtol=0, atol=0)
        nodes.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for layer in layers for p in layer.parameters()))

    def test_predictor_uses_exact_roots_and_never_uses_flat_dsl_or_future_fields(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        features, vocabulary = tree_features([ir_fixture(), ir_fixture()], ('r', 'r/0'))
        model = TreePolicyPredictor(len(vocabulary), 8, message_rounds=3)
        self.assertEqual(len(model.rounds), 3)
        before = model(features)
        poisoned = deepcopy(features)
        poisoned['sequences']['rule'] = [{'invalid_flat_sentence': True}]
        poisoned.update(future_memo=999, response={'execution_ms': 123})
        torch.testing.assert_close(before, model(poisoned), rtol=0, atol=0)
        changed = deepcopy(features)
        changed['edge_roots'][0][0] = 3  # Same direction/policy, a different actual target subtree.
        self.assertFalse(torch.equal(before, model(changed)))
        changed = deepcopy(features)
        changed['edge_roots'][0][1] = 1  # Consumer source subtree, not always its whole-rule root.
        self.assertFalse(torch.equal(before, model(changed)))
        invalid = deepcopy(features)
        invalid['edge_roots'][0][0] = 2  # Source-side root cannot be used as producer target.
        with self.assertRaisesRegex(ValueError, 'wrong side'):
            model(invalid)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad()
        before.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model(features)))
        features['loaded_rules'] = [False, False]
        self.assertTrue(torch.isfinite(model(features)).all())

    def test_tree_graph_rule_renumbering_parallel_ports_and_checkpoint(self):
        from ml_orca.models.rule_tree_model import TreePolicyPredictor
        second = ir_fixture()
        second['constraints'][1]['symbols'] = [6, 3]
        features, vocabulary = tree_features([ir_fixture(), second], ('r', 'r/0'))
        model = TreePolicyPredictor(len(vocabulary), 8, message_rounds=3)
        original = model(features)
        permuted = deepcopy(features)
        permuted['rule_trees'].reverse()  # Feature ranges continue to address the original tensors.
        permuted['sequences']['policy'].reverse()
        permuted['rule_edges'] = [[1, 0], [1, 0]]
        torch.testing.assert_close(original, model(permuted))
        reordered = deepcopy(features)
        for field in ('rule_edges', 'edge_roots'):
            reordered[field].reverse()
        reordered['sequences']['edge'].reverse()
        torch.testing.assert_close(original, model(reordered))
        clone = TreePolicyPredictor(len(vocabulary), 8, message_rounds=3)
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(original, clone(features), rtol=0, atol=0)
        extra = deepcopy(features)
        extra['rule_trees'].append({'unused': True})
        extra['sequences']['policy'].append({'unused': True})
        extra['loaded_rules'].append(False)
        extra['rule_edges'].append([2, 0])
        extra['edge_roots'].append([999, 999])
        extra['sequences']['edge'].append({'unused': True})
        torch.testing.assert_close(original, model(extra), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
