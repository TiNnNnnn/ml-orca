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
