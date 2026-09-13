"""Same equations, ordered trees, all constraints, gradients and optimizer updates."""
from copy import deepcopy
import unittest

from ml_orca.tests.scalar_tree_reference import ScalarRuleTreeEncoder, torch
from ml_orca.models.rule_tree_model import RuleTreeEncoder, TreePolicyPredictor, ObservedSearchPredictor
from ml_orca.tests.test_rule_tree_model import tree_features, ir_fixture
from ml_orca.tests.test_rule_history_encoding import history_fixture
from ml_orca.encoding.observed_search import observed_features
from ml_orca.encoding.rule_policy_encoding import fit_vocabulary, encode_sequence


class ForestBatchingTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(305)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def close(self, left, right, dtype):
        torch.testing.assert_close(left, right, rtol=2e-4 if dtype == torch.float32 else 1e-9,
                                   atol=2e-6 if dtype == torch.float32 else 1e-11)

    def test_scalar_and_layered_forests_all_states_input_and_parameter_gradients(self):
        # Different depths/arities plus repeated child uses exercise accumulation.
        nodes = [{'children': []} for _ in range(8)]
        nodes += [{'children': list(range(n))} for n in (1, 2, 3, 7)]
        for _ in range(8):
            nodes.append({'children': [len(nodes) - 1]})
        nodes += [{'children': [8, 12, 1]}, {'children': [4, 4, 19]}]
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                reference, model = ScalarRuleTreeEncoder(8).to(dtype), RuleTreeEncoder(8).to(dtype)
                model.load_state_dict(reference.state_dict())
                values = torch.randn(len(nodes), 8, dtype=dtype, requires_grad=True)
                other = values.detach().clone().requires_grad_()
                expected, actual = reference.tree_states(values, nodes), model.tree_states(other, nodes)
                self.close(expected, actual, dtype)
                weights = torch.randn_like(actual)
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                self.close(values.grad, other.grad, dtype)
                for (name, p), (other_name, q) in zip(reference.named_parameters(), model.named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertEqual(p.grad is None, q.grad is None, name)
                    if p.grad is not None:
                        self.close(p.grad, q.grad, dtype)
        for invalid in ([{'children': [0]}], [{'children': [-1]}], [{'children': [True]}]):
            with self.assertRaises(ValueError):
                model.tree_states(torch.randn(1, 8, dtype=dtype), invalid)

    def compare_updates(self, features, vocabulary, observed):
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype, observed=observed):
                cls = ObservedSearchPredictor if observed else TreePolicyPredictor
                model = cls(len(vocabulary), 8, 'static', 3, history=observed).to(dtype)
                reference = deepcopy(model)
                reference.rule_trees = ScalarRuleTreeEncoder(8).to(dtype)
                reference.load_state_dict(model.state_dict(), strict=True)
                optimizers = [torch.optim.Adam(m.parameters(), lr=.003) for m in (reference, model)]
                for _ in range(3):
                    outputs = []
                    for net, optimizer in zip((reference, model), optimizers):
                        optimizer.zero_grad()
                        output = net(features)
                        torch.nn.functional.smooth_l1_loss(output, output.new_tensor([1.2, .5])).backward()
                        outputs.append(output)
                    self.close(*outputs, dtype)
                    for (name, p), (other_name, q) in zip(reference.named_parameters(), model.named_parameters()):
                        self.assertEqual(name, other_name)
                        self.assertEqual(p.grad is None, q.grad is None, name)
                        if p.grad is not None:
                            self.close(p.grad, q.grad, dtype)
                    for net, optimizer in zip((reference, model), optimizers):
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
                        optimizer.step()
                    for p, q in zip(reference.parameters(), model.parameters()):
                        self.close(p, q, dtype)

    def test_whole_policy_constraints_roots_and_three_adam_updates(self):
        changed = ir_fixture()
        changed['constraints'][1]['symbols'] = [6, 3]
        changed['constraints'].append(deepcopy(changed['constraints'][0]))
        empty = ir_fixture()
        empty['constraints'] = []
        features, vocabulary = tree_features([ir_fixture(), changed, empty], ('r', 'r/0'))
        self.compare_updates(features, vocabulary, False)
        # Rule-local numeric IDs stay unchanged even with reversed forest order.
        features['rule_trees'].reverse()
        features['sequences']['policy'].reverse()
        features['rule_edges'] = [[2, 0], [2, 0]]
        self.compare_updates(features, vocabulary, False)

    def test_observed_contexts_parallel_edges_keep_all_gradients(self):
        base, history = history_fixture()
        graph = history['runs'][0]
        graph['trees'] = graph['trees'][:1] * 3
        graph['contexts'] = graph['contexts'][:1] * 4
        graph['edges'] = graph['edges'] * 7
        binding = base['query_binding'] | {'sequence': base['sequences']['query'][0]}
        features = observed_features(base, graph, binding, {'0': 0, '1': 1})
        vocabulary = fit_vocabulary([features['sequences']])
        features['sequences'] = {k: [encode_sequence(s, vocabulary) for s in rows]
                                 for k, rows in features['sequences'].items()}
        self.compare_updates(features, vocabulary, True)

    def test_empty_rule_symbols_and_constraints(self):
        ir = {'schema_version': 1, 'symbols': [], 'constraints': [],
              'source': {'op': 'Input', 'symbols': [], 'children': []},
              'target': {'op': 'Input', 'symbols': [], 'children': []}}
        features, vocabulary = tree_features([ir])
        self.compare_updates(features, vocabulary, False)
