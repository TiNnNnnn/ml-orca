from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest('input prefetch tests require PyTorch')

from ml_orca.common.artifacts import artifact_snapshot
from ml_orca.encoding.observed_search import observed_features
from ml_orca.encoding.rule_policy_encoding import encode_sequence, encode_sequence_groups, fit_vocabulary
from ml_orca.models.rule_tree_model import ObservedSearchPredictor
from ml_orca.tests.test_rule_history_encoding import history_fixture
from ml_orca.training.inputs import ObservedInputDataset, input_loader, resident_size


class FixtureBuilder:
    def __init__(self, feature):
        self.feature, self.calls = feature, 0

    def __call__(self, item, raw):
        self.calls += 1
        return deepcopy(self.feature)


class InputPipelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'graph'
        self.path.write_bytes(b'original immutable graph')
        snap = artifact_snapshot({'graph': self.path})['graph']
        self.items = [{'graph_snapshot': snap} for _ in range(3)]
        base, history = history_fixture()
        graph = deepcopy(history['runs'][0])
        graph['trees'] = graph['trees'][:1]
        graph['contexts'] = graph['contexts'][:1]
        binding = {**base['query_binding'], 'sequence': base['sequences']['query'][0]}
        self.feature = observed_features(base, graph, binding, {'0': 0, '1': 1})
        self.vocabulary = fit_vocabulary([self.feature['sequences']])

    def dataset(self, budget=10000000, roots=None):
        return ObservedInputDataset(self.items, FixtureBuilder(self.feature), self.vocabulary, budget, roots)

    def test_warm_cache_preserves_inputs_and_checks_source_content(self):
        dataset = self.dataset()
        _, cold, a = dataset[0]
        _, warm, b = dataset[0]
        expected = deepcopy(self.feature)
        expected['sequences'] = {k: [encode_sequence(s, self.vocabulary) for s in rows]
                                 for k, rows in expected['sequences'].items()}
        self.assertEqual(cold, expected)
        self.assertIs(cold, warm)
        self.assertFalse(a['cache_hit'])
        self.assertTrue(b['cache_hit'])
        self.assertEqual(dataset.builder.calls, 1)
        self.path.write_bytes(b'changed immutable graph!')
        with self.assertRaisesRegex(ValueError, 'snapshot content changed'):
            dataset[0]

    def test_byte_budget_eviction_and_no_learned_values(self):
        dataset = self.dataset()
        _, _, result = dataset[0]
        dataset.cache_bytes = result['cache_bytes']
        dataset[1]
        self.assertEqual(list(dataset.cache), [1])
        self.assertFalse(dataset[0][2]['cache_hit'])
        self.assertLessEqual(dataset.used_bytes, dataset.cache_bytes)
        small = self.dataset(1)
        self.assertFalse(small[0][2]['cache_hit'])
        self.assertEqual(small.used_bytes, 0)
        with self.assertRaises(TypeError):
            resident_size({'learned': torch.ones(1, requires_grad=True)})

    def test_sequence_sharing_preserves_values_occurrences_and_numeric_validation(self):
        groups = {'edge': [[('x', 1.)]] * 100, 'other': [[('x', 1.)], [('x', -0.)], [('x', 0.)]]}
        vocabulary = fit_vocabulary([groups])
        expected = {k: [encode_sequence(s, vocabulary) for s in rows] for k, rows in groups.items()}
        shared = encode_sequence_groups(groups, vocabulary)
        self.assertEqual(shared, expected)
        self.assertIs(shared['edge'][0], shared['edge'][-1])
        self.assertIs(shared['edge'][0], shared['other'][0])
        self.assertIsNot(shared['other'][1], shared['other'][2])
        self.assertLess(resident_size(shared), resident_size(expected) / 4)
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                encode_sequence_groups({'edge': [[('x', invalid)]]}, vocabulary)

    def test_spawn_prefetch_preserves_order_tuples_rng_and_relocation(self):
        original_root = '/original/ml-orca-test'
        self.items = [{'graph_snapshot': {**i['graph_snapshot'], 'path': original_root + '/graph'}} for i in self.items]
        dataset = self.dataset(roots=(original_root, self.directory.name))
        order = [2, 0, 1]
        loader = input_loader(dataset, order, workers=2, prefetch=1)
        rng = torch.get_rng_state().clone()
        first = list(loader)
        order[:] = [1, 2]  # A resumed suffix / validation order, same workers.
        second = list(loader)
        self.assertEqual([i for i, _, _ in first], [2, 0, 1])
        self.assertEqual([i for i, _, _ in second], order)
        for _, data, _ in first + second:
            self.assertEqual(data, dataset[0][1])
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        del loader

    def test_cached_and_prefetched_model_gradients_and_adam_are_identical(self):
        torch.set_num_threads(1)
        torch.manual_seed(929)
        reference = ObservedSearchPredictor(len(self.vocabulary), 4, 'static', 3, history=True)
        candidate = deepcopy(reference)
        optimizers = [torch.optim.Adam(m.parameters(), lr=.003) for m in (reference, candidate)]
        baseline, cached = self.dataset(0), self.dataset()
        order = [2, 0, 2]
        loader = input_loader(cached, order, workers=2, prefetch=1)
        for index, prepared, _ in loader:
            predictions = []
            for model, optimizer, features in zip((reference, candidate), optimizers,
                                                  (baseline[index][1], prepared)):
                optimizer.zero_grad()
                predicted = model(features)
                predictions.append(predicted)
                torch.nn.functional.smooth_l1_loss(predicted, predicted.new_tensor([1., 2.])).backward()
            torch.testing.assert_close(*predictions, rtol=0, atol=0)
            for a, b in zip(reference.parameters(), candidate.parameters()):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            for optimizer in optimizers:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 5.)
                optimizer.step()
            for a, b in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        del loader


if __name__ == '__main__':
    unittest.main()
