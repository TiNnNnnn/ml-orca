from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    raise unittest.SkipTest('install tools/ml-orca/requirements-training.txt for network tests')

from ml_orca.encoding.rule_policy_encoding import encode_sequence, fit_vocabulary
from ml_orca.tests.test_rule_history_encoding import history_fixture
from ml_orca.training.train_observed_search import (ObservedSearchPredictor, internal_split, observed_features,
                                   observed_target, restore_checkpoint, training_contract, validate_resume_contract)
from ml_orca.training.runtime import training_device
from ml_orca.objectives.observed_search import validate_frozen_target


class ObservedSearchTest(unittest.TestCase):
    def test_frozen_labels_allow_only_adjacent_libm_rounding_without_mutation(self):
        summary = {'complete': True, 'returncode': 0, 'attempts': 1,
                   'groups': [{'attempts': 1, 'match_us': 1881, 'constraint_us': 0, 'instantiate_us': 0}]}
        target = observed_target(summary)
        for direction in (-math.inf, math.inf):
            adjacent = math.nextafter(target[0], direction)
            saved = [adjacent, 0.]
            validate_frozen_target(summary, saved)
            self.assertEqual(saved, [adjacent, 0.])
            with self.assertRaises(ValueError):
                validate_frozen_target(summary, [math.nextafter(adjacent, direction), 0.])
        for bad in ([target[0], math.nextafter(0., 1.)], [float('nan'), 0.], [float('inf'), 0.],
                    [-1., 0.], [True, 0.], [], None):
            with self.assertRaises(ValueError):
                validate_frozen_target(summary, bad)
        with self.assertRaises(ValueError):
            validate_frozen_target(summary | {'complete': False}, target)

    def test_device_selection_never_silently_falls_back(self):
        self.assertEqual(training_device('cpu'), torch.device('cpu'))
        with self.assertRaises(ValueError):
            training_device('meta')
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                training_device('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires a CUDA worker')
    def test_cpu_cuda_predictions_gradients_and_resume(self):
        device = training_device('cuda')
        torch.set_num_threads(1)
        base, history = history_fixture()
        graph = deepcopy(history['runs'][0])
        graph['trees'] = graph['trees'][:1]
        graph['contexts'] = graph['contexts'][:1]
        binding = {**base['query_binding'], 'sequence': base['sequences']['query'][0]}
        data = observed_features(base, graph, binding, {'0': 0, '1': 1})
        vocab = fit_vocabulary([data['sequences']])
        data['sequences'] = {k: [encode_sequence(s, vocab) for s in rows] for k, rows in data['sequences'].items()}
        torch.manual_seed(929)
        cpu = ObservedSearchPredictor(len(vocab), 4, 'static', 3, history=True)
        gpu = deepcopy(cpu).to(device)
        optimizers = [torch.optim.Adam(m.parameters(), lr=.003, foreach=False) for m in (cpu, gpu)]
        for _ in range(3):
            predictions = []
            for model, optimizer in zip((cpu, gpu), optimizers):
                optimizer.zero_grad()
                predicted = model(data)
                predictions.append(predicted.detach().cpu())
                torch.nn.functional.smooth_l1_loss(predicted, predicted.new_tensor([1., 2.])).backward()
            torch.testing.assert_close(*predictions, rtol=2e-5, atol=2e-6)
            for a, b in zip(cpu.parameters(), gpu.parameters()):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=5e-4, atol=2e-6)
            for optimizer in optimizers:
                optimizer.step()
            for a, b in zip(cpu.parameters(), gpu.parameters()):
                torch.testing.assert_close(a, b.cpu(), rtol=5e-4, atol=2e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            torch.save({'model': gpu.state_dict(), 'optimizer': optimizers[1].state_dict(), 'steps': 3,
                        'epoch': 1, 'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state(device)}, path)
            clone = deepcopy(gpu)
            opt = torch.optim.Adam(clone.parameters(), lr=.1, foreach=False)
            restore_checkpoint(path, clone, opt, 3)
            for model, optimizer in ((gpu, optimizers[1]), (clone, opt)):
                optimizer.zero_grad()
                model(data).square().mean().backward()
                optimizer.step()
            for a, b in zip(gpu.parameters(), clone.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_resume_rejects_other_network_or_target(self):
        contract = training_contract(32, 'graph')
        validate_resume_contract({'training_contract': contract}, 32, 'graph')
        validate_resume_contract({'width': 32}, 32, 'graph')  # Legacy checkpoint.
        for bad in ({'training_contract': training_contract(16, 'graph')},
                    {'training_contract': training_contract(32, 'context')},
                    {'training_contract': contract | {'objective': {'name': 'policy_value'}}},
                    {'width': 16}):
            with self.assertRaises(ValueError):
                validate_resume_contract(bad, 32, 'graph')

    def test_resume_preserves_optimizer_and_does_not_invent_old_loss(self):
        torch.manual_seed(929)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        def update(net, opt):
            opt.zero_grad()
            net(torch.tensor([[1., 2.]])).square().mean().backward()
            opt.step()
        update(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'steps': 1, 'epoch': 1}
            torch.save(checkpoint, path)
            clone = torch.nn.Linear(2, 1)
            other = torch.optim.Adam(clone.parameters(), lr=.1)
            self.assertEqual(restore_checkpoint(path, clone, other, 3), (1, 0, 1, 0., 0))
            update(model, optimizer)
            update(clone, other)
            for actual, expected in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checkpoint.update(steps=3, epoch_loss_sum=2., epoch_loss_count=3)
            torch.save(checkpoint, path)
            self.assertEqual(restore_checkpoint(path, clone, other, 3), (3, 0, 3, 2., 3))
            for bad in ({'steps': -1}, {'epoch': 2}, {'epoch_loss_count': 4}):
                torch.save(checkpoint | bad, path)
                with self.assertRaises(ValueError):
                    restore_checkpoint(path, clone, other, 3)

    def test_targets_are_existing_microseconds_and_never_imputed(self):
        summary = {'complete': True, 'returncode': 0, 'attempts': 2,
                   'groups': [{'attempts': 2, 'match_us': 2000, 'constraint_us': 3000,
                               'instantiate_us': 7000}]}
        self.assertEqual(observed_target(summary), [math.log1p(5), math.log1p(7)])
        for patch in ({'complete': False}, {'groups': None}, {'attempts': 3}):
            with self.assertRaises(ValueError):
                observed_target(summary | patch)
        summary['groups'][0]['match_us'] = float('nan')
        with self.assertRaises(ValueError):
            observed_target(summary)
        self.assertEqual(internal_split('same_family', 929), internal_split('same_family', 929))

    def test_real_tree_gnn_updates_without_sql_and_excludes_outcome_tokens(self):
        base, history = history_fixture()
        graph = deepcopy(history['runs'][0])
        graph['trees'] = graph['trees'][:1]
        graph['contexts'] = graph['contexts'][:1]
        binding = {**base['query_binding'], 'sequence': base['sequences']['query'][0]}
        feature = observed_features(base, graph, binding, {'0': 0, '1': 1})
        other = deepcopy(graph)
        other['edges'][0]['producer_outcome'] = 'different_outcome'
        self.assertEqual(feature, observed_features(base, other, binding, {'0': 0, '1': 1}))
        self.assertNotIn('history_admission', feature)
        vocab = fit_vocabulary([feature['sequences']])
        feature['sequences'] = {k: [encode_sequence(s, vocab) for s in v]
                                for k, v in feature['sequences'].items()}
        torch.manual_seed(929)
        torch.set_num_threads(1)
        model = ObservedSearchPredictor(len(vocab), 4, 'static', 3, history=True)
        optimizer = torch.optim.Adam(model.parameters())
        before = model.readout[-1].weight.detach().clone()
        predicted = model(feature)
        self.assertEqual(predicted.shape, (2,))
        loss = (predicted - torch.tensor([1., 2.])).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.readout[-1].weight))
        with self.assertRaisesRegex(ValueError, 'retrospective'):
            model({**feature, 'observation_scope': 'prospective'})


if __name__ == '__main__':
    unittest.main()
