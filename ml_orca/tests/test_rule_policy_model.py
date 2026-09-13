"""Optional neural computation checks; no claim about measured policy prediction."""

from copy import deepcopy
import unittest

try:
    import torch
except ImportError:
    torch = None

from ml_orca.encoding.rule_policy_encoding import encode_sequence, fit_vocabulary


@unittest.skipIf(torch is None, 'PyTorch is optional; numeric input checks do not require it')
class SequenceEncoderTest(unittest.TestCase):
    def query_features(self, sql='SELECT a.id FROM a', catalog=None, vocabulary=None):
        from ml_orca.tests.test_rule_policy_encoding import catalog_fixture
        from ml_orca.encoding.rule_policy_encoding import catalog_sequences
        from ml_orca.encoding.query_policy_encoding import query_context
        if catalog is None:
            catalog = catalog_fixture()
            catalog['settings'] = {'search_path': 'public'}
            for relation in catalog['relations']:
                relation['schema'] = 'public'
        binding = query_context(sql, catalog)
        self.assertTrue(binding['complete'], binding['errors'])
        groups = {'query': [binding.pop('sequence')], 'relation': catalog_sequences(catalog)[0]}
        vocabulary = vocabulary or fit_vocabulary([groups])
        return {'query_binding': binding, 'sequences': {
            key: [encode_sequence(s, vocabulary) for s in seqs] for key, seqs in groups.items()}}, vocabulary, catalog

    def setUp(self):
        from ml_orca.models.rule_policy_model import SharedSequenceEncoder
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.raw = [[('op:Input', None), ('symbol', 0)],
                    [('symbol', 0), ('op:Input', None), ('rows', 3.)]]
        vocab = fit_vocabulary([{'rule': self.raw}])
        self.encoded = [encode_sequence(seq, vocab) for seq in self.raw]
        self.model = SharedSequenceEncoder(len(vocab), 8)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_repeat_padding_order_missing_and_parameter_registration(self):
        self.model.eval()
        parameters = {name: id(p) for name, p in self.model.named_parameters()}
        first = self.model(self.encoded)
        torch.testing.assert_close(first, self.model(self.encoded), rtol=0, atol=0)
        torch.testing.assert_close(first[0], self.model(self.encoded[:1])[0])
        torch.testing.assert_close(first.flip(0), self.model(list(reversed(self.encoded))))
        changed = deepcopy(self.encoded[:1])
        changed[0]['known_number'][1] = False
        self.assertFalse(torch.equal(first[:1], self.model(changed)))
        self.assertEqual(parameters, {name: id(p) for name, p in self.model.named_parameters()})
        self.assertEqual(tuple(self.model([]).shape), (0, 8))

    def test_shared_parameters_can_learn_and_reject_invalid_input(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.03)
        target = torch.tensor([[1.] * 8, [-1.] * 8])
        initial = (self.model(self.encoded) - target).square().mean().item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = (self.model(self.encoded) - target).square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial / 4)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        for key, value in (('tokens', [999, 2]), ('numbers', [float('nan'), 0]), ('known_number', [False])):
            bad = deepcopy(self.encoded[:1])
            bad[0][key] = value
            with self.assertRaises(ValueError):
                self.model(bad)

    def test_equal_input_interning_preserves_outputs_gradients_and_validation(self):
        import json
        from unittest.mock import patch
        sequences = [self.encoded[i % 2] for i in range(31)]
        with patch.object(self.model, '_encode', wraps=self.model._encode) as encode, \
                patch('ml_orca.models.sequence.json.dumps', wraps=json.dumps) as serialize:
            actual = self.model(sequences)
            self.assertEqual(len(encode.call_args.args[0]), 2)
            self.assertEqual(serialize.call_count, 2)
        weights = torch.linspace(-2., 3., len(sequences)).unsqueeze(1)
        (actual * weights).sum().backward()
        grads = {name: p.grad.clone() for name, p in self.model.named_parameters()}
        self.model.zero_grad()
        expected = self.model._encode(sequences)
        (expected * weights).sum().backward()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(p.grad, grads[name], rtol=2e-5, atol=2e-6)
        # Equal-looking invalid types cannot hide behind a valid representative.
        bad = deepcopy(sequences)
        bad[-1]['known_number'][0] = int(bad[-1]['known_number'][0])
        with self.assertRaises(ValueError):
            self.model(bad)
        mutated = deepcopy(sequences)
        self.model(mutated)
        mutated[0]['tokens'][0] = True
        with self.assertRaises(ValueError):
            self.model(mutated)  # Identity reuse never bypasses validation across forwards.
        with torch.no_grad():
            self.model.numbers.weight.add_(.1)
        self.assertFalse(torch.equal(actual, self.model(sequences)))

    def test_masked_pooling_matches_unpadded_values_and_gradients(self):
        sequences = self.encoded + [self.encoded[0]] * 4
        # Padded embeddings need not be zero in an imported checkpoint.
        with torch.no_grad():
            self.model.tokens.weight[0].fill_(19)
        contexts = [torch.randn(len(s['tokens']), 8, requires_grad=True) for s in sequences]
        batch = self.model(sequences, contexts)
        batch.square().sum().backward()
        grads = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        input_grads = [c.grad.clone() for c in contexts]
        self.model.zero_grad()
        for c in contexts:
            c.grad = None
        singles = torch.cat([self.model([s], [c]) for s, c in zip(sequences, contexts)])
        singles.square().sum().backward()
        torch.testing.assert_close(batch, singles)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter.grad, grads[name], rtol=2e-5, atol=2e-6)
        for context, gradient in zip(contexts, input_grads):
            torch.testing.assert_close(context.grad, gradient)

    def test_rule_attempt_fuses_local_context_with_shared_parameters(self):
        from ml_orca.models.rule_policy_model import RuleAttemptEncoder
        from ml_orca.encoding.group_expression_encoding import group_expression_sequence
        from ml_orca.trace.profile_rule_candidates import candidate_state
        from ml_orca.tests.test_group_expression_encoding import attempt_fixture
        row = attempt_fixture()
        first = group_expression_sequence(candidate_state(row))
        row['input_context']['root']['rows'] = 10000
        row['input_context']['children'][1]['node']['rows'] = 10000
        second = group_expression_sequence(candidate_state(row))
        vocab = fit_vocabulary([{'rule': self.raw, 'context': [first, second]}])
        contexts = [encode_sequence(seq, vocab) for seq in (first, second)]
        rules = [encode_sequence(self.raw[0], vocab)] * 2
        model = RuleAttemptEncoder(len(vocab), 8)
        before = {name: id(p) for name, p in model.named_parameters()}
        output = model(rules, contexts)
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertFalse(torch.equal(output[0], output[1]))
        torch.testing.assert_close(output.flip(0), model(rules, contexts[::-1]))
        self.assertFalse(torch.equal(output, model(rules, contexts, rules)))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(before, {name: id(p) for name, p in model.named_parameters()})
        self.assertEqual(tuple(model([], []).shape), (0, 8))
        with self.assertRaises(ValueError):
            model(rules, contexts[:1])

    def test_padded_token_contexts_preserve_outputs_and_all_gradients(self):
        contexts = [torch.randn(len(s['tokens']), 8, requires_grad=True) for s in self.encoded]
        padded = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True).detach().requires_grad_()
        expected = self.model(self.encoded, contexts)
        expected.square().sum().backward()
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters()}
        self.model.zero_grad()
        actual = self.model(self.encoded, padded)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        for row, context in enumerate(contexts):
            torch.testing.assert_close(padded.grad[row, :len(context)], context.grad, rtol=0, atol=0)
            self.assertTrue((padded.grad[row, len(context):] == 0).all())
        for invalid in (padded[:, :-1], padded.double(), padded.detach() * float('nan')):
            with self.assertRaisesRegex(ValueError, 'invalid token context'):
                self.model(self.encoded, invalid)

    def test_query_metadata_context_is_bound_per_source_and_identity_invariant(self):
        from ml_orca.models.rule_policy_model import query_context_embedding, SharedSequenceEncoder
        features, vocabulary, catalog = self.query_features('SELECT x.id FROM a x JOIN a y ON x.id=y.id')
        model = SharedSequenceEncoder(len(vocabulary), 8)
        before = query_context_embedding(model, features)
        changed = deepcopy(catalog)
        changed['relations'][0]['estimated_rows'] = 10000
        changed_features, _, _ = self.query_features('SELECT x.id FROM a x JOIN a y ON x.id=y.id', changed, vocabulary)
        self.assertFalse(torch.equal(before, query_context_embedding(model, changed_features)))
        changed = deepcopy(catalog)
        changed['statistics'][0]['n_distinct'] = -.2
        changed_features, _, _ = self.query_features('SELECT x.id FROM a x JOIN a y ON x.id=y.id', changed, vocabulary)
        self.assertFalse(torch.equal(before, query_context_embedding(model, changed_features)))
        changed = deepcopy(catalog)
        changed['columns'][0]['not_null'] = False
        changed_features, _, _ = self.query_features('SELECT x.id FROM a x JOIN a y ON x.id=y.id', changed, vocabulary)
        self.assertFalse(torch.equal(before, query_context_embedding(model, changed_features)))
        # Reordering the catalog changes only reference indices, never its query-conditioned meaning.
        reordered = deepcopy(catalog)
        reordered['relations'].reverse()
        permuted, _, _ = self.query_features('SELECT x.id FROM a x JOIN a y ON x.id=y.id', reordered, vocabulary)
        torch.testing.assert_close(before, query_context_embedding(model, permuted))
        unused = deepcopy(features)
        unused['sequences']['relation'][1]['numbers'] = [999 if known else 0 for known in
                                                       unused['sequences']['relation'][1]['known_number']]
        torch.testing.assert_close(before, query_context_embedding(model, unused), rtol=0, atol=0)
        renamed, _, _ = self.query_features('SELECT u.id FROM a u JOIN a v ON u.id=v.id', catalog, vocabulary)
        torch.testing.assert_close(before, query_context_embedding(model, renamed), rtol=0, atol=0)

    def test_long_sequence_keeps_early_numeric_fields_and_missing_mask(self):
        from ml_orca.models.rule_policy_model import SharedSequenceEncoder
        sequence = [('rows', None)] + [('later_token', None)] * 512
        vocabulary = fit_vocabulary([{'relation': [sequence]}])
        encoder = SharedSequenceEncoder(len(vocabulary), 8)
        values = [encode_sequence([(sequence[0][0], value), *sequence[1:]], vocabulary)
                  for value in (None, 0., 1., 5.)]
        vectors = encoder(values)
        for i in range(1, len(values)):
            self.assertFalse(torch.equal(vectors[0], vectors[i]))
        self.assertFalse(torch.equal(vectors[2], vectors[3]))
        torch.testing.assert_close(vectors[:1], encoder(values[:1]))

    def test_whole_policy_readout_is_shared_and_ignores_future_graph_and_unloaded_rules(self):
        from ml_orca.models.rule_policy_model import WholePolicyPredictor
        features, vocabulary, _ = self.query_features()
        features['sequences'].update(rule=self.encoded, policy=list(reversed(self.encoded)))
        # Use a compatible toy vocabulary to test set/readout mechanics, not DSL semantics.
        features['sequences']['rule'] = features['sequences']['query'] * 2
        features['sequences']['policy'] = features['sequences']['relation'][:2]
        features['loaded_rules'] = [True, True]
        model = WholePolicyPredictor(len(vocabulary), 8)
        before = model(features)
        self.assertEqual(tuple(before.shape), (2,))
        permuted = deepcopy(features)
        for key in ('rule', 'policy'):
            permuted['sequences'][key].reverse()
        torch.testing.assert_close(before, model(permuted))
        extra = deepcopy(features)
        extra['sequences']['rule'].append({'deliberately_unused': True})
        extra['sequences']['policy'].append({'deliberately_unused': True})
        extra['loaded_rules'].append(False)
        extra.update(rule_edges=[[0, 1]], future_memo=999, execution_ms=999)
        torch.testing.assert_close(before, model(extra), rtol=0, atol=0)
        changed = deepcopy(features)
        changed['sequences']['policy'][0] = changed['sequences']['policy'][1]
        self.assertFalse(torch.equal(before, model(changed)))
        before.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        features['loaded_rules'] = [False, False]
        self.assertTrue(torch.isfinite(model(features)).all())
        features['loaded_rules'] = [1, True]
        with self.assertRaises(ValueError):
            model(features)

    def test_large_rule_catalog_readout_keeps_gradient_and_rule_count(self):
        from ml_orca.models.rule_policy_model import WholePolicyPredictor
        feature, vocabulary, _ = self.query_features()
        feature['sequences'].update(rule=feature['sequences']['query'] * 128,
                                    policy=feature['sequences']['query'] * 128)
        feature['loaded_rules'] = [True] * 128
        model = WholePolicyPredictor(len(vocabulary), 8)
        predicted = model(feature)
        predicted.sum().backward()
        self.assertGreater(float(model.readout[0].weight.grad.norm()), 0)
        feature['loaded_rules'] = [True] + [False] * 127
        self.assertFalse(torch.equal(predicted, model(feature)))

    def test_directed_messages_match_reference_and_are_one_step(self):
        from ml_orca.models.rule_policy_model import DirectedRuleAggregation
        model = DirectedRuleAggregation(2)
        nodes = torch.tensor([[.1, .2], [.3, -.4], [.5, .6]])
        edges = [[0, 1], [0, 1], [1, 2], [2, 2]]
        positions = torch.tensor([[.1, .2], [.3, .4], [.5, .6], [.7, .8]])
        actual = model(nodes, edges, positions)
        expected = []
        for i, node in enumerate(nodes):
            incoming = [torch.tanh(model.predecessor(torch.cat((nodes[s], pos))))
                        for (s, t), pos in zip(edges, positions) if t == i]
            outgoing = [torch.tanh(model.successor(torch.cat((nodes[t], pos))))
                        for (s, t), pos in zip(edges, positions) if s == i]
            means = [torch.stack(v).mean(0) if v else torch.zeros(2) for v in (incoming, outgoing)]
            degrees = torch.tensor([len(incoming), len(outgoing)]).log1p()
            expected.append(node + torch.tanh(model.update(torch.cat((node, *means, degrees)))))
        torch.testing.assert_close(actual, torch.stack(expected))
        torch.testing.assert_close(actual, model(nodes, edges[::-1], positions.flip(0)))
        changed = nodes.clone()
        changed[0] += .5
        after = model(changed, edges, positions)
        self.assertFalse(torch.equal(actual[1], after[1]))
        torch.testing.assert_close(actual[2], after[2], rtol=0, atol=0)  # No in-place two-hop propagation.
        self.assertFalse(torch.equal(actual, model(nodes, [[t, s] for s, t in edges], positions)))
        self.assertFalse(torch.equal(actual, model(nodes, edges, positions.flip(0))))
        self.assertFalse(torch.equal(actual, model(nodes, edges[:1] + edges[2:], positions[[0, 2, 3]])))
        # Permute nodes, remap endpoints, and undo the output permutation.
        permutation, inverse = [2, 0, 1], [1, 2, 0]
        remapped = [[inverse[s], inverse[t]] for s, t in edges]
        torch.testing.assert_close(actual, model(nodes[permutation], remapped, positions)[inverse])
        actual.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(tuple(model(nodes[:0], [], positions[:0]).shape), (0, 2))
        with self.assertRaises(ValueError):
            model(nodes, [[True, 1]], positions[:1])

    def test_whole_policy_static_graph_has_matched_self_control_and_no_history_input(self):
        from ml_orca.models.rule_policy_model import WholePolicyPredictor
        feature, vocab, _ = self.query_features()
        feature['sequences'].update(rule=feature['sequences']['relation'][:2],
                                    policy=feature['sequences']['query'] * 2,
                                    edge=list(feature['sequences']['query']))
        feature.update(loaded_rules=[True, True], rule_edges=[[0, 1]], rule_graph_scope='native_static_template')
        model = WholePolicyPredictor(len(vocab), 8, 'static')
        control = WholePolicyPredictor(len(vocab), 8, 'self')
        control.load_state_dict(model.state_dict())
        actual = model(feature)
        self.assertFalse(torch.equal(actual, control(feature)))
        extra = deepcopy(feature)
        extra['sequences']['rule'].append({'unused': True})
        extra['sequences']['policy'].append({'unused': True})
        extra['sequences']['edge'].append({'unused': True})
        extra['loaded_rules'].append(False)
        extra['rule_edges'].append([2, 0])
        extra.update(future_memo=999, runtime_counts=[123], response={'planning_ms': 100})
        torch.testing.assert_close(actual, model(extra), rtol=0, atol=0)
        torch.testing.assert_close(control(feature), control(extra), rtol=0, atol=0)
        extra['rule_graph_scope'] = 'unadmitted_history'
        with self.assertRaisesRegex(ValueError, 'verified native static'):
            model(extra)
        permuted = deepcopy(feature)
        for channel in ('rule', 'policy'):
            permuted['sequences'][channel].reverse()
        permuted['rule_edges'] = [[1, 0]]
        torch.testing.assert_close(actual, model(permuted))
        actual.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        feature['loaded_rules'] = [False, False]
        self.assertTrue(torch.isfinite(model(feature)).all())
        torch.manual_seed(929)
        baseline = WholePolicyPredictor(len(vocab), 8)
        torch.manual_seed(929)
        graph = WholePolicyPredictor(len(vocab), 8, 'static')
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, graph.state_dict()[name], rtol=0, atol=0)

    def test_query_metadata_gradients_and_rule_fusion_share_the_encoder(self):
        from ml_orca.models.rule_policy_model import query_context_embedding, RuleAttemptEncoder
        features, vocabulary, _ = self.query_features()
        model = RuleAttemptEncoder(len(vocabulary), 8)
        query = query_context_embedding(model.encoder, features)
        dummy = features['sequences']['query']
        output = model(dummy, dummy, query_embeddings=query)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(torch.equal(output, model(dummy, dummy)))
        for value in (torch.zeros(1, 9), torch.full((1, 8), float('nan'))):
            with self.assertRaises(ValueError):
                model(dummy, dummy, query_embeddings=value)
        with self.assertRaises(ValueError):
            model(dummy, dummy, queries=dummy, query_embeddings=query)

    def test_query_metadata_missing_derived_and_invalid_binding(self):
        from ml_orca.models.rule_policy_model import query_context_embedding, SharedSequenceEncoder
        features, vocabulary, _ = self.query_features('WITH x AS (SELECT a.id FROM a) SELECT x.id FROM x')
        self.assertTrue(any(s['kind'] == 'derived' for s in features['query_binding']['sources']))
        model = SharedSequenceEncoder(len(vocabulary), 8)
        self.assertTrue(torch.isfinite(query_context_embedding(model, features)).all())
        for mutate in (
            lambda b: b.update(complete=False),
            lambda b: b['source_references'].append(b['source_references'][0]),
            lambda b: b['source_references'][0].update(token=-1),
            lambda b: b['source_references'][0].update(source=999),
            lambda b: b['sources'][0].update(relation=999)):
            changed = deepcopy(features)
            mutate(changed['query_binding'])
            with self.assertRaises(ValueError):
                query_context_embedding(model, changed)
        constant, vocabulary, _ = self.query_features('SELECT 1')
        model = SharedSequenceEncoder(len(vocabulary), 8)
        torch.testing.assert_close(query_context_embedding(model, constant), model(constant['sequences']['query']))
        with self.assertRaises(ValueError):
            model(constant['sequences']['query'], [torch.zeros(1, 8)])


if __name__ == '__main__':
    unittest.main()
