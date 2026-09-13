#!/usr/bin/env python3
"""Dependency-free invariants for the learning input interface, not model accuracy."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from ml_orca.encoding.rule_policy_encoding import (catalog_sequences, encode_sequence, fit_vocabulary,
                                  input_sequences, number, policy_sequence, rule_sequence)


def ir_fixture():
    def tree(offset):
        return {'op': 'Filter', 'symbols': [offset, offset + 1], 'children': [
            {'op': 'Filter', 'symbols': [offset + 2, offset + 3], 'children': [
                {'op': 'Input', 'symbols': [offset + 4], 'children': []}]}]}
    return {'schema_version': 1, 'source': tree(0), 'target': tree(5),
            'symbols': [{'kind': k, 'side': side} for side in ('source', 'target') for k in ('p','a','p','a','t')],
            'constraints': [{'kind': k, 'symbols': refs} for k, refs in (
                ('TableEq', [9,4]), ('AttrsEq', [6,1]), ('AttrsEq', [8,3]),
                ('PredicateEq', [5,0]), ('PredicateEq', [7,2]))]}


def catalog_fixture():
    return {'relations': [{'oid': '1', 'name': 'a', 'relkind': 'r', 'estimated_rows': None},
                         {'oid': '2', 'name': 'b', 'relkind': 'r', 'estimated_rows': 0}],
            'columns': [{'relation_oid': '1', 'position': 1, 'name': 'id', 'type': 'integer', 'not_null': True}],
            'statistics': [{'relation_oid': '1', 'position': 1, 'inherited': False, 'n_distinct': -0.5,
                            'null_frac': 0, 'avg_width': 4, 'correlation': None, 'most_common_freqs': None}],
            'constraints': [{'relation_oid': '1', 'kind': 'f', 'column_positions': [1],
                             'referenced_relation_oid': '2', 'referenced_column_positions': [1], 'validated': True}],
            'indexes': []}


class EncodingTest(unittest.TestCase):
    def test_structure_and_reference_information_survives(self):
        ir = ir_fixture()
        original = rule_sequence(ir)
        rebound = deepcopy(ir)
        rebound['constraints'][-2]['symbols'] = [5,2]
        self.assertNotEqual(rule_sequence(rebound), original)
        self.assertEqual(len(rule_sequence(rebound)), len(original))
        self.assertIn(('op:Input', None), original)  # Arbitrary subtree, not Scan.
        changed_shape = deepcopy(ir)
        changed_shape['source']['children'][0]['op'] = 'Proj*'
        self.assertNotEqual(rule_sequence(changed_shape), original)
        self.assertLess(original.index(('source', None)), original.index(('target', None)))
        for reference in (-1, True, 10):
            invalid = deepcopy(ir)
            invalid['constraints'][0]['symbols'] = [reference]
            with self.assertRaises(ValueError):
                rule_sequence(invalid)
        invalid = deepcopy(ir)
        invalid['source']['symbols'] = [1,0]
        with self.assertRaises(ValueError):
            rule_sequence(invalid)

    def test_masks_distinguish_zero_unknown_and_relative_ndv(self):
        rows, _ = catalog_sequences(catalog_fixture())
        self.assertIn(('estimated_rows', None), rows[0])
        self.assertIn(('estimated_rows', 0), rows[1])
        self.assertIn(('ndv_absolute', None), rows[0])
        self.assertIn(('ndv_fraction', .5), rows[0])
        vocab = fit_vocabulary([{'relation': rows}])
        missing = encode_sequence([('estimated_rows', None)], vocab)
        zero = encode_sequence([('estimated_rows', 0)], vocab)
        self.assertEqual(missing['tokens'], zero['tokens'])
        self.assertEqual(missing['numbers'], zero['numbers'])
        self.assertNotEqual(missing['known_number'], zero['known_number'])
        for value in (float('nan'), float('inf'), -2, True):
            bad = catalog_fixture()
            bad['statistics'][0]['n_distinct'] = value
            with self.assertRaises(ValueError):
                catalog_sequences(bad)
        with self.assertRaises(ValueError):
            number([], 'rows', -1, lower=0)

    def test_catalog_identity_is_only_a_join_key(self):
        catalog = catalog_fixture()
        original = catalog_sequences(catalog)
        renamed = deepcopy(catalog)
        for relation in renamed['relations']:
            relation['oid'] += '00'
            relation['name'] = 'renamed'
        for field in ('columns', 'statistics', 'constraints'):
            for record in renamed[field]:
                record['relation_oid'] += '00'
                record['name'] = 'other'
        renamed['constraints'][0]['referenced_relation_oid'] += '00'
        self.assertEqual(catalog_sequences(renamed), original)
        permuted = deepcopy(catalog)
        permuted['relations'].reverse()
        rows, edges = catalog_sequences(permuted)
        self.assertEqual(rows, list(reversed(original[0])))
        self.assertEqual((edges[0]['source'], edges[0]['target']), (1,0))
        catalog['constraints'].append({**catalog['constraints'][0], 'validated': False})
        before = catalog_sequences(catalog)
        catalog['constraints'].reverse()
        self.assertEqual(catalog_sequences(catalog)[0], before[0])

    def test_policy_membership_budget_and_vocabulary_freeze(self):
        enabled = {'enabled': True, 'placement': 'cbo', 'budget': {'per_rule': 0}}
        sequence = policy_sequence(enabled)
        self.assertIn(('budget_unbounded:per_rule', 1), sequence)
        self.assertNotEqual(sequence, policy_sequence({**enabled, 'budget': {'per_rule': 1}}))
        self.assertNotEqual(policy_sequence(None), policy_sequence({'enabled': False}))
        renamed = {**enabled, 'rule_hash': 'secret', 'source_line': 99}
        self.assertEqual(sequence, policy_sequence(renamed))
        vocabulary = fit_vocabulary([{'policy': [sequence]}])
        before = dict(vocabulary)
        encoded = encode_sequence([('future_operator', None)], vocabulary)
        self.assertEqual(encoded['unknown_tokens'], 1)
        self.assertEqual(encoded['tokens'], [1])
        self.assertEqual(vocabulary, before)

    def test_full_adapter_ignores_response_fields_and_preserves_positions(self):
        graph = {'nodes': [{'rule_hash': 'a'*16, 'learning_ir': ir_fixture()}],
                 'edges': [{'src_rule': 'a'*16, 'dst_rule': 'a'*16, 'evidence': 'runtime_observed',
                            'target_path': path} for path in ('r/0', 'r/1')]}
        context = {'catalog': catalog_fixture()}
        inputs = {'graph_snapshot': {'path': 'graph'}, 'catalog_snapshot': {'path': 'context'},
                  'candidate_policy': [{'rule_hash': 'a'*16, 'enabled': True}]}
        def read(snapshot):
            return json.dumps(graph if snapshot['path'] == 'graph' else context).encode()
        with patch('ml_orca.encoding.rule_policy_encoding.read_snapshot', side_effect=read):
            original = input_sequences(inputs)
            self.assertEqual(len(original['rule_edges']), 2)
            self.assertNotEqual(*original['sequences']['edge'])
            self.assertFalse(original['complete_prediction_input'])
            self.assertEqual(input_sequences({**inputs, 'future_memo': 999, 'execution_ms': 9}), original)
            graph['nodes'][0]['candidate_observations'] = 9999
            graph['edges'][0]['observations'] = 777
            self.assertEqual(input_sequences(inputs), original)
            graph['nodes'][0]['rule_hash'] = 'b'*16
            for edge in graph['edges']:
                edge.update(src_rule='b'*16, dst_rule='b'*16)
            inputs['candidate_policy'][0]['rule_hash'] = 'b'*16
            self.assertEqual(input_sequences(inputs), original)

    def test_native_static_graph_rejects_ir_drift_and_ignores_runtime_edges(self):
        native = {'schema_version': 1, 'nodes': [{'rule_hash': 'a', 'learning_ir': ir_fixture()}],
                  'edges': [{'src_rule': 'a', 'dst_rule': 'a', 'evidence': 'static_template',
                             'target_path': path, 'src_target_path': path, 'dst_source_path': 'r'}
                            for path in ('r', 'r/0')]}
        graph = deepcopy(native)
        graph['edges'].append({'evidence': 'runtime_observed', 'future': 123})
        inputs = {'graph_snapshot': {'path': 'graph'}, 'catalog_snapshot': {'path': 'context'},
                  'candidate_policy': [{'rule_hash': 'a', 'enabled': True}]}
        documents = {'graph': graph, 'static': native, 'context': {'catalog': catalog_fixture()}}
        with patch('ml_orca.encoding.rule_policy_encoding.read_snapshot', side_effect=lambda s: json.dumps(documents[s['path']]).encode()):
            result = input_sequences(inputs, {'path': 'static'})
            self.assertEqual(result['rule_graph_scope'], 'native_static_template')
            self.assertEqual(result['rule_edges'], [[0, 0], [0, 0]])
            self.assertNotEqual(*result['sequences']['edge'])
            graph['edges'] = [{'arbitrary_future_field': 1}]
            graph['nodes'][0]['candidate_observations'] = 99999
            self.assertEqual(input_sequences(inputs, {'path': 'static'}), result)
            native['edges'][0]['observations'] = 1
            with self.assertRaisesRegex(ValueError, 'without runtime fields'):
                input_sequences(inputs, {'path': 'static'})
            del native['edges'][0]['observations']
            native['edges'].append(deepcopy(native['edges'][0]))
            with self.assertRaisesRegex(ValueError, 'duplicate static position edge'):
                input_sequences(inputs, {'path': 'static'})
            native['edges'].pop()
            native['nodes'][0]['learning_ir']['source']['op'] = 'Proj*'
            with self.assertRaisesRegex(ValueError, 'different rule IR universe'):
                input_sequences(inputs, {'path': 'static'})


if __name__ == '__main__':
    unittest.main()
