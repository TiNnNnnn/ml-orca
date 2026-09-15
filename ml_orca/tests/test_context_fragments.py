"""Lossless tree transport, Unicode boundaries and fail-closed fragment admission."""

from copy import deepcopy
import json
import unittest

from ml_orca.trace.context_fragments import ContextFragments
from ml_orca.collect.run_workload_comparison import trace_records


def fragments(value, size=5, identity=1):
    raw = json.dumps(value, ensure_ascii=False).encode()
    chunks = [raw[i:i + size] for i in range(0, len(raw), size)]
    return [{'kind': 'candidate_context_fragment', 'engine': 'pgorca', 'field': 'input_context',
             'context_id': identity, 'encoding': 'utf8_hex', 'part': i, 'parts': len(chunks),
             'total_bytes': len(raw), 'fragment': chunk.hex()} for i, chunk in enumerate(chunks)]


class ContextFragmentsTest(unittest.TestCase):
    def test_query_input_transport_is_separate_and_precedes_search(self):
        value = {'schema_version': 1, 'experiment': 'q', 'input_context': {'capture': 'before_memo_initialization'}}
        pieces = [dict(r, field='query_input_context') for r in fragments(value)]
        candidate = {'kind': 'rule_candidate', 'placement': 'cbo', 'sequence': 1}
        outcome = {'kind': 'experiment_outcome', 'experiment': 'q', 'query_input_context_version': 1}
        text = lambda rows: '\n'.join('DSL_TRACE ' + json.dumps(r) for r in rows)
        rows = pieces + [candidate, outcome]
        parsed = trace_records(text(rows + rows))
        contexts = [r for r in parsed if r.get('field') == 'query_input_context']
        self.assertEqual([r['value'] for r in contexts], [value, value])
        direct = {'kind': 'candidate_context', 'field': 'query_input_context', 'context_id': 1, 'value': value}
        for invalid in ([candidate, *pieces, outcome], [outcome], [direct, direct, outcome],
                        [direct, dict(outcome, experiment='other')],
                        [{'kind': 'cost_candidate'}, direct, outcome]):
            with self.assertRaises(ValueError):
                trace_records(text(invalid))
        # Pending RBO attempts may be logged first; the CBO-only encoder checks
        # the nonzero preceding-attempt watermark separately.
        self.assertEqual(len(trace_records(text([dict(candidate, placement='rbo'), direct, outcome]))), 3)

    def test_shared_trace_reader_resolves_fragments_before_candidate_references(self):
        value = {'source_tree': {'nodes': [{'operator': '根🙂'}] * 200}, 'capture': 'before_evaluation'}
        pieces = fragments(value, size=41)
        candidate = {'kind': 'rule_candidate', 'sequence': 1, 'input_context_id': 1}
        records = pieces + [candidate, {'kind': 'experiment_outcome'}]
        text = '\n'.join('LOG: DSL_TRACE ' + json.dumps(r) for r in records)
        decoded = trace_records(text + '\n' + text)
        attempts = [r for r in decoded if r['kind'] == 'rule_candidate']
        self.assertEqual(len(attempts), 2)
        for row in attempts:
            self.assertEqual(row['input_context'], value)
            self.assertNotIn('context_resolution_errors', row)
        self.assertFalse(any(r['kind'] == 'candidate_context_fragment' for r in decoded))
        truncated = '\n'.join('DSL_TRACE ' + json.dumps(r) for r in pieces[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            trace_records(truncated)

    def test_route_input_context_is_resolved_without_candidate_expansion(self):
        value = {'source_tree': {'nodes': [{'operator': 'CLogicalSelect'}]}}
        pieces = [dict(row, field='route_input_context') for row in fragments(value)]
        route = {'kind': 'rule_route', 'route_input_context_id': 1, 'candidate_rules': 0}
        decoded = trace_records('\n'.join('DSL_TRACE ' + json.dumps(row)
                                          for row in [*pieces, route]))
        self.assertEqual(decoded[-1]['route_input_context'], value)

    def test_full_large_tree_and_multibyte_boundaries(self):
        value = {'source_tree': {'nodes': [{'operator': '根🙂', 'arity': 0}] * 4101, 'complete': True},
                 'escaped': '"\\\n\u0000'}
        pieces = fragments(value, size=2048)
        decoder = ContextFragments()
        results = [decoded for r in pieces if (decoded := decoder.accept(r)) is not None]
        decoder.finish()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['value'], value)
        self.assertGreater(results[0]['transport_parts'], 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            decoder.accept(pieces[0])
        decoder.accept({'kind': 'experiment_outcome'})
        for piece in pieces:
            last = decoder.accept(piece)
        self.assertEqual(last['value'], value)  # IDs may be reused only after query completion.

    def test_missing_reordered_conflicting_and_invalid_fragments(self):
        pieces = fragments({'rows': 3, 'text': '查询🙂'})
        decoder = ContextFragments()
        decoder.accept(pieces[0])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            decoder.finish()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            decoder.accept({'kind': 'experiment_outcome'})
        for field, value in (('part', 1), ('parts', 0), ('context_id', True),
                             ('encoding', 'unknown'), ('field', 'unknown'),
                             ('fragment', '00 '), ('fragment', 'zz')):
            bad = dict(pieces[0], **{field: value})
            with self.assertRaises(ValueError, msg=field):
                ContextFragments().accept(bad)
        decoder = ContextFragments()
        decoder.accept(pieces[0])
        bad = dict(pieces[1], total_bytes=pieces[1]['total_bytes'] + 1)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            decoder.accept(bad)
        bad = deepcopy(pieces)
        for piece in bad:
            piece['total_bytes'] += 1
        decoder = ContextFragments()
        with self.assertRaisesRegex(ValueError, 'byte count'):
            for piece in bad:
                decoder.accept(piece)
        decoder = ContextFragments()
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            for piece in fragments([]):
                decoder.accept(piece)

    def test_small_legacy_records_are_unchanged(self):
        record = {'kind': 'candidate_context', 'context_id': 1, 'field': 'input_context', 'value': {}}
        self.assertIs(ContextFragments().accept(record), record)


if __name__ == '__main__':
    unittest.main()
