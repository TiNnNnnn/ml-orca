"""Lossless tree transport, Unicode boundaries and fail-closed fragment admission."""

from copy import deepcopy
import json
import unittest

from ml_orca.trace.context_fragments import ContextFragments


def fragments(value, size=5, identity=1):
    raw = json.dumps(value, ensure_ascii=False).encode()
    chunks = [raw[i:i + size] for i in range(0, len(raw), size)]
    return [{'kind': 'candidate_context_fragment', 'engine': 'pgorca', 'field': 'input_context',
             'context_id': identity, 'encoding': 'utf8_hex', 'part': i, 'parts': len(chunks),
             'total_bytes': len(raw), 'fragment': chunk.hex()} for i, chunk in enumerate(chunks)]


class ContextFragmentsTest(unittest.TestCase):
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
