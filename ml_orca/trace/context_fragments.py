"""Reassemble bounded native context messages without truncating observed trees."""

import json


class ContextFragments:
    def __init__(self):
        self.pending = {}
        self.completed = set()

    def accept(self, record):
        kind = record.get('kind')
        if kind == 'experiment_outcome':
            self.finish()
            self.completed.clear()
            return record
        if kind != 'candidate_context_fragment':
            return record
        field, identity = record.get('field'), record.get('context_id')
        part, parts, size = (record.get(k) for k in ('part', 'parts', 'total_bytes'))
        text = record.get('fragment')
        if (record.get('engine') != 'pgorca'
                or field not in ('input_context', 'binding_context', 'query_input_context', 'route_input_context') or type(identity) is not int or identity < 1
                or any(type(v) is not int for v in (part, parts, size))
                or not 0 <= part < parts <= size or record.get('encoding') != 'utf8_hex'
                or not isinstance(text, str) or not text):
            raise ValueError('invalid context fragment header')
        raw = bytes.fromhex(text)
        if raw.hex() != text:
            raise ValueError('context fragment must be canonical hex')
        key = field, identity
        if key in self.completed:
            raise ValueError('duplicate completed context fragments')
        if key not in self.pending:
            self.pending[key] = {'parts': parts, 'size': size, 'next': 0, 'bytes': bytearray()}
        state = self.pending[key]
        if (state['parts'] != parts or state['size'] != size or state['next'] != part
                or len(state['bytes']) + len(raw) > size):
            raise ValueError('conflicting or out-of-order context fragments')
        state['bytes'].extend(raw)
        state['next'] += 1
        if state['next'] != parts:
            return None
        if len(state['bytes']) != size:
            raise ValueError('context fragment byte count mismatch')
        value = json.loads(state['bytes'].decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError('context fragments must contain one JSON object')
        del self.pending[key]
        self.completed.add(key)
        return {'kind': 'candidate_context', 'engine': record.get('engine'),
                'field': field, 'context_id': identity, 'value': value,
                'transport': 'utf8_hex', 'transport_parts': parts}

    def finish(self):
        if self.pending:
            raise ValueError('incomplete context fragments')
