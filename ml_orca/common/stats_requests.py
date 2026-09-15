"""Validate native request snapshots; this is not another configuration parser."""

import math
import re


def validate_request_snapshot(data):
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data['schema_version'] != 1
            or data.get('scope') != 'native_stats_requests_not_runtime_resolution'
            or not isinstance(data.get('experiment'), str) or not data['experiment']
            or type(data.get('discover')) is not bool
            or not isinstance(data.get('requests'), list)):
        raise ValueError('invalid native stats request snapshot structure')
    selectors = set()
    for request in data['requests']:
        if (not isinstance(request, dict)
                or not isinstance(request.get('relations'), list)
                or any(not isinstance(alias, str) or not alias for alias in request['relations'])
                or request['relations'] != sorted(set(request['relations']))
                or not isinstance(request.get('expression'), str)
                or not isinstance(request.get('operator'), str)
                or bool(request['relations']) == bool(request['expression'])
                or (request['expression'] and
                    (not re.fullmatch('[0-9a-f]{16}', request['expression']) or not request['operator']))
                or type(request.get('requested_rows')) not in (int, float)):
            raise ValueError('invalid native stats request')
        try:
            valid_rows = math.isfinite(request['requested_rows']) and request['requested_rows'] >= 1
        except OverflowError:
            valid_rows = False
        if not valid_rows:
            raise ValueError('invalid native stats requested rows')
        selector = (('relations', *request['relations']) if request['relations'] else
                    ('expression', request['operator'], request['expression']))
        if selector in selectors:
            raise ValueError('duplicate native stats request selector')
        selectors.add(selector)
    return data['requests']
