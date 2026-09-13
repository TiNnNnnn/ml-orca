"""Audited training population gates, independent of network and objective."""
from collections import Counter
from pathlib import Path
import json
from ml_orca.common.artifacts import read_snapshot
from ml_orca.data.export_history_corpus import iter_history_runs

def history_training_gate(history, allowed_queries, minimum):
    """Count distinct admitted dynamic query graphs, never edges or repeated captures."""
    from ml_orca.data.audit_history_corpus import graph_exclusions
    if type(minimum) is not int or minimum < 0:
        raise ValueError('invalid minimum dynamic history query count')
    if history is None:
        if minimum:
            raise ValueError('declared training population requires audited history')
        return {'minimum': 0, 'qualified_dynamic_queries': 0, 'required': False}
    queries, qualified, exclusions = set(), set(), Counter()
    for run in iter_history_runs(history):
        unit, inputs = run['unit'], run['inputs']
        identity = unit['workload'] + ':' + Path(unit['query']).stem
        sql = inputs['query_sql']
        if allowed_queries.get(identity) != sql:
            raise ValueError('history query is outside assigned training population')
        query = unit['workload'], sql
        queries.add(query)
        errors = graph_exclusions(run, {r['rule_hash'] for r in inputs['candidate_policy']})
        exclusions.update(errors)
        if not errors and run['edges']:
            qualified.add(query)
    if len(qualified) < minimum:
        raise ValueError(f'qualified dynamic history queries {len(qualified)} < declared minimum {minimum}')
    return {'minimum': minimum, 'qualified_dynamic_queries': len(qualified),
            'assigned_history_queries': len(queries), 'required': minimum > 0,
            'excluded_graph_reasons': dict(exclusions)}

def assigned_history_queries(manifest):
    """Historical training population may exceed the independently labelled subset."""
    if 'history_population' not in manifest:
        return {a['case_id']: a['query'] for a in manifest['queries'] if a['split'] == 'train'}
    cohort = json.loads(read_snapshot(manifest['history_population']))
    assigned = {a['case_id']: a for a in cohort['entries']}
    if len(assigned) != len(cohort['entries']):
        raise ValueError('duplicate historical cohort identity')
    for query in manifest['queries']:
        original = assigned.get(query['case_id'], {})
        if any(query[k] != original.get(k) for k in ('query', 'family', 'split')):
            raise ValueError('label assignment differs from preregistered history cohort')
    return {k: a['query'] for k, a in assigned.items() if a['split'] == 'train'}
