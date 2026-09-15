#!/usr/bin/env python3
"""Encode the observed pre-evaluation Memo input, never a future plan or match result."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

from ml_orca.trace.profile_rule_candidates import candidate_evidence, candidate_state, input_context_features
from ml_orca.encoding.rule_policy_encoding import category, flag, number
from ml_orca.common.stats_requests import validate_request_snapshot


def group_expression_sequence(state):
    """Partial observation only: IDs/provenance and after-evaluation bindings are excluded."""
    out = []
    flag(out, 'ge:captured', state['captured'])
    if not state['captured']:
        return out
    if state.get('capture') not in ('before_evaluation', 'before_memo_initialization'):
        raise ValueError('GroupExpression features must precede evaluation')
    features = state['features']

    def count(field, value):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('invalid count: ' + field)
        number(out, 'ge:' + field, value, lower=0, log=True)

    def node(value):
        category(out, 'ge:operator', value.get('operator'))
        count('arity', value.get('arity'))
        category(out, 'ge:stats_source', value.get('stats_source'))
        flag(out, 'ge:rows_available', value['rows_available'])
        number(out, 'ge:rows', value['rows'] if value['rows_available'] else None, lower=0, log=True)
        flag(out, 'ge:empty', value.get('empty'))
        count('memo_group_expressions', value.get('memo_group_expressions'))
        for field in ('group_explored', 'group_implemented', 'expression_explored', 'expression_implemented'):
            flag(out, 'ge:' + field, (value.get('memo_state') or {}).get(field))
        properties = value.get('logical_properties')
        flag(out, 'ge:properties_available', properties is not None)
        for field in ('output_columns', 'outer_columns', 'not_null_columns', 'key_count', 'join_depth'):
            count(field, (properties or {}).get(field))

    out.append(('ge:root', None))
    node(features['root'])
    children = features['children']
    positions = [c['position'] for c in children]
    if (any(type(p) is not int or p < 0 for p in positions)
            or positions != sorted(set(positions))):
        raise ValueError('invalid child positions')
    for field in ('relational_children', 'omitted_children'):
        count(field, features.get(field))
    total, omitted = features.get('relational_children'), features.get('omitted_children')
    if total is not None and omitted is not None and len(children) + omitted != total:
        raise ValueError('inconsistent child coverage')
    for child in children:
        number(out, 'ge:child_position', child['position'], lower=0)
        node(child['node'])
    shape = features.get('source_shape') or {}
    flag(out, 'ge:shape_complete', shape.get('complete'))
    for field in ('nodes', 'scalar_nodes', 'pattern_nodes', 'depth'):
        count('shape:' + field, shape.get(field))
    for operator, occurrences in sorted(shape.get('operators', {}).items()):
        category(out, 'ge:shape_operator', operator)
        count('shape_occurrences', occurrences)
    tree = features.get('source_tree')
    flag(out, 'ge:tree_available', tree is not None)
    flag(out, 'ge:tree_complete', None if tree is None else tree['complete'])
    if tree is not None:
        slots = 1
        if not tree['nodes']:
            raise ValueError('empty tree prefix')
        for entry in tree['nodes']:
            if slots <= 0:
                raise ValueError('extra node after completed tree')
            category(out, 'ge:tree_operator', entry['operator'])
            count('tree_arity', entry['arity'])
            if entry['arity'] is None:
                raise ValueError('missing tree arity')
            slots += entry['arity'] - 1
        if type(tree['complete']) is not bool or tree['complete'] != (slots == 0):
            raise ValueError('inconsistent tree coverage')
    return out


def attempt_samples(run):
    audit = candidate_evidence(run)
    records = []
    for row in audit['rows']:
        state = candidate_state(row)
        errors = list(audit['exclusions'])
        try:
            sequence = group_expression_sequence(state)
        except (KeyError, TypeError, ValueError) as error:
            sequence = None
            errors.append(str(error))
        if not state['captured']:
            errors.append('pre_evaluation_context_unavailable')
        records.append({
            'schema_version': 1, 'unit': 'cbo_dispatched_rule_attempt',
            'provenance': state['provenance'],
            'inputs': {'group_expression_sequence': sequence},
            'response': {k: row.get(k) for k in ('status', 'failed_constraint', 'match_us',
                'constraint_us', 'instantiate_us', 'direct_insertions', 'memo_outcome')},
            'admission': {'trace_complete': audit['complete'], 'feature_exclusions': errors,
                'model_training_eligible': False,
                'scope': 'partial_local_observation_not_whole_policy_pre_run_input'}})
    return records


def group_expression_tree(state, catalog_features=None, stats_requests=None):
    """Observed prefix -> ordered postorder tree; omitted children stay unknown.

    Only root/direct relational children have cached statistics in trace v1.
    Deeper nodes receive explicit missing attributes, never table/future rows.
    Optional same-database pre-workload catalog features remain a separate channel;
    relation OIDs join that snapshot but never become numeric model inputs.
    Optional native request ordinals join pre-workload configuration, not outcomes.
    Only directly resolved operators are observed, not all indirect Memo effects.
    """
    group_expression_sequence(state)  # Validate counts, prefix and capture boundary.
    features = state['features']
    tree = features.get('source_tree') if state['captured'] else None
    if tree is None:
        return None
    requests = None if stats_requests is None else validate_request_snapshot(stats_requests)
    binding = tree.get('request_binding')
    if requests is not None and binding not in (None, 'resolved_operator_only'):
        raise ValueError('unknown stats request binding scope')
    observed = {'r': features['root']}
    observed.update({'r/' + str(c['position']): c['node'] for c in features['children']})
    preorder, stack = [], []
    for entry in tree['nodes']:
        while stack and len(preorder[stack[-1]]['children']) == preorder[stack[-1]]['arity']:
            stack.pop()
        path = 'r' if not stack else preorder[stack[-1]]['path'] + '/' + str(len(preorder[stack[-1]]['children']))
        index = len(preorder)
        if stack:
            preorder[stack[-1]]['children'].append(index)
        preorder.append({**entry, 'path': path, 'children': []})
        stack.append(index)
    # Reverse preorder is child-before-parent; child lists keep their original order.
    remap = {old: len(preorder) - 1 - old for old in range(len(preorder))}
    nodes, sequences = [], []
    for old in reversed(range(len(preorder))):
        node = preorder[old]
        attrs = observed.get(node['path'])
        if attrs is not None and (attrs['operator'] != node['operator'] or attrs['arity'] != node['arity']):
            raise ValueError('cached node attributes disagree with source tree')
        attrs = attrs or {'operator': node['operator'], 'arity': node['arity'], 'stats_source': 'unobserved',
                          'rows_available': False, 'rows': None}
        local = {'captured': True, 'capture': state['capture'], 'features': {
            'root': attrs, 'children': [], 'relational_children': None, 'omitted_children': None}}
        sequence = group_expression_sequence(local)
        if requests is not None:
            index = node.get('request_index')
            known = binding == 'resolved_operator_only' and 'request_index' in node
            if index is not None and (not known or type(index) is not int or not 0 <= index < len(requests)):
                raise ValueError('invalid stats request index')
            request = None if index is None else requests[index]
            if request is not None and request['expression'] and request['operator'] != node['operator']:
                raise ValueError('stats request operator disagrees with source tree')
            flag(sequence, 'ge:request_binding_observed', known)
            flag(sequence, 'ge:request_direct_target', index is not None if known else None)
            number(sequence, 'ge:requested_rows', None if request is None else request['requested_rows'], lower=1, log=True)
            if request is not None:
                category(sequence, 'ge:request_source', 'declared_config_resolved_operator')
        if catalog_features is not None:
            oid = node.get('relation_oid')
            if oid is not None and (type(oid) is not int or not 0 < oid < 2**32):
                raise ValueError('invalid catalog relation identity')
            catalog = catalog_features.get(str(oid)) if oid is not None else None
            flag(sequence, 'ge:catalog_identity_observed', 'relation_oid' in node)
            flag(sequence, 'ge:catalog_relation_present', oid is not None)
            flag(sequence, 'ge:catalog_available', catalog is not None)
            if catalog is not None:
                category(sequence, 'ge:catalog_source', 'pre_workload_catalog')
                sequence.extend(('ge:catalog:' + field, value) for field, value in catalog)
        flag(sequence, 'ge:node_attributes_observed', node['path'] in observed)
        number(sequence, 'ge:unobserved_child_slots', node['arity'] - len(node['children']), lower=0)
        sequences.append(sequence)
        nodes.append({'path': node['path'], 'children': [remap[c] for c in node['children']]})
    result = {'nodes': nodes, 'root': len(nodes) - 1, 'complete': tree['complete'], 'sequences': sequences}
    if state['capture'] == 'before_memo_initialization':
        result['capture'] = state['capture']
    return result


def query_input_tree(run, *, catalog_features=None, stats_requests=None):
    """Audited pre-Memo whole query, not a future candidate or an RBO policy input.

    Its operators are post-preprocessing. This CBO-only gate deliberately rejects
    preceding DSL attempts, since that input could depend on an RBO policy.
    """
    records = run.get('query_input_contexts', [])
    outcomes = run.get('experiment_outcomes', [])
    if len(records) != 1 or len(outcomes) > 1:
        raise ValueError('missing or ambiguous pre-Memo query input')
    record = records[0]
    if outcomes and (type(outcomes[0].get('query_input_context_version')) is not int or
                     outcomes[0]['query_input_context_version'] != 1 or
                     record.get('experiment') != outcomes[0].get('experiment')):
        raise ValueError('mismatched pre-Memo query outcome')
    if (type(record.get('schema_version')) is not int or record['schema_version'] != 1 or
            not isinstance(record.get('experiment'), str) or not record['experiment'] or
            any(type(record.get(k)) is not int or record[k] != 0 for k in
                ('preceding_rule_candidates', 'preceding_cost_candidates', 'preceding_search_checks'))):
        raise ValueError('query input does not precede the CBO-only experiment')
    context = record.get('input_context') or {}
    if (context.get('capture') != 'before_memo_initialization' or
            context.get('scope') != 'query_after_preprocessing' or
            type(context.get('stats_lifecycle_sequence')) is not int or context['stats_lifecycle_sequence'] != 0):
        raise ValueError('invalid pre-Memo query capture boundary')
    observed = [context.get('root', {}), *[child['node'] for child in context.get('children', [])]]
    if any(node.get('memo_state') is not None or node.get('memo_group_expressions') is not None or
           node.get('logical_properties') is not None or node.get('stats_source') == 'memo_group'
           for node in observed):
        raise ValueError('pre-Memo query input contains Memo state')
    if stats_requests is not None:
        validate_request_snapshot(stats_requests)
        if stats_requests['experiment'] != record['experiment']:
            raise ValueError('request snapshot does not belong to query input')
    state = {'captured': True, 'capture': 'before_memo_initialization',
             'features': input_context_features(context)}
    tree = group_expression_tree(state, catalog_features, stats_requests)
    if tree is None or tree['complete'] is not True:
        raise ValueError('incomplete pre-Memo query tree')
    if stats_requests is not None:
        source = context['source_tree']
        if source.get('request_binding') != 'resolved_operator_only' or any('request_index' not in n for n in source['nodes']):
            raise ValueError('missing pre-Memo query request binding')
        # Initial resolution must account for every declared target, even if
        # later policy-dependent attempt trees never visit that part of query.
        if {n['request_index'] for n in source['nodes'] if n['request_index'] is not None} != set(range(len(stats_requests['requests']))):
            raise ValueError('query tree does not cover all declared requests')
    tree['scope'] = 'pre_memo_query_root_expression'
    tree['not_observed'] = ['external_cte_producer_definitions', 'required_physical_properties',
                            'full_scalar_semantics']
    return tree


def stats_timeline(run):
    """Join cache-write history by the captured watermark, never backfill attempt inputs."""
    audit = candidate_evidence(run)
    errors = list(audit['exclusions'])
    finals = run.get('experiment_outcomes', [])
    final = finals[0] if len(finals) == 1 else {}
    events = run.get('stats_lifecycle_events', [])
    count = final.get('stats_lifecycle_events')
    if final.get('stats_lifecycle_version') != 1:
        errors.append('stats_lifecycle_version_missing')
    if type(count) is not int or count != len(events):
        errors.append('stats_lifecycle_count_mismatch')
    by_group, links = defaultdict(list), []
    previous_rules = previous_costs = 0
    for index, event in enumerate(events, 1):
        valid = (event.get('sequence') == index and type(event.get('sequence')) is int
                 and type(event.get('group')) is int and event['group'] >= 0
                 and event.get('experiment') == final.get('experiment')
                 and event.get('scope') == 'group_cache_after_write')
        rules, costs = event.get('preceding_rule_candidates'), event.get('preceding_cost_candidates')
        valid = valid and (type(rules) is int and type(costs) is int
            and type(final.get('rule_candidates')) is int and type(final.get('cost_candidates')) is int
            and previous_rules <= rules <= final.get('rule_candidates', -1)
            and previous_costs <= costs <= final.get('cost_candidates', -1))
        rows, empty, status = event.get('rows'), event.get('empty'), event.get('status')
        valid = valid and ((status == 'reset' and rows is None and empty is None)
            or (status == 'available' and type(empty) is bool and (rows is None or
                (type(rows) in (int, float) and math.isfinite(rows) and rows >= 0))))
        if not valid:
            errors.append('invalid_stats_lifecycle_event')
            continue
        previous_rules, previous_costs = rules, costs
        by_group[event['group']].append(event)
    sequences = {group: [e['sequence'] for e in history] for group, history in by_group.items()}

    def valid_watermark(cutoff, sequence, counter):
        if type(cutoff) is not int or not 0 <= cutoff <= len(events) or type(sequence) is not int:
            return False
        # Events counted in the snapshot must precede the current attempt/cost event.
        before = events[cutoff - 1].get(counter) if cutoff else 0
        after = events[cutoff].get(counter) if cutoff < len(events) else sequence
        return type(before) is int and type(after) is int and before < sequence and after >= sequence - 1

    for row in audit['rows']:
        state = candidate_state(row)
        if not state['captured']:
            continue
        cutoff = state['provenance']['stats_lifecycle_sequence']
        if not valid_watermark(cutoff, row['sequence'], 'preceding_rule_candidates'):
            errors.append('missing_or_invalid_pre_evaluation_stats_watermark')
            continue
        slots = [('root', state['provenance']['root_memo'], state['features']['root'])]
        slots += [(str(c['position']), memo, c['node']) for c, memo in zip(
            state['features']['children'], state['provenance']['child_memo'])]
        for slot, memo, node in slots:
            owner = memo.get('statistics_owner_group')
            history = by_group.get(owner, [])
            position = bisect_right(sequences.get(owner, []), cutoff)
            prior = history[position - 1] if position else None
            later = history[position] if position < len(history) else None
            if prior and node['stats_source'] == 'memo_group' and node['rows_available']:
                if prior['status'] != 'available' or prior['rows'] != node['rows']:
                    errors.append('cached_rows_disagree_with_prior_event')
            links.append({'candidate_sequence': row['sequence'], 'slot': slot,
                'statistics_owner_group': owner, 'capture_stats_sequence': cutoff,
                'prior_event_sequence': None if prior is None else prior['sequence'],
                'next_event_sequence': None if later is None else later['sequence'],
                'observed_stats_source': node['stats_source'], 'observed_rows': node['rows']})
    for event in run.get('cost_events', []):
        cutoff = event.get('stats_lifecycle_sequence')
        if not valid_watermark(cutoff, event.get('sequence'), 'preceding_cost_candidates'):
            errors.append('missing_or_invalid_cost_stats_watermark')
    return {'scope': 'group_cache_history_not_counterfactual_rule_effect_or_future_input',
            'complete': not errors, 'exclusions': sorted(set(errors)), 'events': events, 'links': links,
            'not_observed': ['expression_private_stats', 'in_place_histogram_mutations',
                             'execution_actual_rows', 'causal_attribution_to_one_rule']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='complete-policy comparison.json')
    parser.add_argument('--output', type=Path, required=True, help='exclusive .jsonl.gz')
    args = parser.parse_args()
    result = json.loads(args.input.read_bytes())
    profile = result['policy_comparison']
    counters = Counter()
    runs = []
    with gzip.open(args.output, 'xt') as stream:
        for index, scenario in enumerate(profile['scenarios']):
            for arm in profile['arms']:
                run = scenario['arms'][arm]
                observed = scenario['stats_experiment'] is not None
                records = attempt_samples(run) if observed else []
                audit = candidate_evidence(run) if observed else {
                    'complete': None, 'exclusions': ['observation_not_requested']}
                # Empty/failed runs are retained in the stream, not lost behind zero attempts.
                summary = {'kind': 'run', 'scenario': index, 'policy': arm,
                    'comparison_path': str(args.input.resolve()), 'attempts': len(records),
                    'observation_requested': observed,
                    'trace_complete': audit['complete'], 'exclusions': audit['exclusions']}
                runs.append(summary)
                stream.write(json.dumps(summary, allow_nan=False) + '\n')
                if observed:
                    stream.write(json.dumps({'kind': 'stats_timeline', 'scenario': index, 'policy': arm,
                        'observation': stats_timeline(run)}, allow_nan=False) + '\n')
                for record in records:
                    record.update(kind='attempt', scenario=index, policy=arm)
                    counters['attempts'] += 1
                    counters['with_context'] += not record['admission']['feature_exclusions']
                    counters[record['response']['status']] += 1
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps({'runs': runs, 'counts': counters, 'model_trained': False}))


if __name__ == '__main__':
    main()
