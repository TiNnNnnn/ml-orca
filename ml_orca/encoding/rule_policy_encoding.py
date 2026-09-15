#!/usr/bin/env python3
"""Shared structural/numeric inputs, without rule-ID embeddings or response features."""

import argparse
import gzip
import json
import math
from pathlib import Path
import re

from ml_orca.common.artifacts import read_snapshot


def number(out, field, value, *, lower=None, upper=None, log=False):
    if value is not None:
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (lower is not None and value < lower) or (upper is not None and value > upper)):
            raise ValueError('invalid numeric feature: ' + field)
        value = math.copysign(math.log1p(abs(value)), value) if log else value
    out.append((field, value))


def category(out, field, value):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError('invalid category: ' + field)
    out.append((field + ':' + ('<missing>' if value is None else value), None))


def flag(out, field, value):
    if value is not None and type(value) is not bool:
        raise ValueError('invalid boolean feature: ' + field)
    number(out, field, None if value is None else int(value))


def rule_sequence(ir):
    """Prefix trees retain ordered children and canonical, rule-local symbol links."""
    if ir.get('schema_version') != 1:
        raise ValueError('require native learning_ir version 1')
    out, seen = [], []
    symbols = ir['symbols']
    for symbol in symbols:
        if (set(symbol) != {'kind', 'side'} or symbol['side'] not in ('source', 'target')
                or not isinstance(symbol['kind'], str) or len(symbol['kind']) != 1):
            raise ValueError('invalid typed symbol definition')

    def references(refs):
        number(out, 'symbol_slots', len(refs))
        for ref in refs:
            if type(ref) is not int or not 0 <= ref < len(symbols):
                raise ValueError('invalid symbol reference')
            if ref not in seen:
                seen.append(ref)
            symbol = symbols[ref]
            number(out, 'ref:' + symbol['kind'] + ':' + symbol['side'], ref)

    def tree(node):
        if set(node) != {'op', 'symbols', 'children'}:
            raise ValueError('unexpected native operator fields')
        category(out, 'op', node['op'])
        references(node['symbols'])
        number(out, 'children', len(node['children']))
        for child in node['children']:
            tree(child)
        out.append(('end_operator', None))

    for side in ('source', 'target'):
        out.append((side, None))
        tree(ir[side])
    number(out, 'constraints', len(ir['constraints']))
    for constraint in ir['constraints']:
        if set(constraint) != {'kind', 'symbols'}:
            raise ValueError('unexpected native constraint fields')
        category(out, 'constraint', constraint['kind'])
        references(constraint['symbols'])
    if seen != list(range(len(symbols))):
        raise ValueError('IR symbols are not encounter-order canonical')
    return out


def policy_sequence(row):
    out = []
    flag(out, 'loaded', row is not None)
    if row is None:
        return out
    for key in ('enabled', 'fixpoint'):
        flag(out, key, row.get(key))
    for key in ('placement', 'phase', 'effect', 'order', 'candidate_list'):
        category(out, key, row.get(key))
    number(out, 'priority', row.get('priority'), log=True)
    number(out, 'candidate_list_position', row.get('candidate_list_position'), lower=0, log=True)
    for key in ('per_node', 'per_rule', 'per_query'):
        value = row.get('budget', {}).get(key)
        number(out, 'budget:' + key, value, lower=0, log=True)
        flag(out, 'budget_unbounded:' + key, None if value is None else value == 0)
    return out


def rule_structure(ir):
    """Native ordered trees and constraint/symbol incidence, not a flattened DSL sentence."""
    rule_sequence(ir)  # Reuse the native version, field and canonical-reference checks.
    nodes, roots = [], {}
    symbols = ir['symbols']
    sequences = {'rule_node': [], 'rule_symbol': [], 'rule_constraint': []}
    occurrences = [[] for _ in symbols]

    def visit(node, side, path):
        children = [visit(child, side, path + '/' + str(i)) for i, child in enumerate(node['children'])]
        index = len(nodes)
        features = [(side, None), ('op:' + node['op'], None), ('children', len(children)),
                    ('symbol_slots', len(node['symbols']))]
        for slot, ref in enumerate(node['symbols']):
            symbol = symbols[ref]
            features.append(('ref:' + symbol['kind'] + ':' + symbol['side'], ref))
            occurrences[ref].append([index, slot])
        nodes.append({'side': side, 'path': path, 'children': children, 'symbols': list(node['symbols'])})
        sequences['rule_node'].append(features)
        return index

    for side in ('source', 'target'):
        roots[side] = visit(ir[side], side, 'r')
    for ref, symbol in enumerate(symbols):
        sequences['rule_symbol'].append([('symbol_kind:' + symbol['kind'], None),
                                        ('symbol_side:' + symbol['side'], None), ('symbol_ref', ref)])
    constraints = []
    for constraint in ir['constraints']:
        features = [('constraint:' + constraint['kind'], None), ('arguments', len(constraint['symbols']))]
        references = []
        for ref in constraint['symbols']:
            symbol = symbols[ref]
            references.append({'token': len(features), 'symbol': ref})
            features.append(('ref:' + symbol['kind'] + ':' + symbol['side'], ref))
        constraints.append(references)
        sequences['rule_constraint'].append(features)
    return {'nodes': nodes, 'roots': roots, 'symbol_occurrences': occurrences,
            'constraint_references': constraints, 'sequences': sequences}


def rule_root_index(structure, side, path):
    """Resolve an actual template node; never replace an unresolved path with the rule root."""
    if side not in ('source', 'target') or not isinstance(path, str) or not re.fullmatch(r'r(?:/(?:0|[1-9][0-9]*))*', path):
        raise ValueError('invalid rule endpoint path')
    index = structure['roots'][side]
    for step in path.split('/')[1:]:
        children = structure['nodes'][index]['children']
        if int(step) >= len(children):
            raise ValueError('dependency root is not a node in the declared rule template')
        index = children[int(step)]
    return index


def catalog_sequences(catalog):
    """OIDs are join keys only; relation permutation permutes rows and FK endpoints."""
    relations = catalog['relations']
    indices = {r['oid']: i for i, r in enumerate(relations)}
    if len(indices) != len(relations):
        raise ValueError('duplicate catalog relation')
    rows, foreign_keys = [], []
    for relation in relations:
        oid, out = relation['oid'], []
        category(out, 'relation_kind', relation['relkind'])
        flag(out, 'is_partition', relation.get('is_partition'))
        for field in ('estimated_rows', 'pages'):
            number(out, field, relation.get(field), lower=0, log=True)
        columns = sorted((c for c in catalog['columns'] if c['relation_oid'] == oid), key=lambda c: c['position'])
        number(out, 'columns', len(columns))
        for column in columns:
            number(out, 'column_position', column['position'], lower=1)
            category(out, 'column_type', column['type'])
            flag(out, 'not_null', column.get('not_null'))
            statistics = sorted((s for s in catalog['statistics'] if s['relation_oid'] == oid
                                 and s['position'] == column['position']), key=lambda s: s['inherited'])
            number(out, 'statistics_entries', len(statistics))
            for stats in statistics:
                flag(out, 'inherited', stats['inherited'])
                number(out, 'null_frac', stats.get('null_frac'), lower=0, upper=1)
                number(out, 'avg_width', stats.get('avg_width'), lower=0, log=True)
                number(out, 'correlation', stats.get('correlation'), lower=-1, upper=1)
                ndv = stats.get('n_distinct')
                if ndv is not None and (type(ndv) not in (int, float) or not math.isfinite(ndv)):
                    raise ValueError('invalid n_distinct')
                number(out, 'ndv_absolute', ndv if ndv is not None and ndv >= 0 else None, lower=0, log=True)
                number(out, 'ndv_fraction', -ndv if ndv is not None and ndv < 0 else None, lower=0, upper=1)
                freqs = stats.get('most_common_freqs')
                number(out, 'mcv_count', None if freqs is None else len(freqs))
                for frequency in freqs or []:
                    number(out, 'mcv_frequency', frequency, lower=0, upper=1)
        constraints = [c for c in catalog['constraints'] if c['relation_oid'] == oid]
        # Catalog names/order are not features. Column positions remain structural.
        constraints.sort(key=lambda c: (c['kind'], c.get('column_positions') or [],
                                        c.get('referenced_column_positions') or [],
                                        tuple(-1 if c.get(k) is None else int(c[k])
                                              for k in ('validated', 'deferrable', 'initially_deferred')),
                                        c.get('column_positions') is None))
        number(out, 'constraint_count', len(constraints))
        for constraint in constraints:
            category(out, 'constraint_kind', constraint['kind'])
            for key in ('validated', 'deferrable', 'initially_deferred'):
                flag(out, key, constraint.get(key))
            positions = constraint.get('column_positions')
            number(out, 'constraint_columns', None if positions is None else len(positions))
            for position in positions or []:
                number(out, 'constraint_column', position, lower=1)
            if constraint['kind'] == 'f':
                target = constraint['referenced_relation_oid']
                if target not in indices:
                    raise ValueError('foreign key target absent from catalog snapshot')
                foreign_keys.append({'source': indices[oid], 'target': indices[target],
                                     'columns': positions, 'target_columns': constraint['referenced_column_positions'],
                                     'validated': constraint['validated']})
        indexes = [i for i in catalog['indexes'] if i['relation_oid'] == oid]
        number(out, 'indexes', len(indexes))
        for index in sorted(indexes, key=lambda i: tuple(-1 if i.get(k) is None else int(i[k])
                                                       for k in ('is_unique', 'is_valid', 'is_ready'))):
            for key in ('is_unique', 'is_valid', 'is_ready'):
                flag(out, key, index.get(key))
        rows.append(out)
    return rows, foreign_keys


def input_sequences(inputs, static_graph_snapshot=None, tree_rules=False):
    """Accept only the input partition; no labels, current candidates or final plan."""
    from ml_orca.encoding.query_policy_encoding import query_context
    graph = json.loads(read_snapshot(inputs['graph_snapshot']))
    context = json.loads(read_snapshot(inputs['catalog_snapshot']))
    nodes = graph['nodes']
    indices = {n['rule_hash']: i for i, n in enumerate(nodes)}
    policies = {r['rule_hash']: r for r in inputs['candidate_policy']}
    if len(indices) != len(nodes) or len(policies) != len(inputs['candidate_policy']) or not policies.keys() <= indices.keys():
        raise ValueError('duplicate or unknown policy/graph rule')
    graph_edges = graph['edges']
    if static_graph_snapshot is not None:
        # Read the native template-only export, not a runtime graph relabelled
        # static. Its exact IR universe must match the frozen collection.
        static = json.loads(read_snapshot(static_graph_snapshot))
        ir = {n['rule_hash']: n['learning_ir'] for n in static['nodes']}
        if (static.get('schema_version') != 1 or len(ir) != len(static['nodes'])
                or ir != {n['rule_hash']: n['learning_ir'] for n in nodes}):
            raise ValueError('native static graph has a different rule IR universe')
        graph_edges = static['edges']
        allowed = {'src_rule', 'dst_rule', 'target_path', 'src_target_path', 'dst_source_path', 'evidence'}
        for edge in graph_edges:
            if (edge.get('evidence') != 'static_template' or set(edge) != allowed
                    or edge['target_path'] != edge['src_target_path'] or edge['dst_source_path'] != 'r'):
                raise ValueError('require native static template edges without runtime fields')
        keys = [tuple(edge[k] for k in sorted(allowed)) for edge in graph_edges]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate static position edge')
    edges, edge_sequences = [], []
    for edge in graph_edges:
        if edge['src_rule'] not in indices or edge['dst_rule'] not in indices:
            raise ValueError('unknown rule graph endpoint')
        edges.append([indices[edge['src_rule']], indices[edge['dst_rule']]])
        out = []
        for key in ('evidence', 'relation', 'producer_relation', 'producer_outcome', 'scheduler', 'path_kind'):
            category(out, key, edge.get(key))
        for key in ('target_path', 'src_target_path', 'dst_source_path', 'dst_binding_path'):
            path = edge.get(key)
            if path is not None and (not isinstance(path, str) or not re.fullmatch(r'r(?:/\d+)*', path)):
                raise ValueError('invalid rule edge position')
            steps = None if path is None else path.split('/')[1:]
            number(out, key + ':length', None if steps is None else len(steps))
            for step in steps or []:
                number(out, key + ':step', int(step), lower=0)
        edge_sequences.append(out)
    catalog, foreign_keys = catalog_sequences(context['catalog'])
    query = query_context(inputs.get('query_sql'), context['catalog'])
    result = {'sequences': {'rule': [rule_sequence(n['learning_ir']) for n in nodes],
                          'policy': [policy_sequence(policies.get(n['rule_hash'])) for n in nodes],
                          'relation': catalog, 'edge': edge_sequences, 'query': [query.pop('sequence')]},
            'rule_edges': edges, 'foreign_keys': foreign_keys,
            'rule_graph_scope': 'native_static_template' if static_graph_snapshot is not None else 'unadmitted_history',
            'loaded_rules': [n['rule_hash'] in policies for n in nodes],
            'query_binding': query,
            'not_encoded': ([] if query['complete'] else ['query_structure_and_predicate_to_catalog_binding']) + ['statistics_intervention',
                            'history_counts_without_behavior_policy_attribution', 'settings',
                            'check_and_index_expressions', 'collation_semantics', 'histogram_boundaries'],
            'complete_prediction_input': False}
    if tree_rules:
        structures = [rule_structure(node['learning_ir']) for node in nodes]
        for channel in ('rule_node', 'rule_symbol', 'rule_constraint'):
            result['sequences'][channel] = []
            for structure in structures:
                start = len(result['sequences'][channel])
                result['sequences'][channel].extend(structure['sequences'][channel])
                structure[channel + '_range'] = [start, len(result['sequences'][channel])]
        for structure in structures:
            del structure['sequences']
        result['rule_trees'] = structures
        # Runtime binding paths are not template paths. Until their history and
        # placeholder bindings are admitted, do not silently approximate them.
        if static_graph_snapshot is None:
            raise ValueError('root-bound tree graph requires an admitted native static snapshot')
        result['edge_roots'] = [
            [rule_root_index(structures[s], 'target', edge['src_target_path']),
             rule_root_index(structures[t], 'source', edge['dst_source_path'])]
            for (s, t), edge in zip(edges, graph_edges)]
    return result


def pre_memo_input_sequences(inputs, query_context, static_graph_snapshot):
    """Opt-in post-preprocessing CBO inputs; callers supply a verified native record.

    This later decision point is deliberately not the legacy pre-workload API.
    Current attempts, costs and final plans are neither accepted nor encoded.
    """
    from ml_orca.encoding.group_expression_encoding import query_input_tree
    if any(row.get('enabled') and row.get('placement') != 'cbo'
           for row in inputs['candidate_policy']):
        raise ValueError('pre-Memo features require a fixed CBO-only policy')
    requests = inputs.get('stats_experiment_requests')
    if requests is None:
        raise ValueError('pre-Memo features require the native request snapshot, including empty requests')
    features = input_sequences(inputs, static_graph_snapshot, tree_rules=True)
    context = json.loads(read_snapshot(inputs['catalog_snapshot']))
    catalog, _ = catalog_sequences(context['catalog'])
    oids = [str(r['oid']) for r in context['catalog']['relations']]
    if len(set(oids)) != len(oids) or any(not re.fullmatch(r'[1-9][0-9]{0,9}', oid) or int(oid) >= 2**32 for oid in oids):
        raise ValueError('invalid catalog relation identity')
    tree = query_input_tree({'query_input_contexts': [query_context]},
        catalog_features=dict(zip(oids, catalog)), stats_requests=requests)
    features['sequences']['query_input_node'] = tree.pop('sequences')
    features['query_input_tree'] = tree
    features['decision_point'] = 'post_preprocessing_pre_cbo'
    features['not_encoded'] = [k for k in features['not_encoded'] if k != 'statistics_intervention'] + [
        'indirect_statistics_propagation', *tree['not_observed']]
    return features


def fit_vocabulary(sequence_groups):
    tokens = sorted({token for groups in sequence_groups for seqs in groups.values() for seq in seqs for token, _ in seq})
    if '<pad>' in tokens or '<unknown>' in tokens:
        raise ValueError('reserved vocabulary token')
    return {token: i for i, token in enumerate(['<pad>', '<unknown>', *tokens])}


def encode_sequence(sequence, vocabulary):
    if vocabulary.get('<pad>') != 0 or vocabulary.get('<unknown>') != 1:
        raise ValueError('invalid vocabulary special tokens')
    return {'tokens': [vocabulary.get(token, 1) for token, _ in sequence],
            'numbers': [0.0 if value is None else float(value) for _, value in sequence],
            'known_number': [value is not None for _, value in sequence],
            'unknown_tokens': sum(token not in vocabulary for token, _ in sequence)}


def encode_sequence_groups(groups, vocabulary):
    """Share equal, read-only input sequences; retain every ordered occurrence.

    This is input storage only, not an embedding cache. JSON keys preserve
    numeric types and signed zero, and reject nonfinite values as the encoder does.
    """
    shared, encoded = {}, {}
    for channel, sequences in groups.items():
        encoded[channel] = []
        for sequence in sequences:
            key = json.dumps(sequence, allow_nan=False)
            if key not in shared:
                shared[key] = encode_sequence(sequence, vocabulary)
            encoded[channel].append(shared[key])
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    vocab = parser.add_mutually_exclusive_group(required=True)
    vocab.add_argument('--fit-vocabulary', type=Path, help='create from explicitly selected fitting inputs only')
    vocab.add_argument('--vocabulary', type=Path, help='reuse frozen vocabulary; unseen categories stay unknown')
    parser.add_argument('--output', type=Path, required=True, help='exclusive .jsonl.gz output')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    features = [input_sequences(record['inputs']) if record['admission']['feature_integrity_verified'] else None
                for record in records]
    if args.fit_vocabulary:
        vocabulary = fit_vocabulary([f['sequences'] for f in features if f is not None])
        with args.fit_vocabulary.open('x') as stream:
            json.dump(vocabulary, stream, indent=2)
    else:
        vocabulary = json.loads(args.vocabulary.read_text())
    with gzip.open(args.output, 'xt') as stream:
        for index, feature in enumerate(features):
            if feature is not None:
                feature['sequences'] = {key: [encode_sequence(seq, vocabulary) for seq in seqs]
                                        for key, seqs in feature['sequences'].items()}
            stream.write(json.dumps({'record_index': index, 'features': feature,
                                     'admission': records[index]['admission']}, allow_nan=False) + '\n')
    print(json.dumps({'records': len(records), 'encoded': sum(f is not None for f in features),
                      'vocabulary': len(vocabulary), 'complete_prediction_input': False}))


if __name__ == '__main__':
    main()
