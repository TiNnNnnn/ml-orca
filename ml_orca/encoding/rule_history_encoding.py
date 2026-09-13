"""Freeze audited behavior-policy Memo observations, then admit them before prediction.

Runtime paths address instantiated expressions, NOT identically numbered DSL nodes.
The GE at a binding port carries producer provenance at consumption time; its
child alternatives need not be the producer's original generated tree. Template
correspondence remains unknown unless separately
proved; neither endpoint is silently replaced with the whole-rule template root.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from ml_orca.data.export_policy_learning_samples import export_comparison, read_snapshot
from ml_orca.encoding.group_expression_encoding import group_expression_tree, stats_timeline
from ml_orca.trace.profile_rule_candidates import binding_origin_evidence, candidate_evidence, candidate_state
from ml_orca.encoding.rule_policy_encoding import category, flag, input_sequences, number
from ml_orca.common.artifacts import artifact_snapshot


def utc(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError('history requires timezone-aware timestamps')
    return result


def freeze_run(snapshot, scenario, arm):
    result = json.loads(read_snapshot(snapshot))
    records = export_comparison(Path(snapshot['path']))
    record = next(r for r in records if r['unit']['scenario'] == scenario and r['unit']['policy'] == arm)
    if not record['admission']['feature_integrity_verified']:
        raise ValueError('history input provenance is not verified')
    run = result['policy_comparison']['scenarios'][scenario]['arms'][arm]
    # Arbitrary interventions need their own encoder; do not silently omit them.
    document = record['inputs']['stats_experiment_document']
    if document is None or document.strip() != 'experiment: rule-input-observation\ndiscover: true\ncardinalities:':
        raise ValueError('history currently requires the read-only input observation experiment')
    encoded = encode_observations(run)
    # Read twice to detect changes during parsing; never infer time from file mtime.
    read_snapshot(snapshot)
    return {'source': snapshot, 'unit': record['unit'], 'inputs': record['inputs'], **encoded}


def encode_observations(run):
    """Shared lossless rooted GE/edge encoding for both policy and corpus captures."""
    audit, origins, timeline = candidate_evidence(run), binding_origin_evidence(run), stats_timeline(run)
    if not all(a['complete'] for a in (audit, origins, timeline)):
        raise ValueError('incomplete history evidence: ' + str([a['exclusions'] for a in (audit, origins, timeline)]))
    trees, contexts, lookup, attempts, mapping, excluded = [], [], {}, {}, {}, Counter()
    for row in audit['rows']:
        state = candidate_state(row)
        try:
            tree = group_expression_tree(state)
        except (KeyError, ValueError, TypeError) as error:
            tree = None
            excluded['invalid_context:' + str(error)] += 1
        if tree is None:
            excluded['context_unavailable'] += 1
        key = json.dumps(tree, sort_keys=True)
        if key not in lookup:
            lookup[key] = len(trees)
            trees.append(tree)
        tree_index = lookup[key]
        identity = (row['rule_hash'], tree_index)
        if identity not in attempts:
            attempts[identity] = len(contexts)
            contexts.append({'rule_hash': row['rule_hash'], 'tree': tree_index, 'attempts': 0})
        contexts[attempts[identity]]['attempts'] += 1
        mapping[row['sequence']] = tree_index
    edges = []
    for edge in origins['edges']:
        tree_index = mapping[edge['dst_candidate_sequence']]
        tree = trees[tree_index]
        path = edge['dst_binding_path']
        nodes = {} if tree is None else {n['path']: i for i, n in enumerate(tree['nodes'])}
        if path not in nodes:
            excluded['edge_binding_root_unobserved'] += 1
            continue
        edges.append({k: edge[k] for k in ('src_rule', 'dst_rule', 'src_target_path',
            'dst_binding_path', 'producer_relation', 'producer_outcome')} | {
                'tree': tree_index, 'root': nodes[path]})
    return {'trees': trees, 'contexts': contexts, 'edges': edges,
            'denominators': {'attempts': len(audit['rows']), 'observed_edges': len(origins['edges']),
                             'admitted_edges': len(edges), 'exclusions': dict(excluded)}}


def attach_history(features, bundle, static_snapshot, prediction_at, allowed_queries):
    """Only admitted training/discovery queries, frozen before this prediction.

    allowed_queries maps case_id to exact SQL assigned to the history population;
    the trainer derives it from TRAIN assignments, never validation/test labels.
    The caller supplies the recorded pre-workload catalog timestamp, not 'now'.
    """
    if (bundle.get('schema_version') != 1 or bundle.get('capture') != 'audited_history_frozen'
            or utc(bundle['frozen_at_utc']) >= utc(prediction_at)
            or features.get('rule_graph_scope') != 'native_static_template'):
        raise ValueError('history must be frozen strictly before prediction and attached to native static IR')
    if 'history_runs' in features:
        raise ValueError('history already attached')
    sequences = features['sequences']
    channels = ('history_node', 'history_query', 'history_relation', 'history_policy', 'history_attempt', 'history_edge')
    added = {channel: [] for channel in channels}
    history_runs, trees, contexts, edges, receipts = [], [], [], [], []
    seen_runs = set()

    def extend(channel, values):
        start = len(added[channel])
        added[channel].extend(values)
        return [start, len(added[channel])]

    for run_index, history in enumerate(bundle['runs']):
        unit, inputs = history['unit'], history['inputs']
        run_key = json.dumps([history['source'], unit], sort_keys=True)
        if run_key in seen_runs:
            raise ValueError('duplicate historical observation run')
        seen_runs.add(run_key)
        # Comparison units use filenames; workload manifests use query stems.
        identity = unit['workload'] + ':' + Path(unit['query']).stem
        if allowed_queries.get(identity) != inputs['query_sql']:
            raise ValueError('history query is outside the assigned training population')
        # Source receipts are checked on every preparation, not used as features.
        read_snapshot(history['source'])
        context = json.loads(read_snapshot(inputs['catalog_snapshot']))
        if utc(context['catalog']['captured_at']) >= utc(bundle['frozen_at_utc']):
            raise ValueError('history catalog was not captured before freezing')
        encoded = input_sequences(inputs, static_snapshot, tree_rules=True)
        if not encoded['query_binding']['complete']:
            raise ValueError('historical query/catalog binding unavailable')
        if encoded['rule_trees'] != features['rule_trees']:
            raise ValueError('historical rule IR universe/order differs')
        graph = json.loads(read_snapshot(inputs['graph_snapshot']))
        indices = {n['rule_hash']: i for i, n in enumerate(graph['nodes'])}
        history_runs.append({'query_binding': encoded['query_binding'],
            'query_range': extend('history_query', encoded['sequences']['query']),
            'relation_range': extend('history_relation', encoded['sequences']['relation']),
            'policy_range': extend('history_policy', encoded['sequences']['policy']),
            'loaded_rules': encoded['loaded_rules']})
        start = len(trees)
        for tree in history['trees']:
            if tree is None:
                trees.append(None)
            else:
                trees.append({k: tree[k] for k in ('nodes', 'root', 'complete')} | {
                    'node_range': extend('history_node', tree['sequences'])})
        for item in history['contexts']:
            if (type(item['attempts']) is not int or item['attempts'] < 1
                    or type(item['tree']) is not int or not 0 <= item['tree'] < len(history['trees'])):
                raise ValueError('invalid historical attempt membership/count')
            rule = indices[item['rule_hash']]
            if not encoded['loaded_rules'][rule]:
                raise ValueError('attempt rule absent from behavior policy')
            tree = trees[start + item['tree']]
            sequence = []
            flag(sequence, 'history:context_available', tree is not None)
            flag(sequence, 'history:tree_complete', None if tree is None else tree['complete'])
            number(sequence, 'history:attempts', item['attempts'], lower=1, log=True)
            added['history_attempt'].append(sequence)
            contexts.append({'run': run_index, 'rule': rule, 'tree': start + item['tree'], 'count': item['attempts']})
        if (sum(c['attempts'] for c in history['contexts']) != history['denominators']['attempts']
                or len(history['edges']) != history['denominators']['admitted_edges']):
            raise ValueError('historical observation denominator mismatch')
        for edge in history['edges']:
            if type(edge['tree']) is not int or not 0 <= edge['tree'] < len(history['trees']):
                raise ValueError('invalid actual binding tree index')
            tree_index = start + edge['tree']
            tree, root = trees[tree_index], edge['root']
            if (tree is None or type(root) is not int or not 0 <= root < len(tree['nodes'])
                    or tree['nodes'][root]['path'] != edge['dst_binding_path']):
                raise ValueError('unresolved actual binding root')
            sequence = []
            for key in ('producer_relation', 'producer_outcome'):
                category(sequence, 'history:' + key, edge[key])
            for key in ('src_target_path', 'dst_binding_path'):
                path = edge[key]
                if not isinstance(path, str) or not re.fullmatch(r'r(?:/(?:0|[1-9][0-9]*))*', path):
                    raise ValueError('invalid instantiated expression path')
                number(sequence, 'history:' + key + ':length', len(path.split('/')) - 1)
                for step in path.split('/')[1:]:
                    number(sequence, 'history:' + key + ':step', int(step))
            category(sequence, 'history:port_domain', 'actual_binding_not_template')
            if any(not encoded['loaded_rules'][indices[edge[k]]] for k in ('src_rule', 'dst_rule')):
                raise ValueError('historical edge rule absent from behavior policy')
            added['history_edge'].append(sequence)
            edges.append({'rules': [indices[edge[k]] for k in ('src_rule', 'dst_rule')],
                          'run': run_index, 'tree': tree_index, 'root': root})
        receipts.append({'source': history['source'], 'unit': unit, 'denominators': history['denominators']})
    sequences.update(added)
    features.update(history_runs=history_runs, history_trees=trees, history_contexts=contexts,
                    history_edges=edges, history_admission={'capture': bundle['capture'],
                    'frozen_at_utc': bundle['frozen_at_utc'], 'prediction_at': prediction_at, 'runs': receipts})
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--scenario', type=int, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    snapshot = artifact_snapshot({'comparison': args.comparison})['comparison']
    history = freeze_run(snapshot, args.scenario, args.arm)
    bundle = {'schema_version': 1, 'capture': 'audited_history_frozen',
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': [history]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(bundle, stream, allow_nan=False)
    print(json.dumps(history['denominators']))


def reuse_history(features, prepared, prediction_at):
    """Share immutable input data only; learned representations are recomputed each step."""
    admission = prepared['history_admission']
    if (utc(admission['frozen_at_utc']) >= utc(prediction_at)
            or features.get('rule_graph_scope') != 'native_static_template'
            or features['rule_trees'] != prepared['rule_trees']
            or features['rule_hashes'] != prepared['rule_hashes']
            or any(features['sequences'][k] != prepared['sequences'][k]
                   for k in ('rule_node', 'rule_symbol', 'rule_constraint'))):
        raise ValueError('shared history time or rule IR mismatch')
    if any(k.startswith('history_') for k in features) or any(
            k.startswith('history_') for k in features['sequences']):
        raise ValueError('history already attached')
    features.update({k: v for k, v in prepared.items() if k.startswith('history_')})
    features['history_admission'] = {**admission, 'prediction_at': prediction_at}
    features['sequences'].update({k: v for k, v in prepared['sequences'].items()
                                 if k.startswith('history_')})


if __name__ == '__main__':
    main()
