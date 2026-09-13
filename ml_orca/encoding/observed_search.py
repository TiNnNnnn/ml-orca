"""Current observed graph features, with an explicit retrospective boundary."""
import math
import gzip
import json
from ml_orca.common.artifacts import read_snapshot
from ml_orca.encoding.query_policy_encoding import query_context
from ml_orca.encoding.rule_policy_encoding import category, number, input_sequences


class ObservedFeatureBuilder:
    """Parameter-free preparation shared by training, prefetch workers and profiling."""

    def __init__(self, snapshots, rule_indices):
        self.snapshots, self.rule_indices = snapshots, rule_indices
        self.catalogs, self.bases = {}, {}

    def __call__(self, item, raw=None):
        graph = json.loads(gzip.decompress(read_snapshot(item['graph_snapshot']) if raw is None else raw))
        if graph['case'] != item['case']:
            raise ValueError('graph/query identity mismatch')
        app = item['case']['dataset']
        snapshot = graph['source_files']['context']
        key = 'catalog:' + app
        if key in self.snapshots and self.snapshots[key] != snapshot:
            raise ValueError('observed catalog differs from frozen application context')
        if app not in self.catalogs:
            context = json.loads(read_snapshot(snapshot))
            if not context['capture_input_endpoints_equal'] or context['status'] != 'ok':
                raise ValueError('invalid catalog capture')
            policy = context['resolved_policies']['behavior']
            if policy['status'] != 'ok':
                raise ValueError('unresolved policy')
            static = self.snapshots['graph']
            self.bases[app] = input_sequences({'query_sql': item['case']['query'],
                'graph_snapshot': static, 'catalog_snapshot': snapshot,
                'candidate_policy': policy['snapshot']['rules']}, static, tree_rules=True)
            self.catalogs[app] = context
            self.snapshots[key] = snapshot
        binding = query_context(item['case']['query'], self.catalogs[app]['catalog'])
        return observed_features(self.bases[app], graph, binding, self.rule_indices)

def observed_features(base, graph, binding, rule_indices):
    """No timing/outcome labels enter inputs. Current search state is explicitly retrospective."""
    if not binding['complete']:
        raise ValueError('query_binding_unavailable')
    feature = {**base, 'query_binding': binding, 'observation_scope': 'retrospective_current_trace'}
    sequences = {k: v for k, v in base['sequences'].items() if k != 'rule'}
    sequences['query'] = [binding['sequence']]
    sequences.update(history_query=sequences['query'], history_relation=sequences['relation'],
                     history_policy=sequences['policy'], history_node=[], history_attempt=[], history_edge=[])
    trees = []
    for tree in graph['trees']:
        if tree is None:
            raise ValueError('context_unavailable')
        start = len(sequences['history_node'])
        sequences['history_node'].extend(tree['sequences'])
        trees.append({k: tree[k] for k in ('nodes', 'root', 'complete')} |
                     {'node_range': [start, len(sequences['history_node'])]})
    contexts = []
    for item in graph['contexts']:
        rule = rule_indices[item['rule_hash']]
        if not base['loaded_rules'][rule]:
            raise ValueError('observed_rule_not_loaded')
        contexts.append({'run': 0, 'rule': rule, 'tree': item['tree'], 'count': item['attempts']})
        sequences['history_attempt'].append([('observed_attempts_log1p', math.log1p(item['attempts']))])
    edges = []
    for edge in graph['edges']:
        sequence = []
        category(sequence, 'observed:relation', edge['producer_relation'])
        # Producer outcome/status is deliberately not encoded. Runtime paths are
        # actual binding addresses, never guessed source/target DSL node indices.
        for key in ('src_target_path', 'dst_binding_path'):
            steps = edge[key].split('/')[1:]
            number(sequence, key + ':length', len(steps))
            for step in steps:
                number(sequence, key + ':step', int(step), lower=0)
        sequences['history_edge'].append(sequence)
        edges.append({'rules': [rule_indices[edge[k]] for k in ('src_rule', 'dst_rule')],
                      'run': 0, 'tree': edge['tree'], 'root': edge['root']})
    feature.update(sequences=sequences, history_trees=trees, history_contexts=contexts,
                   history_edges=edges, history_runs=[{'query_binding': binding,
                       'query_range': [0, 1], 'relation_range': [0, len(sequences['relation'])],
                       'policy_range': [0, len(sequences['policy'])], 'loaded_rules': base['loaded_rules']}])
    return feature
