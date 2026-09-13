#!/usr/bin/env python3
"""Count usable real-query graphs, not edges or repeats; keep every assigned failure."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from ml_orca.common.artifacts import read_snapshot
from ml_orca.encoding.query_policy_encoding import query_context
from ml_orca.common.artifacts import artifact_snapshot


def graph_exclusions(graph, loaded):
    errors = set(graph['denominators']['exclusions'])
    contexts, edges, trees = graph['contexts'], graph['edges'], graph['trees']
    counts = graph['denominators']
    if (any(type(counts.get(k)) is not int or counts[k] < 0
            for k in ('attempts', 'observed_edges', 'admitted_edges'))
            or counts['attempts'] < 1
            or counts['observed_edges'] < counts['admitted_edges']
            or counts['admitted_edges'] != len(edges)):
        errors.add('invalid_graph_denominator')
    total = 0
    for item in contexts:
        if (type(item['attempts']) is not int or item['attempts'] < 1
                or item['rule_hash'] not in loaded or type(item['tree']) is not int
                or not 0 <= item['tree'] < len(trees)):
            errors.add('invalid_context_membership')
        else:
            total += item['attempts']
            if trees[item['tree']] is None:
                errors.add('context_unavailable')
    if total != counts['attempts']:
        errors.add('context_attempt_count_mismatch')
    for edge in edges:
        if (any(edge[k] not in loaded for k in ('src_rule', 'dst_rule'))
                or type(edge['tree']) is not int or not 0 <= edge['tree'] < len(trees)):
            errors.add('invalid_edge_membership')
            continue
        tree, root = trees[edge['tree']], edge['root']
        if (tree is None or type(root) is not int or not 0 <= root < len(tree['nodes'])
                or tree['nodes'][root]['path'] != edge['dst_binding_path']):
            errors.add('unresolved_actual_binding_root')
    return sorted(errors)


def audit_corpus(root, progress=False):
    manifest = json.loads((root / 'manifest.json').read_bytes())
    if manifest.get('history_split') not in ('train', 'validation', 'test'):
        raise ValueError('require a preregistered deduplicated history partition')
    final_path = root / 'summary.json'
    try:
        final = json.loads(final_path.read_bytes()) if final_path.exists() else None
    except json.JSONDecodeError:
        if not progress:
            raise
        final = None  # The running collector may still be writing its final report.
    finalized = final is not None and final.get('collection_artifacts_unchanged') is True
    if not progress and not finalized:
        raise ValueError('collection is unfinished or artifact endpoints changed')
    if finalized:
        for snapshot in manifest['artifact_start'].values():
            read_snapshot(snapshot)
    assigned = {c['case_id']: c for d in manifest['datasets'] for c in d['cases']}
    if len(assigned) != sum(len(d['cases']) for d in manifest['datasets']):
        raise ValueError('duplicate assigned query identity')
    query_keys = [(c['dataset'], c['query']) for c in assigned.values()]
    if len(set(query_keys)) != len(query_keys):
        raise ValueError('duplicate SQL within a catalog cannot count as another query')
    verified, catalogs, rows = set(), {}, []

    def verify(snapshot):
        key = (snapshot['path'], snapshot['size'], snapshot['crc32'])
        if key not in verified:
            read_snapshot(snapshot)
            verified.add(key)

    for identity, case in assigned.items():
        app, stem = identity.split(':')
        result_path = root / app / f'{stem}.summary.json'
        row = {'case_id': identity, 'eligible': False, 'exclusions': ['summary_pending']}
        rows.append(row)
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_bytes())
        except json.JSONDecodeError:
            if not progress:
                raise
            continue
        row['exclusions'] = list(result['exclusions'])
        if not result['complete']:
            if not row['exclusions']:
                row['exclusions'].append('incomplete_trace')
            continue
        if result.get('stats_timeline_complete') is not True or not result.get('graph'):
            row['exclusions'].append('missing_audited_graph')
            continue
        graph_path = Path(result['graph'])
        with gzip.open(graph_path, 'rt') as stream:
            graph = json.load(stream)
        if (graph['case'] != case or graph['schema_version'] != 1
                or graph['scope'] != 'audited_planning_history_not_execution_labels'):
            raise ValueError('graph differs from assigned query: ' + identity)
        sources = graph['source_files']
        if (Path(sources['manifest']['path']).resolve() != (root / 'manifest.json').resolve()
                or Path(sources['context']['path']).resolve() != (root / app / 'pre-workload-context.json').resolve()):
            raise ValueError('graph receipt belongs to a different collection/catalog')
        for snapshot in sources.values():
            verify(snapshot)
        if app not in catalogs:
            catalogs[app] = json.loads(read_snapshot(sources['context']))
        context = catalogs[app]
        policy = context['resolved_policies']['behavior']
        if (context['status'] != 'ok' or not context['capture_input_endpoints_equal']
                or policy['status'] != 'ok'):
            raise ValueError('invalid pre-query catalog/policy capture')
        before, frozen = (datetime.fromisoformat(value) for value in
                          (context['catalog']['captured_at'], graph['frozen_at_utc']))
        if before.utcoffset() is None or frozen.utcoffset() is None or before >= frozen:
            raise ValueError('history capture order is not established')
        loaded = {r['rule_hash'] for r in policy['snapshot']['rules']}
        row['exclusions'].extend(graph_exclusions(graph, loaded))
        binding = query_context(case['query'], context['catalog'])
        if not binding['complete']:
            row['exclusions'].append('query_catalog_binding_unavailable')
            row['binding_errors'] = binding['errors']
        row.update(eligible=not row['exclusions'], dynamic_edges=len(graph['edges']),
                   attempts=graph['denominators']['attempts'],
                   partial_trees=sum(t is not None and not t['complete'] for t in graph['trees']),
                   graph_snapshot=artifact_snapshot({'graph': graph_path})['graph'])
    usable = [r for r in rows if r['eligible']]
    dynamic = sum(r['dynamic_edges'] > 0 for r in usable)
    return {'schema_version': 1, 'scope': 'query_graph_admission_not_execution_labels_or_model_benefit',
            'collection_finalized': finalized, 'progress_only': progress,
            'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
            'assigned_queries': len(assigned), 'eligible_query_graphs': len(usable),
            'eligible_graphs_with_dynamic_edges': dynamic,
            'eligible_graphs_with_partial_trees': sum(r['partial_trees'] > 0 for r in usable),
            'minimum_training_graphs_met': not progress and finalized
                and manifest['history_split'] == 'train' and dynamic >= 2000,
            'minimum_scope': '2000_distinct_queries_with_observed_dependency_edges_and_resolved_roots',
            'exclusion_counts': dict(Counter(e for r in rows for e in r['exclusions'])),
            'queries': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--progress', action='store_true', help='unfinished counts only, never training admission')
    args = parser.parse_args()
    report = audit_corpus(args.corpus, args.progress)
    with args.output.open('x') as stream:
        json.dump(report, stream, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('queries', 'exclusion_counts')}))


if __name__ == '__main__':
    main()
