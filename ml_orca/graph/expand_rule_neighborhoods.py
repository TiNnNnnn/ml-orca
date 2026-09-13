#!/usr/bin/env python3
"""Discovery-only multi-source graph neighborhoods; not a DRO certificate.

Graph proximity proposes which rules MAY change. It never disables a rule or
requires a bridge to be part of an intervention. No outcome labels are read.
"""

import argparse
from collections import Counter, deque
from fractions import Fraction
import itertools
import json
from pathlib import Path
import re
import zlib

from ml_orca.graph.merge_rule_graph import merge_graph


def propose(graph, summaries, *, hotspots=3, max_hops=3, enumerate_limit=12):
    for value, minimum in ((hotspots, 1), (max_hops, 0), (enumerate_limit, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError('invalid hotspot/hop/enumeration limit')
    # ponytail: explicit subsets are only for tiny libraries; larger layers
    # expose exact counts, never truncate silently or allocate their power set.
    if enumerate_limit > 16:
        raise ValueError('explicit enumeration is limited to 16 rules')
    hashes = [node['rule_hash'] for node in graph['nodes']]
    if (len(set(hashes)) != len(hashes) or not hashes
            or any(not re.fullmatch(r'[0-9a-f]{16}', h) for h in hashes)):
        raise ValueError('need distinct canonical DSL rule hashes')
    known = set(hashes)
    query_counts, coverage = Counter(), {h: Counter() for h in hashes}
    seen, edges = set(), []
    complete = 0
    for summary in summaries:
        for dataset in summary.get('datasets', [summary]):
            name = dataset['dataset']
            for run in dataset['runs']:
                key = name, run['case_id']
                if key in seen:
                    raise ValueError('duplicate discovery query; do not count repeated summaries twice')
                seen.add(key)
                query_counts[name] += 1
                complete += run.get('complete') is True
                attempted = set(run.get('attempted_rules', []))
                if not attempted <= known:
                    raise ValueError('discovery references rules absent from the graph snapshot')
                for h in attempted:
                    coverage[h][name] += 1
                # Attempted/rejected bindings are evidence too, not just winners.
                edges.extend(run.get('edges', []))
    if not seen:
        raise ValueError('no discovery queries')
    merged = merge_graph(graph, edges)
    adjacency = {h: set() for h in hashes}
    for edge in merged['edges']:
        a, b = edge['src_rule'], edge['dst_rule']
        if a not in known or b not in known:
            raise ValueError('graph edge references an unknown rule')
        adjacency[a].add(b)
        adjacency[b].add(a)
    weights = {h: sum((Fraction(coverage[h][d], n) for d, n in query_counts.items()),
                      Fraction()) / len(query_counts) for h in hashes}
    ranked = sorted((h for h in hashes if weights[h] > 0), key=lambda h: (-weights[h], h))
    if not ranked:
        raise ValueError('no observed attempted rule')
    seeds = ranked[:hotspots]
    distances, parents = dict.fromkeys(seeds, 0), dict.fromkeys(seeds)
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if distances[node] == max_hops:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor not in distances:
                distances[neighbor], parents[neighbor] = distances[node] + 1, node
                queue.append(neighbor)
    layers = []
    for hop in range(max_hops + 1):
        rules = sorted(h for h, distance in distances.items() if distance <= hop)
        policies = None
        if len(rules) <= enumerate_limit:
            policies = [list(s) for k in range(len(rules) + 1)
                        for s in itertools.combinations(rules, k)]
        layers.append({'hop': hop, 'rules': rules,
                       'new_rules': sorted(h for h in rules if distances[h] == hop),
                       'policy_count_including_baseline': 1 << len(rules),
                       'disabled_subsets': policies,
                       'enumeration_complete': policies is not None})
    return {
        'schema_version': 1, 'scope': 'discovery_proposal_not_population_or_optimality_certificate',
        'background': 'frozen_CBO_configuration; only a chosen disabled subset may change',
        'hotspot_metric': 'dataset_equal_weight_observed_query_coverage; hash_breaks_ties',
        'queries': len(seen), 'complete_queries': complete,
        'dataset_query_counts': dict(sorted(query_counts.items())),
        'hotspots': seeds, 'ranking': [
            {'rule_hash': h, 'score_numerator': weights[h].numerator,
             'score_denominator': weights[h].denominator,
             'observed_queries': sum(coverage[h].values())} for h in ranked],
        'graph': merged, 'traversal': 'multi_source_BFS_on_undirected_projection',
        'distance': distances, 'parent': parents, 'layers': layers,
        'unreached_rules': sorted(known - distances.keys()),
        'warnings': ['edge_absence_is_not_independence',
                     'partial_queries_provide_positive_evidence_not_absence_evidence',
                     'graph_paths_may_span_queries_or_policies_not_one_executable_chain',
                     'freeze_candidates_before_independent_calibration',
                     'unlisted_rules_and_bridges_keep_background_configuration'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--attempt-summary', type=Path, action='append', required=True)
    parser.add_argument('--hotspots', type=int, default=3)
    parser.add_argument('--max-hops', type=int, default=3)
    parser.add_argument('--enumerate-limit', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.graph, *args.attempt_summary]
    raw = [p.read_bytes() for p in paths]
    result = propose(json.loads(raw[0]), [json.loads(b) for b in raw[1:]],
                     hotspots=args.hotspots, max_hops=args.max_hops,
                     enumerate_limit=args.enumerate_limit)
    result['input_snapshots'] = [{'path': str(p.resolve()), 'size': len(b),
                                  'crc32': f'{zlib.crc32(b):08x}'} for p, b in zip(paths, raw)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'queries': result['queries'], 'complete': result['complete_queries'],
                      'layers': [{'hop': x['hop'], 'rules': len(x['rules']),
                                  'policies': x['policy_count_including_baseline'],
                                  'enumerated': x['enumeration_complete']} for x in result['layers']]},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
