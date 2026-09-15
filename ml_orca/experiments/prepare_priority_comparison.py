"""Freeze uncapped CBO ordering controls and selected real corpus SQL, without a server."""
import argparse
import json
from pathlib import Path

from ml_orca.common.artifacts import artifact_snapshot
from ml_orca.collect.run_workload_comparison import collect_policy_context, collect_stats_requests
from ml_orca.experiments.generate_priority_control import policy_text, scaled_orderings, uncapped_orderings


def policy_roundtrip_equal(expected, actual):
    fields = ('rule_hash', 'source_line', 'enabled', 'placement', 'phase', 'effect',
              'priority', 'order', 'fixpoint', 'budget')
    if [[row.get(k) for k in fields] for row in expected] != [[row.get(k) for k in fields] for row in actual]:
        return False
    positions = [row['candidate_list_position'] for row in actual if row['enabled']]
    return (positions == list(range(len(positions)))
            and all(row['candidate_list'] == 'cbo' for row in actual if row['enabled'])
            and all(row['candidate_list'] is None and row['candidate_list_position'] is None
                    for row in actual if not row['enabled']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-bin', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--base-policy', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--source-schema', type=Path, required=True)
    parser.add_argument('--query', type=int, action='append', required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--size', type=int, action='append', default=[],
                        help='nested enabled-rule count; repeat for a scale experiment')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ml_orca.data.import_wetune_workloads import detect_dialect, mysql_schema, postgres_schema
    queries = (args.corpus / 'cases.sql').read_text().splitlines()
    if len(set(args.query)) != len(args.query) or any(not 0 < i <= len(queries) or
            not queries[i-1].strip() or queries[i-1].lstrip().startswith('--') for i in args.query):
        parser.error('query ids must be distinct existing nonempty corpus SQL lines')
    before = collect_policy_context(args.audit_bin, args.rules, {'base': args.base_policy}, 60)['base']
    if before['status'] != 'ok':
        raise ValueError('native base policy failed: ' + str(before))
    policies = (scaled_orderings(before['snapshot']['rules'], args.size, args.seed)
                if args.size else uncapped_orderings(before['snapshot']['rules'], args.seed))
    schema = args.source_schema.read_text()
    schema = mysql_schema(schema) if detect_dialect(schema) == 'mysql' else postgres_schema(schema)
    args.output.mkdir(parents=True, exist_ok=False)
    workload = args.output / 'workload' / args.corpus.name
    (workload / 'sql').mkdir(parents=True)
    (workload / 'schema.sql').write_text(schema)
    for index in args.query:
        (workload / 'sql' / f'{index}.sql').write_text(queries[index-1].rstrip(';') + ';\n')
    paths = {}
    for arm, rules in policies.items():
        paths[arm] = args.output / (arm + '.policy')
        paths[arm].write_text(policy_text(rules))
    after = collect_policy_context(args.audit_bin, args.rules, paths, 60)
    for arm, resolved in after.items():
        if (resolved['status'] != 'ok' or not policy_roundtrip_equal(policies[arm], resolved['snapshot']['rules'])
                or resolved['snapshot']['load'] != before['snapshot']['load']):
            raise ValueError('native ordering roundtrip failed: ' + arm)
    observe = args.output / 'observe.yaml'
    observe.write_text('experiment: priority-uncapped-observe\ndiscover: true\ncardinalities:\n')
    requests = collect_stats_requests(args.audit_bin, [observe], 60)
    if requests['0']['status'] != 'ok' or requests['0']['snapshot']['requests']:
        raise ValueError('observation-only native preflight failed: ' + str(requests))
    identities = artifact_snapshot({**paths, 'observe': observe, 'rules': args.rules,
        'base_policy': args.base_policy, 'source_schema': args.source_schema, 'fixture': args.fixture,
        'source_sql': args.corpus / 'cases.sql', 'schema': workload / 'schema.sql',
        'audit': args.audit_bin, 'prepare': Path(__file__),
        **{f'query:{i}': workload / 'sql' / f'{i}.sql' for i in args.query}})
    if any('error' in r for r in identities.values()):
        raise ValueError('cannot freeze inputs: ' + str(identities))
    manifest = dict(scope='uncapped_local_DSL_candidate_order_not_global_Cascades_promise',
        workload=args.corpus.name, queries=args.query, seed=args.seed, identities=identities,
        sizes=sorted(args.size), membership='seeded_nested_prefix_independent_of_query_outcomes',
        policies=after, stats_preflight=requests, heldout_accessed=False, model_updates=0)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    first = next(iter(policies.values()))
    print(json.dumps(dict(queries=len(args.query), policies=len(policies), rules=len(first),
        enabled_counts=sorted({sum(r['enabled'] for r in rows) for rows in policies.values()}),
        native_roundtrip=True)))


if __name__ == '__main__':
    main()
