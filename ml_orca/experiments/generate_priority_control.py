"""Seeded CBO ordering control; preserve membership, stages and all budgets."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

from ml_orca.collect.run_workload_comparison import collect_policy_context


def shuffled_rules(rules, seed):
    if not rules or any(r['placement'] != 'cbo' for r in rules):
        raise ValueError('require a nonempty CBO-only policy')
    identities = [r['rule_hash'] for r in rules]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate rule identities')
    enabled = sorted(r['rule_hash'] for r in rules if r['enabled'])
    random.Random(seed).shuffle(enabled)
    priorities = {identity: rank + 1 for rank, identity in enumerate(enabled)}
    result = deepcopy(rules)
    for rule in result:
        if rule['enabled']:
            rule['priority'] = priorities[rule['rule_hash']]
    return result


def prioritized_rules(rules, ordered_identities):
    """Assign positive priorities to an explicit high-to-low ranking."""
    identities = [r['rule_hash'] for r in rules]
    admitted = set(identities)
    ranking = list(ordered_identities)
    if (len(set(identities)) != len(identities) or len(set(ranking)) != len(ranking)
            or any(identity not in admitted for identity in ranking)):
        raise ValueError('ranking must contain distinct admitted rule identities')
    priorities = {identity: len(ranking) - rank for rank, identity in enumerate(ranking)}
    result = deepcopy(rules)
    for rule in result:
        if rule['enabled'] and rule['rule_hash'] in priorities:
            rule['priority'] = priorities[rule['rule_hash']]
    return result


def policy_text(rules):
    # Explicit entries replace the wildcard record: never emit priority alone.
    lines = []
    for rule in rules:
        lines.append('- rule: ' + rule['rule_hash'])
        for key in ('enabled', 'placement', 'phase', 'effect', 'priority', 'order', 'fixpoint'):
            value = rule[key]
            lines.append(f'  {key}: {str(value).lower() if type(value) is bool else value}')
        lines.append('  budget:')
        for key in ('per_node', 'per_rule', 'per_query'):
            lines.append(f'    {key}: {rule["budget"][key]}')
    return '\n'.join(lines) + '\n'


def uncapped_orderings(rules, seed):
    """Keep membership and native list order; compare only CBO candidate priorities."""
    default = deepcopy(rules)
    if not default or any(r['placement'] != 'cbo' for r in default):
        raise ValueError('require a nonempty CBO-only policy')
    for rule in default:
        rule['priority'] = 0
        rule['budget'] = dict(per_node=0, per_rule=0, per_query=0)
    reverse = deepcopy(default)
    for rank, rule in enumerate(reverse, 1):
        if rule['enabled']:
            rule['priority'] = rank
    return dict(default=default, reverse=reverse, random=shuffled_rules(default, seed))


def scaled_orderings(rules, sizes, seed):
    """Outcome-blind nested membership samples; three orderings per size."""
    enabled = sorted(r['rule_hash'] for r in rules if r['enabled'])
    if (not sizes or len(set(sizes)) != len(sizes)
            or any(type(size) is not int or not 0 < size <= len(enabled) for size in sizes)):
        raise ValueError('sizes must be distinct positive enabled-rule counts')
    random.Random(seed).shuffle(enabled)
    result = {}
    for size in sorted(sizes):
        selected = set(enabled[:size])
        subset = deepcopy(rules)
        for rule in subset:
            rule['enabled'] = rule['enabled'] and rule['rule_hash'] in selected
        for arm, ordering in uncapped_orderings(subset, seed + 1).items():
            result[f's{size:03d}-{arm}'] = ordering
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-bin', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    def snapshot(path):
        row = collect_policy_context(args.audit_bin.resolve(), args.rules.resolve(), {'arm': path}, 60)['arm']
        if row['status'] != 'ok' or row['snapshot']['load']['failed']:
            raise ValueError('native policy resolution failed: ' + str(row))
        return row['snapshot']

    before = snapshot(args.policy.resolve())
    rules = shuffled_rules(before['rules'], args.seed)
    with args.output.open('x') as output:
        output.write(f'# Random ordering control, NOT learned priority. Seed: {args.seed}\n')
        output.write(policy_text(rules))
    after = snapshot(args.output.resolve())
    if after['load'] != before['load'] or after['rules'] != rules:
        raise ValueError('native roundtrip changed fields beyond the intended priorities; do not use output')
    print(json.dumps({'seed': args.seed, 'native_roundtrip_equal': True,
                      'rules': len(rules), 'enabled': sum(r['enabled'] for r in rules)}))


if __name__ == '__main__':
    main()
